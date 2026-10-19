"""执行全切片导航循环，协调裁剪、专家会诊、错误恢复和最终诊断。"""

# 推理入口先打开切片并生成缩略图，随后循环组装消息、调用模型和执行动作。
# 裁剪结果可送专科模型会诊；达到步数上限时强制请求最终答案。

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from giant.agent.context import ContextManager
from giant.agent.trajectory import Trajectory
from giant.config import settings
from giant.core.crop_engine import CropEngine
from giant.geometry.overlay import (
    AxisGuideGenerator,
    OverlayService,
    OverlayStyle,
    highlight_regions,
)
from giant.geometry.primitives import Point, Region, Size
from giant.geometry.validators import GeometryValidator, ValidationError
from giant.llm.protocol import (
    BoundingBoxAction,
    ConchAction,
    FinalAnswerAction,
    LLMError,
    LLMParseError,
    LLMProvider,
    LLMResponse,
    Message,
    MessageContent,
    SelectAction,
    StepResponse,
    TokenUsage,
)
from giant.llm.patho_r1_client import PathoR1Client
from giant.utils.logging import get_logger
from giant.vision.conch import (
    ConchScorer,
    ConchUnavailableError,
    UnconfiguredConchScorer,
)
from giant.wsi.reader import WSIReader

logger = get_logger(__name__)


_PATHO_R1_MAX_RETRIES = 3

# 会诊重试仍失败时写进轨迹的占位结论。仲裁阶段要把它剔除——它不是专家的判断。
_PATHO_R1_FAILED_MARKER = "[Patho-R1 会诊失败，未获得专家判断]"


_KNOWN_ORGANS = (
    "结直肠", "回盲部", "结肠", "直肠", "阑尾", "胃", "食管", "十二指肠",
    "肺", "乳腺", "肝脏", "胆管", "胰腺", "肾脏", "膀胱", "前列腺",
    "卵巢", "子宫内膜", "宫颈", "甲状腺", "淋巴结", "皮肤", "软组织",
)


def _derive_tissue_context(question: str) -> str | None:

    for organ in sorted(_KNOWN_ORGANS, key=len, reverse=True):
        if organ in question:
            return organ
    return None


def _magnification_bucket(region: Region) -> str:

    size = max(region.width, region.height)
    if size >= 8000:
        return "low"
    if size >= 4000:
        return "medium"
    return "high"


_TISSUE_SATURATION_MIN = 30
_BLANK_TISSUE_FRACTION_MAX = 0.02


def _tissue_saturation_fraction(image: Image.Image) -> float:

    sat = np.asarray(image.convert("HSV"), dtype=np.uint8)[:, :, 1]
    if sat.size == 0:
        return 0.0
    return float((sat > _TISSUE_SATURATION_MIN).mean())


def _decode_base64_image(image_base64: str) -> Image.Image:

    return Image.open(BytesIO(base64.b64decode(image_base64))).convert("RGB")


def _is_blank_crop(image_base64: str) -> bool:

    try:
        image = _decode_base64_image(image_base64)
    except Exception:
        logger.debug("Blank-crop check skipped: image could not be decoded")
        return False
    return _tissue_saturation_fraction(image) < _BLANK_TISSUE_FRACTION_MAX


def _tissue_bounds_level0(
    thumbnail: Image.Image, slide_width: int, slide_height: int
) -> tuple[int, int, int, int] | None:

    sat = np.asarray(thumbnail.convert("HSV"), dtype=np.uint8)[:, :, 1]
    mask = sat > _TISSUE_SATURATION_MIN
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    if width == 0 or height == 0:
        return None
    scale_x = slide_width / width
    scale_y = slide_height / height
    x0, x1 = np.percentile(xs, [0.5, 99.5])
    y0, y1 = np.percentile(ys, [0.5, 99.5])
    return (
        int(x0 * scale_x),
        int(y0 * scale_y),
        int(x1 * scale_x),
        int(y1 * scale_y),
    )


# =============================================================================
# 错误反馈模板
# =============================================================================

ERROR_FEEDBACK_TEMPLATE = """
Error: Your crop coordinates were invalid.
Requested region: x={x}, y={y}, width={width}, height={height}
Image bounds: width={max_width}, height={max_height}

Issues:
{issues}

Re-examine the axis guides on the thumbnail and provide valid Level-0 coordinates.
Your coordinates must satisfy:
- 0 <= x < {max_width}
- 0 <= y < {max_height}
- x + width <= {max_width}
- y + height <= {max_height}
"""


# 作用：把模型刚才给的错误坐标、切片真实边界、具体违了哪条约束，全都填进去发回给模型，让它重新看缩略图上的坐标刻度、给合法坐标。


FORCE_ANSWER_TEMPLATE = """
You have reached the maximum number of navigation steps ({max_steps}).
Based on all the regions you have examined, you MUST now provide your final answer.

Review your observations:
{observation_summary}

Question: {question}

Provide your best answer using the `answer` action.
"""


BLANK_CROP_FEEDBACK_TEMPLATE = """
Error: Your crop region contains NO tissue — it is blank background (glass).
Requested region: x={x}, y={y}, width={width}, height={height}
Image bounds: width={max_width}, height={max_height}
{tissue_hint}
Re-examine the axis guides on the thumbnail and provide Level-0 coordinates that
land INSIDE the tissue. Do not crop the empty margin around the tissue.
"""


EARLY_ANSWER_FEEDBACK_TEMPLATE = """
You attempted to answer at step {step} of {max_steps}, but answering early is
not allowed.
You MUST continue navigating and only answer on the FINAL step.

Return a StepResponse with action_type='crop' (or 'conch' if enabled).
Do NOT use action_type='answer' until step {max_steps}.
"""


# 普通模式（默认）下，模型提前回答是被允许的（提前终止，省成本）；只有固定迭代模式才需要这个模板去"拦截早答"。

# =============================================================================
# 结果模型
# =============================================================================


class RunResult(BaseModel):
    """保存推理是否成功、答案、费用和轨迹。"""

    answer: str = Field(default="", description="Final answer text")
    trajectory: Trajectory = Field(..., description="Full navigation trajectory")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens used")
    total_cost: float = Field(default=0.0, ge=0.0, description="Total cost in USD")
    success: bool = Field(default=False, description="Whether run succeeded")
    error_message: str | None = Field(default=None, description="Error description")


@dataclass(frozen=True)
class _StepDecision:
    run_result: RunResult | None = None
    forced_summary: str | None = None
    break_to_force_answer: bool = False


# =============================================================================
# 智能体配置
# =============================================================================


@dataclass(frozen=True)
class AgentConfig:
    """约束导航步数、裁剪尺寸、预算和可选专家能力。"""

    max_steps: int = field(default_factory=lambda: settings.MAX_ITERATIONS)
    max_retries: int = 3
    budget_usd: float | None = None
    thumbnail_size: int = field(default_factory=lambda: settings.THUMBNAIL_SIZE)
    force_answer_retries: int = 3
    strict_font_check: bool = False
    enable_conch: bool = False
    conch_scorer: ConchScorer | None = None
    system_prompt: str | None = None
    enforce_fixed_iterations: bool = False
    candidate_regions: list[Region] | None = None
    candidate_centers: list[tuple[int, int]] | None = None
    enable_neuro_symbolic: bool = False
    enable_patho_r1: bool = False
    tissue_context: str | None = None
    plip_plan: dict | None = None

    # 知识检索：注入式传入（与 conch_scorer 相同的做法），None 表示不启用 RAG。
    knowledge: Any | None = None

    def __post_init__(self) -> None:
        if self.budget_usd is not None and self.budget_usd < 0.0:
            raise ValueError("budget_usd must be non-negative")


# =============================================================================

# =============================================================================


@dataclass
class GIANTAgent:
    """驱动导航循环并协调切片读取、模型调用和动作执行。"""

    wsi_path: str | Path
    question: str
    llm_provider: LLMProvider
    config: AgentConfig = field(default_factory=AgentConfig)

    # 内部状态
    _context: ContextManager = field(init=False, repr=False)
    _reader: WSIReader = field(init=False, repr=False)
    _crop_engine: CropEngine = field(init=False, repr=False)
    _validator: GeometryValidator = field(init=False, repr=False)
    _overlay_service: OverlayService = field(init=False, repr=False)
    _thumbnail_base64: str = field(init=False, repr=False)
    _thumbnail_marked_base64: str = field(init=False, repr=False, default="")
    _slide_bounds: Size = field(init=False, repr=False)
    _total_tokens: int = field(init=False, default=0, repr=False)
    _total_cost: float = field(init=False, default=0.0, repr=False)
    _consecutive_errors: int = field(init=False, default=0, repr=False)
    _fatal_error_answer: str | None = field(init=False, default=None, repr=False)
    _conch_scorer: ConchScorer = field(init=False, repr=False)
    _selected: bool = field(init=False, default=False, repr=False)

    _patho_r1_client: PathoR1Client | None = field(init=False, default=None, repr=False)

    _observation_tasks: dict[str, str] = field(init=False, default_factory=dict, repr=False)

    _tissue_bounds: tuple[int, int, int, int] | None = field(
        init=False, default=None, repr=False
    )

    _planning_evidence_ids: list[str] = field(
        init=False, default_factory=list, repr=False
    )

    _gate_log: list[dict] = field(init=False, default_factory=list, repr=False)


    async def run(self) -> RunResult:
        """打开切片、初始化组件、执行导航并返回完整轨迹。"""
        wsi_path_str = str(self.wsi_path)

        # 初始化组件。
        self._validator = GeometryValidator()
        overlay_style = OverlayStyle(strict_font_check=self.config.strict_font_check) #样式配置	刻度长什么样
        self._overlay_service = OverlayService(
            generator=AxisGuideGenerator(style=overlay_style) #画线工人	怎么画出刻度
        )#协调者	怎么把刻度合成到缩略图上
        '''
        OverlayService —— 你选中的这个（协调者）
        它自己不画线，而是负责"合成"：内部持有一个 AxisGuideGenerator，核心方法 create_navigable_thumbnail(thumbnail, metadata) 做三步：


        # ① 让 generator 画一张透明的刻度图
        overlay = self.generator.generate(thumbnail_size, slide_dimensions)
        # ② 把缩略图转成 RGBA
        thumb_rgba = thumbnail.convert("RGBA")
        # ③ 用 alpha_composite 把刻度叠到缩略图上，再转回 RGB
        composited = Image.alpha_composite(thumb_rgba, overlay)
        return composited.convert("RGB")
        所以在 runner.py 里，_generate_thumbnail 调的就是这一句：


        navigable = self._overlay_service.create_navigable_thumbnail(thumbnail, metadata)
        它还有个调试用方法 create_overlay_only() —— 只生成刻度图、不合成，方便你单独看刻度长啥样。
        '''
        self._total_tokens = 0
        self._total_cost = 0.0
        self._consecutive_errors = 0
        self._fatal_error_answer = None
        self._conch_scorer = self.config.conch_scorer or UnconfiguredConchScorer()

        self._patho_r1_client = None
        patho_r1_plan_dict: dict[str, str] | None = None

        # ① 规划阶段检索：一次问答只检索一次，结果同时进入观察计划与轨迹。
        planning_evidence = self._retrieve_evidence(self.question, stage="planning")
        self._planning_evidence_ids = (
            planning_evidence.citation_ids() if planning_evidence is not None else []
        )

        if self.config.enable_patho_r1:

            # 避免判据硬编码、换题失效；生成失败回退空计划（用通用反偏误规则兜底）。
            from giant.llm.plip_planner import plan_observation

            obs = await plan_observation(self.question, evidence=planning_evidence)
            self._observation_tasks = {
                "low": obs.low_task,
                "medium": obs.medium_task,
                "high": obs.high_task,
            }
            patho_r1_plan_dict = {
                "criteria": obs.criteria,
                "low_task": obs.low_task,
                "medium_task": obs.medium_task,
                "high_task": obs.high_task,
            }
            self._patho_r1_client = PathoR1Client(
                base_url=settings.PATHO_R1_BASE_URL,
                model=settings.PATHO_R1_MODEL,
                question=self.question,
                criteria=obs.criteria,
                tissue_context=self.config.tissue_context
                or _derive_tissue_context(self.question),
            )

        try:

            with WSIReader(wsi_path_str) as reader:
                self._reader = reader
                self._crop_engine = CropEngine(reader)


                metadata = reader.get_metadata()
                self._slide_bounds = Size(width=metadata.width, height=metadata.height)


                provider_name = self._infer_provider_name()

                system_prompt = (
                    self.config.system_prompt
                    or settings.get_giant_system_prompt(provider=provider_name)
                )
                # 建上下文：把问题、步数上限、系统提示词等都存进去
                self._context = ContextManager(
                    wsi_path=wsi_path_str,
                    question=self.question,
                    max_steps=self.config.max_steps,
                    system_prompt=system_prompt,
                    enable_conch=self.config.enable_conch,
                    enforce_fixed_iterations=self.config.enforce_fixed_iterations,
                    candidate_regions=self.config.candidate_regions,
                    enable_neuro_symbolic=self.config.enable_neuro_symbolic,
                )


                self._context.trajectory.plip_plan = self.config.plip_plan
                self._context.trajectory.patho_r1_plan = patho_r1_plan_dict

                # 把规划阶段的证据写进轨迹，保证「结论—证据」可追溯。
                if planning_evidence is not None and planning_evidence.passages:
                    self._context.trajectory.rag_evidence = list(
                        planning_evidence.passages
                    )
                    self._context.trajectory.rag_corpus_version = (
                        planning_evidence.corpus_version
                    )
                self._context.trajectory.rag_gate = list(self._gate_log)


                self._thumbnail_base64 = self._generate_thumbnail(reader)

                # 把切片尺寸和缩略图写进轨迹，供可视化和评测使用

                self._context.trajectory.slide_width = self._slide_bounds.width
                self._context.trajectory.slide_height = self._slide_bounds.height
                self._context.trajectory.thumbnail_base64 = (
                    self._thumbnail_marked_base64 or self._thumbnail_base64
                )

                # 一切就绪，进入核心导航循环
                return await self._navigation_loop()

        except Exception as e:
            logger.exception("Agent run failed with exception")
            # 如果上下文已经初始化，保留部分轨迹。
            trajectory = (
                self._context.trajectory
                if hasattr(self, "_context")
                else Trajectory(wsi_path=wsi_path_str, question=self.question)
            )
            return RunResult(
                answer="",
                trajectory=trajectory,
                total_tokens=self._total_tokens,
                total_cost=self._total_cost,
                success=False,
                error_message=f"Agent failed: {e}",
            )


    def _infer_provider_name(self) -> str | None:


        get_provider_name = getattr(self.llm_provider, "get_provider_name", None)
        if callable(get_provider_name):
            provider_name = get_provider_name()
            if isinstance(provider_name, str):
                return provider_name


        name = type(self.llm_provider).__name__.lower()
        if "openai" in name:
            return "openai"
        if "anthropic" in name:
            return "anthropic"

        return None


    def _retrieve_evidence(self, query: str, *, stage: str) -> Any | None:
        """调用知识层检索，并过一道相关性闸门；任何失败都降级为「无证据」。

        闸门的作用是挡住跨器官噪声（实测：问「亚型」会命中肛管 AIN、甲状腺、卵巢）。
        判定基于问题锚点词，与检索用的 query 无关，因此仲裁阶段也用同一个问题做判定。
        """
        knowledge = self.config.knowledge
        if knowledge is None:
            return None
        try:
            bundle = knowledge.retrieve(query, stage=stage)
        except Exception as e:  # 检索失败不应让一次推理整体失败
            logger.warning("知识检索失败，跳过 RAG 注入: %s", e)
            return None

        try:
            from giant.knowledge.gate import apply as apply_gate

            gated, decision = apply_gate(bundle, self.question)
            self._gate_log.append({"stage": stage, **decision.model_dump()})
            if not decision.allows_injection:
                logger.info(
                    "检索闸门拦截：注入被跳过",
                    stage=stage,
                    relevance=decision.relevance,
                    kept=decision.kept,
                )
            return gated
        except Exception as e:  # 闸门自身异常时退回原始证据
            logger.warning("检索闸门异常，使用未过滤证据: %s", e)
            return bundle


    async def _call_llm_step(
        self, messages: list[Message]
    ) -> tuple[LLMResponse | None, str | None]:
        try:
            response = await self.llm_provider.generate_response(messages)
            self._accumulate_usage(response)
        except (LLMError, LLMParseError) as e:
            logger.warning("LLM call failed: %s", e)
            if isinstance(e, LLMParseError):
                if e.usage is not None:
                    self._accumulate_token_usage(e.usage)
            self._consecutive_errors += 1
            if self._consecutive_errors >= self.config.max_retries:
                if isinstance(e, LLMParseError) and e.raw_output is not None:
                    self._fatal_error_answer = e.raw_output
                return (
                    None,
                    f"Max retries ({self.config.max_retries}) exceeded: {e}",
                )
            return None, None

        self._consecutive_errors = 0
        return response, None


    async def _navigation_loop(self) -> RunResult:
        """逐步调用模型并处理动作，直到获得答案或耗尽步数。"""
        forced_summary: str | None = None
        final_result: RunResult | None = None

        while self._context.current_step <= self.config.max_steps:
            # 检查预算限制。
            if self._is_budget_exceeded():
                break

            # 组装消息并调用大语言模型。
            messages = self._context.get_messages(self._thumbnail_base64)

            response, fatal_error = await self._call_llm_step(messages)
            if fatal_error is not None:
                final_result = self._build_error_result(
                    fatal_error,
                    answer=self._fatal_error_answer or "",
                )
                break
            if response is None:
                continue

            decision = await self._handle_step_action(response.step_response, messages)
            if decision.forced_summary is not None:
                forced_summary = decision.forced_summary
            if decision.run_result is not None:
                final_result = decision.run_result
                break
            if decision.break_to_force_answer:
                break

        # 达到步数上限或循环结束时，强制生成最终答案。
        if final_result is not None and (final_result.success or final_result.answer):
            return final_result
        if final_result is not None:
            # 导航中途出错（无答案）：基于已观察内容再试一次强制回答，

            logger.warning(
                "Navigation ended with error; attempting forced final answer: %s",
                final_result.error_message,
            )
            summary = self._build_observation_summary()
            if final_result.error_message:
                summary = (
                    f"Navigation ended early: {final_result.error_message}\n\n{summary}"
                )
            forced = await self._force_final_answer(observation_summary=summary)
            if forced.success:
                return forced
            return final_result
        if self._is_budget_exceeded():
            return await self._handle_budget_exceeded()
        return await self._force_final_answer(observation_summary=forced_summary)


    async def _handle_step_action(
        self, step_response: StepResponse, messages: list[Message]
    ) -> _StepDecision:
        action = step_response.action  # 模型这一步决定做的动作

        decision: _StepDecision

        if (
            self.config.enable_neuro_symbolic
            and self.config.candidate_regions
            and not self._selected
            and not isinstance(action, SelectAction)
        ):
            decision = await self._handle_select_required(messages)
            return decision


        if isinstance(action, FinalAnswerAction):
            if (
                not self.config.enforce_fixed_iterations
                and not self._context.is_final_step
            ):
                decision = _StepDecision(
                    run_result=await self._handle_answer(step_response)
                )
            elif self.config.enforce_fixed_iterations and not self._context.is_final_step:
                # 固定迭代模式下，还没到最后一步就想答 → 拦截，逼它继续导航
                logger.warning(
                    "Model returned early answer at step %d/%d (fixed-iteration mode)",
                    self._context.current_step,
                    self.config.max_steps,
                )
                decision = await self._handle_early_answer_before_final_step(messages)
            else:

                logger.warning(
                    "Model returned answer at step %d/%d",
                    self._context.current_step,
                    self.config.max_steps,
                )
                decision = _StepDecision(run_result=await self._handle_answer(step_response))

        elif isinstance(action, BoundingBoxAction):
            decision = await self._handle_crop_action(step_response, action, messages)

        elif isinstance(action, ConchAction):
            if self._context.is_final_step:

                logger.warning(
                    "Model returned conch at final step %d, forcing answer.",
                    self._context.current_step,
                )
                forced_summary = self._build_forced_summary_for_final_step(
                    "invoke CONCH"
                )
                decision = _StepDecision(
                    forced_summary=forced_summary,
                    break_to_force_answer=True,
                )
            elif not self.config.enable_conch:

                decision = await self._handle_conch_disabled(messages)
            else:

                decision = await self._handle_conch_action(step_response, action)

        elif isinstance(action, SelectAction):
            if self._selected:

                logger.warning("Model repeated select; ignoring")
                decision = _StepDecision()
            else:
                self._selected = True


                real_action = SelectAction(
                    candidates=[
                        r.center for r in (self.config.candidate_regions or [])
                    ]
                )
                real_response = StepResponse(
                    reasoning=step_response.reasoning,
                    action=real_action,
                )
                self._context.add_turn(
                    image_base64=self._thumbnail_marked_base64
                    or self._thumbnail_base64,
                    response=real_response,
                    region=None,
                )
                logger.info("PLIP pre-screening (select) recorded as step 0")
                decision = _StepDecision()
        # ── 分支 4：其它情况（无法识别的动作） ──
        elif self._context.is_final_step:
            # 最后一步收到无法识别的动作 → 转强制回答
            decision = _StepDecision(break_to_force_answer=True)
        else:

            decision = _StepDecision()

        return decision

    async def _handle_select_required(
        self, messages: list[Message]
    ) -> _StepDecision:

        feedback_text = (
            "Before examining the slide, you MUST first run PLIP pre-screening by "
            "outputting the select action. This deterministically selects candidate "
            "regions to guide your navigation. Output action_type='select'."
        )
        feedback_message = Message(
            role="user",
            content=[MessageContent(type="text", text=feedback_text)],
        )

        for attempt in range(self.config.max_retries):
            messages_with_feedback = [*messages, feedback_message]
            try:
                response = await self.llm_provider.generate_response(
                    messages_with_feedback
                )
                self._accumulate_usage(response)
            except (LLMError, LLMParseError) as e:
                logger.warning(
                    "Select-required retry LLM call failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.config.max_retries,
                    e,
                )
                continue

            step_response_new = response.step_response
            if isinstance(step_response_new.action, SelectAction):
                return await self._handle_step_action(
                    step_response_new, messages_with_feedback
                )


        return _StepDecision(
            run_result=self._build_error_result(
                "Model did not run select after "
                f"{self.config.max_retries} retries"
            )
        )

    async def _handle_conch_disabled(self, messages: list[Message]) -> _StepDecision:

        feedback_text = (
            "CONCH is disabled for this run.\n"
            "You MUST NOT use action_type='conch'.\n\n"
            "Return a StepResponse with action_type='crop'. "
            "Do not answer unless you are on the final step."
        )
        feedback_message = Message(
            role="user",
            content=[MessageContent(type="text", text=feedback_text)],
        )

        for attempt in range(self.config.max_retries):
            messages_with_feedback = [*messages, feedback_message]

            try:
                response = await self.llm_provider.generate_response(
                    messages_with_feedback
                )
                self._accumulate_usage(response)
            except (LLMError, LLMParseError) as e:
                logger.warning(
                    "CONCH-disabled retry LLM call failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.config.max_retries,
                    e,
                )
                continue

            step_response = response.step_response
            action = step_response.action

            if isinstance(action, ConchAction):
                logger.warning(
                    "Model requested CONCH but enable_conch is False (attempt %d/%d).",
                    attempt + 1,
                    self.config.max_retries,
                )
                continue

            return await self._handle_step_action(step_response, messages_with_feedback)

        return _StepDecision(
            run_result=self._build_error_result(
                "Model requested CONCH but tool is disabled "
                f"after {self.config.max_retries} retries"
            )
        )


    async def _handle_early_answer_before_final_step(
        self, messages: list[Message]
    ) -> _StepDecision:

        feedback_text = EARLY_ANSWER_FEEDBACK_TEMPLATE.format(
            step=self._context.current_step,
            max_steps=self.config.max_steps,
        )
        feedback_message = Message(
            role="user",
            content=[MessageContent(type="text", text=feedback_text)],
        )

        for attempt in range(self.config.max_retries):
            messages_with_feedback = [*messages, feedback_message]

            try:
                response = await self.llm_provider.generate_response(
                    messages_with_feedback
                )
                self._accumulate_usage(response)
            except (LLMError, LLMParseError) as e:
                logger.warning(
                    "Early-answer retry LLM call failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.config.max_retries,
                    e,
                )
                continue

            step_response = response.step_response
            action = step_response.action

            if isinstance(action, FinalAnswerAction):
                logger.warning(
                    "Model still answered early at step %d/%d (attempt %d/%d)",
                    self._context.current_step,
                    self.config.max_steps,
                    attempt + 1,
                    self.config.max_retries,
                )
                continue

            return await self._handle_step_action(step_response, messages_with_feedback)

        return _StepDecision(
            run_result=self._build_error_result(
                "Early answer rejected but model continued answering early "
                f"after {self.config.max_retries} retries"
            )
        )

    def _build_forced_summary_for_final_step(self, attempted_action: str) -> str:
        base_summary = self._build_observation_summary()
        note = (
            "\n(Note: You attempted to "
            f"{attempted_action} on the final step, but the step limit has been "
            "reached. You must answer now based on previous observations.)"
        )
        return base_summary + note

    async def _handle_crop_action(
        self,
        step_response: StepResponse,
        action: BoundingBoxAction,
        messages: list[Message],
    ) -> _StepDecision:
        if self._context.is_final_step:
            logger.warning(
                "Model returned crop at final step %d, forcing answer.",
                self._context.current_step,
            )
            forced_summary = self._build_forced_summary_for_final_step("crop")
            return _StepDecision(
                forced_summary=forced_summary,
                break_to_force_answer=True,
            )

        result = await self._handle_crop(step_response, action, messages)
        if result is not None:
            return _StepDecision(run_result=result)
        return _StepDecision()


    async def _handle_conch_action(
        self,
        step_response: StepResponse,
        action: ConchAction,
    ) -> _StepDecision:
        result = await self._handle_conch(step_response, action)
        if result is not None:
            return _StepDecision(run_result=result)
        return _StepDecision()

    def _get_current_observation(self) -> tuple[str, Region | None]:
        observation_image = self._thumbnail_base64
        observation_region = None
        if self._context.trajectory.turns:
            last_turn = self._context.trajectory.turns[-1]
            observation_image = last_turn.image_base64
            observation_region = last_turn.region
        return observation_image, observation_region

    async def _consult_patho_r1(
        self,
        crop_base64: str,
        observation_task: str = "",
    ) -> str | None:

        if self._patho_r1_client is None:
            return None

        result = None
        last_error: Exception | None = None
        for attempt in range(_PATHO_R1_MAX_RETRIES):
            try:
                image_bytes = base64.b64decode(crop_base64)
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                result = await self._patho_r1_client.consult(
                    image, observation_task=observation_task
                )
                break
            except Exception as e:
                last_error = e
                logger.warning(
                    "Patho-R1 consultation attempt %d/%d failed: %s",
                    attempt + 1, _PATHO_R1_MAX_RETRIES, e,
                )
                if attempt < _PATHO_R1_MAX_RETRIES - 1:
                    await asyncio.sleep(2.0 * (attempt + 1))

        if result is None:
            # 重试仍失败：留一个可见标记，而不是静默跳过。
            logger.warning("Patho-R1 consultation failed after retries: %s", last_error)
            return _PATHO_R1_FAILED_MARKER

        return result.opinion


    async def _handle_crop(
        self,
        step_response: StepResponse,
        action: BoundingBoxAction,
        messages: list[Message],
    ) -> RunResult | None:

        region = Region(
            x=action.x,
            y=action.y,
            width=action.width,
            height=action.height,
        )

        # 校验坐标边界。
        try:
            self._validator.validate(region, self._slide_bounds, strict=True)
        except ValidationError as e:
            logger.warning("Invalid crop region: %s", e)
            return await self._handle_invalid_region(action, messages, str(e))

        # 检查是否反复探索同一区域。
        redundant, reason = self._is_redundant_crop(region)
        if redundant:
            logger.warning("Redundant crop detected: %s", reason)
            return await self._handle_invalid_region(action, messages, reason)

        # 执行裁剪。
        target_size = self.llm_provider.get_target_size()
        try:
            cropped = self._crop_engine.crop(
                region,
                target_size=target_size,
                bias=settings.OVERSAMPLING_BIAS,
            )
        except ValueError as e:
            logger.warning("Crop failed: %s", e)
            return await self._handle_invalid_region(action, messages, str(e))


        if _is_blank_crop(cropped.base64_content):
            logger.warning("Crop region contains no tissue: %s", region)
            return await self._handle_invalid_region(
                action, messages, "blank background", blank=True
            )


        await self._record_crop_turn(region, cropped.base64_content, step_response)

        logger.info(
            "Step %d: Cropped region (%d, %d, %d, %d)",
            self._context.current_step - 1,
            region.x,
            region.y,
            region.width,
            region.height,
        )

        return None

    async def _record_crop_turn(
        self,
        region: Region,
        cropped_base64: str,
        step_response: StepResponse,
    ) -> None:

        mag = _magnification_bucket(region)
        task = self._observation_tasks.get(mag, "")
        expert_opinion = await self._consult_patho_r1(cropped_base64, task)

        self._context.add_turn(
            image_base64=cropped_base64,
            response=step_response,
            region=region,
            expert_opinion=expert_opinion,
            magnification=mag,
            observation_task=task,
            evidence_ids=list(self._planning_evidence_ids),
        )


    async def _handle_conch(
        self,
        step_response: StepResponse,
        action: ConchAction,
    ) -> RunResult | None:


        observation_image, observation_region = self._get_current_observation()


        try:
            image_bytes = base64.b64decode(observation_image)
            pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            return self._build_error_result(f"Failed to decode observation image: {e}")


        try:
            scores = self._conch_scorer.score_hypotheses(
                pil_image,
                list(action.hypotheses),
            )
        except ConchUnavailableError as e:
            return self._build_error_result(str(e))
        except Exception as e:
            return self._build_error_result(f"CONCH scoring failed: {e}")

        # ④ 校验分数数量是否和候选数量一致
        if len(scores) != len(action.hypotheses):
            return self._build_error_result(
                "CONCH scorer returned a score list of unexpected length: "
                f"{len(scores)} != {len(action.hypotheses)}"
            )


        self._context.add_turn(
            image_base64=observation_image,
            response=step_response,
            region=observation_region,
            conch_scores=[float(s) for s in scores],
        )

        logger.info(
            "Step %d: CONCH scored %d hypotheses",
            self._context.current_step - 1,
            len(action.hypotheses),
        )


        return None


    async def _handle_invalid_region(
        self,
        action: BoundingBoxAction,
        messages: list[Message],
        error_detail: str,
        *,
        blank: bool = False,
    ) -> RunResult | None:

        current_action = action
        current_error = error_detail
        is_blank = blank

        for attempt in range(self.config.max_retries):
            if is_blank:
                feedback = self._blank_crop_feedback(current_action)
            else:
                feedback = ERROR_FEEDBACK_TEMPLATE.format(
                    x=current_action.x,
                    y=current_action.y,
                    width=current_action.width,
                    height=current_action.height,
                    max_width=self._slide_bounds.width,
                    max_height=self._slide_bounds.height,
                    issues=current_error,
                )

            error_message = Message(
                role="user",
                content=[MessageContent(type="text", text=feedback)],
            )
            messages_with_error = [*messages, error_message]

            try:
                response = await self.llm_provider.generate_response(
                    messages_with_error
                )
                self._accumulate_usage(response)
            except (LLMError, LLMParseError) as e:
                logger.warning(
                    "Retry LLM call failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.config.max_retries,
                    e,
                )
                continue

            new_action = response.step_response.action

            if isinstance(new_action, ConchAction):
                current_error = (
                    "CONCH is not allowed while correcting invalid crop coordinates. "
                    "Provide corrected crop coordinates."
                )
                continue

            if isinstance(new_action, FinalAnswerAction):
                if (
                    self.config.enforce_fixed_iterations
                    and not self._context.is_final_step
                ):
                    logger.warning(
                        "Model attempted early answer while correcting invalid crop "
                        "at step %d/%d",
                        self._context.current_step,
                        self.config.max_steps,
                    )
                    current_error = (
                        "You must not answer yet. Provide corrected crop coordinates."
                    )
                    continue
                self._consecutive_errors = 0
                return await self._handle_answer(response.step_response)

            if not isinstance(new_action, BoundingBoxAction):
                current_error = (
                    "Unexpected action while correcting invalid crop coordinates. "
                    "Provide corrected crop coordinates."
                )
                continue

            region = Region(
                x=new_action.x,
                y=new_action.y,
                width=new_action.width,
                height=new_action.height,
            )

            is_blank = False

            try:
                self._validator.validate(region, self._slide_bounds, strict=True)
            except ValidationError as e:
                logger.warning("Retry crop still invalid: %s", e)
                current_action = new_action
                current_error = str(e)
                continue


            redundant, reason = self._is_redundant_crop(region)
            if redundant:
                logger.warning("Retry crop redundant: %s", reason)
                current_action = new_action
                current_error = reason
                continue

            target_size = self.llm_provider.get_target_size()
            try:
                cropped = self._crop_engine.crop(
                    region,
                    target_size=target_size,
                    bias=settings.OVERSAMPLING_BIAS,
                )
            except ValueError as e:
                logger.warning("Retry crop execution failed: %s", e)
                current_action = new_action
                current_error = str(e)
                continue


            if _is_blank_crop(cropped.base64_content):
                logger.warning("Retry crop contains no tissue: %s", region)
                current_action = new_action
                current_error = "blank background"
                is_blank = True
                continue

            await self._record_crop_turn(
                region, cropped.base64_content, response.step_response
            )

            logger.info(
                "Step %d: Cropped region (%d, %d, %d, %d) after retry",
                self._context.current_step - 1,
                region.x,
                region.y,
                region.width,
                region.height,
            )

            self._consecutive_errors = 0
            return None

        return self._build_error_result(
            f"Max retries ({self.config.max_retries}) on invalid coordinates"
        )

    def _blank_crop_feedback(self, action: BoundingBoxAction) -> str:

        if self._tissue_bounds is not None:
            x0, y0, x1, y1 = self._tissue_bounds
            tissue_hint = (
                "\nThe tissue on this slide occupies approximately (Level-0 pixels):\n"
                f"  x from {x0} to {x1}, y from {y0} to {y1}\n"
                "A valid crop should sit well inside that rectangle.\n"
            )
        else:
            tissue_hint = (
                "\nPlace your crop on the stained (pink/purple) tissue, not on the "
                "white area around it.\n"
            )


        # 「组织占比 ≥80%」的窗口里挑的，一定落在组织上——直接指给模型最稳。
        if self.config.candidate_centers:
            centers = ", ".join(
                f"({x}, {y})" for x, y in self.config.candidate_centers
            )
            tissue_hint += (
                "\nThe following PLIP candidate centres are all located ON tissue — "
                f"cropping near one of them is always safe: {centers}\n"
            )

        return BLANK_CROP_FEEDBACK_TEMPLATE.format(
            x=action.x,
            y=action.y,
            width=action.width,
            height=action.height,
            max_width=self._slide_bounds.width,
            max_height=self._slide_bounds.height,
            tissue_hint=tissue_hint,
        )


    async def _handle_answer(self, step_response: StepResponse) -> RunResult:

        action = step_response.action
        if not isinstance(action, FinalAnswerAction):
            raise TypeError(f"Expected FinalAnswerAction, got {type(action).__name__}")

        # 记录答案步骤时不要附加当前观察图像：
        # 模型作答时看到的是上一张裁剪图（或缩略图），
        # 此图已经在上一步骤中展示；再次附加会使
        # 轨迹可视化出现重复图像。
        self._context.add_turn(
            image_base64="",
            response=step_response,
            region=None,
        )

        answer_text = action.answer_text


        # 图像观察会盖过专家共识（0752/1079 都是这么被带偏的）。
        if self.config.enable_patho_r1:
            arbitrated = await self._arbitrate_patho_r1(answer_text)
            if arbitrated:
                answer_text = arbitrated
                self._context.trajectory.final_answer = answer_text

        logger.info("Agent provided answer: %s", answer_text[:100])

        return RunResult(
            answer=answer_text,
            trajectory=self._context.trajectory,
            total_tokens=self._total_tokens,
            total_cost=self._total_cost,
            success=True,
            error_message=None,
        )

    def _collect_expert_opinions(self) -> list[str]:

        lines: list[str] = []
        for turn in self._context.trajectory.turns:
            opinion = (turn.expert_opinion or "").strip()
            if not opinion or opinion == _PATHO_R1_FAILED_MARKER:
                continue
            region = turn.region
            where = (
                f"区域 ({region.x}, {region.y}, {region.width}, {region.height})"
                if region is not None
                else "未标注区域"
            )
            mag = turn.magnification or "未知倍率"
            lines.append(f"- {where}·{mag}：{opinion}")
        return lines

    async def _arbitrate_patho_r1(self, original_answer: str) -> str:

        opinions = self._collect_expert_opinions()
        if not opinions:
            logger.info("Patho-R1 仲裁跳过：没有可用的专家结论")
            return ""

        from giant.knowledge.cards import citation_check
        from giant.llm.plip_planner import arbitrate_structured

        # ③ 仲裁阶段检索：用「问题 + 专家结论」联合检索，让证据贴近实际争议点。
        query = f"{self.question} {' '.join(opinions)[:400]}".strip()
        evidence = self._retrieve_evidence(query, stage="arbitration")
        if evidence is not None and evidence.passages:
            merged = {p.id: p for p in self._context.trajectory.rag_evidence}
            for passage in evidence.passages:
                merged.setdefault(passage.id, passage)
            self._context.trajectory.rag_evidence = list(merged.values())
            self._context.trajectory.rag_corpus_version = (
                evidence.corpus_version
                or self._context.trajectory.rag_corpus_version
            )

        result = await arbitrate_structured(self.question, opinions, evidence=evidence)
        arbitrated = result.answer

        citation = (
            citation_check(arbitrated, evidence) if evidence is not None else None
        )
        self._context.trajectory.patho_r1_arbitration = {
            "opinions": opinions,
            "answer": arbitrated,
            "original_answer": original_answer,
            "evidence_ids": (
                evidence.citation_ids() if evidence is not None else []
            ),
            "structured": result.structured,
            "weighted_evidence": [
                item.model_dump() for item in result.weighted_evidence
            ],
            "conflicts": result.conflicts,
            "minority": result.minority,
            "citations": result.citations,
        }
        self._context.trajectory.rag_gate = list(self._gate_log)

        if citation is not None:
            self._context.trajectory.final_citation = citation.model_dump()
            if not citation.has_citations:
                logger.warning(
                    "仲裁结论未引用任何证据，可解释性降级",
                    provided=len(citation.unused),
                )
            if citation.invalid:
                logger.warning(
                    "仲裁引用了不存在的证据编号", invalid=citation.invalid
                )
        if arbitrated:
            logger.info(
                "Patho-R1 仲裁覆盖最终答案（%d 条专家结论）", len(opinions)
            )
        return arbitrated


    async def _force_final_answer(
        self,
        *,
        observation_summary: str | None = None,
    ) -> RunResult:

        logger.info("Forcing final answer at max steps")

        # 根据轨迹构建观察摘要。
        summary = observation_summary or self._build_observation_summary()

        force_prompt = FORCE_ANSWER_TEMPLATE.format(
            max_steps=self.config.max_steps,
            observation_summary=summary,
            question=self.question,
        )

        for attempt in range(self.config.force_answer_retries):
            messages = self._context.get_messages(self._thumbnail_base64)
            force_message = Message(
                role="user",
                content=[MessageContent(type="text", text=force_prompt)],
            )
            messages_with_force = [*messages, force_message]

            try:
                response = await self.llm_provider.generate_response(
                    messages_with_force
                )
                self._accumulate_usage(response)
            except (LLMError, LLMParseError) as e:
                logger.warning("Force answer LLM call failed: %s", e)
                continue

            action = response.step_response.action

            if isinstance(action, FinalAnswerAction):
                return await self._handle_answer(response.step_response)

            # 模型仍尝试裁剪，重新请求答案。
            logger.warning(
                "Model attempted crop at force answer, attempt %d/%d",
                attempt + 1,
                self.config.force_answer_retries,
            )

        return self._build_error_result(
            f"Exceeded step limit after {self.config.force_answer_retries} retries"
        )

    async def _handle_budget_exceeded(self) -> RunResult:

        logger.warning(
            "Budget exceeded: $%.4f >= $%.4f",
            self._total_cost,
            self.config.budget_usd,
        )

        budget_note = (
            f"Budget reached: ${self._total_cost:.4f} >= ${self.config.budget_usd:.4f}."
        )
        summary = f"{budget_note}\n\n{self._build_observation_summary()}"

        # 尝试获取最终答案，并明确提示预算已用尽。
        result = await self._force_final_answer(observation_summary=summary)

        # 标记为预算耗尽导致的失败。
        return RunResult(
            answer=result.answer,
            trajectory=result.trajectory,
            total_tokens=result.total_tokens,
            total_cost=result.total_cost,
            success=False,
            error_message="Budget exceeded",
        )

    def _is_budget_exceeded(self) -> bool:

        if self.config.budget_usd is None:
            return False
        return self._total_cost >= self.config.budget_usd

    def _accumulate_usage(self, response: LLMResponse) -> None:

        self._total_tokens += response.usage.total_tokens
        self._total_cost += response.usage.cost_usd

    def _accumulate_token_usage(self, usage: TokenUsage) -> None:

        self._total_tokens += usage.total_tokens
        self._total_cost += usage.cost_usd

    def _is_redundant_crop(self, region: Region) -> tuple[bool, str]:

        for turn in self._context.trajectory.turns:
            if turn.region is None:
                continue
            past = turn.region
            inter = region.intersection(past)
            if inter is None:
                continue
            union_area = region.area + past.area - inter.area
            if union_area == 0:
                continue
            iou = inter.area / union_area
            # 允许深入放大：新区块明显更小，且位于先前区块内部。
            is_deep_zoom = (
                region.area < past.area * 0.25
                and past.x <= region.x
                and past.right >= region.right
                and past.y <= region.y
                and past.bottom >= region.bottom
            )
            if iou > 0.5 and not is_deep_zoom:
                return True, (
                    f"This crop overlaps significantly (IoU={iou:.2f}) with a previously "
                    f"examined region at ({past.x}, {past.y}, {past.width}, {past.height}). "
                    "Do not repeatedly examine the same area at similar magnification. "
                    "Explore a different candidate or an unexamined part of the slide."
                )


        recent_regions = [
            t.region
            for t in self._context.trajectory.turns
            if t.region is not None
        ][-3:]
        if len(recent_regions) >= 2 and region.area > 0:
            covered_area = 0
            for r in recent_regions:


                # 否则合法的 低倍→中倍→高倍 连续放大会被误判为重复而卡死。
                is_zoom_in = (
                    region.area < r.area
                    and r.x <= region.x
                    and r.right >= region.right
                    and r.y <= region.y
                    and r.bottom >= region.bottom
                )
                if is_zoom_in:
                    continue
                inter = region.intersection(r)
                if inter is not None:
                    covered_area += inter.area
            overlap_frac = covered_area / region.area
            if overlap_frac > 0.5:
                return True, (
                    f"This crop overlaps {overlap_frac:.0%} (cumulative) with your last "
                    "few crops. You are crawling over the same region — move to a clearly "
                    "different candidate or an unexamined part instead."
                )
        return False, ""


    # 供「强制回答」提示词回顾用。
    def _build_observation_summary(self) -> str:

        if not self._context.trajectory.turns:
            return "No observations yet."

        lines = []
        for turn in self._context.trajectory.turns:
            region_str = ""
            if turn.region:
                r = turn.region
                region_str = f" at ({r.x}, {r.y}, {r.width}, {r.height})"
            line = f"- Step {turn.step_index + 1}{region_str}: {turn.response.reasoning}"
            if turn.expert_opinion:
                line += f"\n  [Patho-R1 结论] {turn.expert_opinion}"
            lines.append(line)

        return "\n".join(lines)

    def _build_error_result(self, error_message: str, *, answer: str = "") -> RunResult:

        return RunResult(
            answer=answer,
            trajectory=self._context.trajectory,
            total_tokens=self._total_tokens,
            total_cost=self._total_cost,
            success=False,
            error_message=error_message,
        )


    # 这是模型看到的第一张图。
    def _generate_thumbnail(self, reader: WSIReader) -> str:

        max_size = (self.config.thumbnail_size, self.config.thumbnail_size)
        thumbnail = reader.get_thumbnail(max_size)
        metadata = reader.get_metadata()


        self._tissue_bounds = _tissue_bounds_level0(
            thumbnail, metadata.width, metadata.height
        )

        # 添加坐标轴参考线。
        navigable = self._overlay_service.create_navigable_thumbnail(
            thumbnail, metadata
        )

        # 生成供人工查看的标记版图像（含候选点）。
        if self.config.candidate_regions:
            navigable_marked = highlight_regions(
                navigable,
                self.config.candidate_regions,
                metadata,
                centers=self.config.candidate_centers,
            )
            self._thumbnail_marked_base64 = self._encode_image_base64(
                navigable_marked
            )
        else:
            self._thumbnail_marked_base64 = self._encode_image_base64(navigable)

        # 返回供视觉语言模型使用的无标记图像。
        return self._encode_image_base64(navigable)

    def _encode_image_base64(
        self, image: Image.Image, quality: int | None = None
    ) -> str:

        if quality is None:
            quality = settings.JPEG_QUALITY
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("ascii")
