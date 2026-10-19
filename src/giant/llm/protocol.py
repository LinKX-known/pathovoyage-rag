"""定义导航动作、消息、模型响应和服务提供方接口的数据契约。"""

# 动作类型决定下一步是预筛、裁剪、评分还是回答；统一消息结构由各客户端转换。

from __future__ import annotations

from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, Field

# =============================================================================
# 动作模型（描述大语言模型的决定）。
# =============================================================================


# 结构化观察（一次动作 = 描述当前 + 导航到新）。
class BoundingBoxAction(BaseModel, frozen=True):
    """表示在原始切片坐标系中执行的裁剪动作。"""

    action_type: Literal["crop"] = "crop"
    x: int = Field(..., ge=0, description="Left edge of bounding box")  # 左边界
    y: int = Field(..., ge=0, description="Top edge of bounding box")  # 上边界
    width: int = Field(..., gt=0, description="Width of bounding box")  # 宽
    height: int = Field(..., gt=0, description="Height of bounding box")  # 高


# 诊断结论 + 置信度 + 支持证据 + 鉴别诊断（可解释）。
class DiagnosticReport(BaseModel, frozen=True):
    """保存模型生成的结构化诊断内容。"""

    diagnosis: str = Field(..., min_length=1, description="Diagnostic conclusion")  # 诊断结论
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium", description="Subjective confidence"  # 置信度
    )
    evidence: list[str] = Field(
        default_factory=list, description="Supporting pathological features"  # 支持证据
    )
    differential: list[str] = Field(
        default_factory=list, description="Differential diagnoses considered"  # 鉴别诊断
    )


class FinalAnswerAction(BaseModel, frozen=True):
    """表示结束导航并给出诊断的动作。"""

    action_type: Literal["answer"] = "answer"
    answer_text: str = Field(..., min_length=1, description="The final answer")  # 答案文本
    report: DiagnosticReport | None = Field(
        default=None,
        description="Optional structured diagnostic report",  # 结构化诊断报告（可选）
    )


HypothesisText = Annotated[str, Field(min_length=1)]


class ConchAction(BaseModel, frozen=True):
    """表示对多个候选诊断执行图文评分的动作。"""

    action_type: Literal["conch"] = "conch"
    hypotheses: list[HypothesisText] = Field(
        ...,
        min_length=1,
        description="Textual hypotheses to score against the current image",  # 候选诊断列表
    )


class SelectAction(BaseModel, frozen=True):
    """表示先执行候选区域预筛的动作。"""

    action_type: Literal["select"] = "select"
    candidates: list[tuple[int, int]] = Field(
        default_factory=list,
        description="PLIP candidate centres (Level-0 coordinates)",  # 候选框中心点
    )


Action = Annotated[
    BoundingBoxAction | FinalAnswerAction | ConchAction | SelectAction,
    Field(discriminator="action_type"),
]


# =============================================================================
# 响应模型
# =============================================================================


# 解析后返回的核心结果。
class StepResponse(BaseModel, frozen=True):
    """将模型的推理文本和本轮动作绑定。"""

    reasoning: str = Field(
        ...,
        min_length=1,
        description="Chain-of-thought reasoning for the action",  # 推理（思维链）
    )
    action: Action = Field(..., description="The action to take")


class TokenUsage(BaseModel, frozen=True):
    """记录输入、输出与合计的令牌用量。"""

    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)
    cost_usd: float = Field(..., ge=0.0)  # 成本（美元）


class LLMResponse(BaseModel, frozen=True):
    """保存模型动作、用量、费用与调用耗时。"""

    step_response: StepResponse  # 解析好的推理+动作
    usage: TokenUsage
    model: str = Field(..., description="Model identifier used for this call")  # 用的哪个模型
    latency_ms: float = Field(..., ge=0.0, description="API call latency in ms")  # 耗时


# =============================================================================
# 消息模型（大语言模型的输入）。
# =============================================================================


class MessageContent(BaseModel, frozen=True):
    """表示一段文本或一张图像。"""

    type: Literal["text", "image"]  # 内容类型：文本 / 图片
    text: str | None = None
    image_base64: str | None = None
    media_type: str = Field(
        default="image/jpeg",
        description="MIME type for image content",
    )


class Message(BaseModel, frozen=True):
    """保存消息角色及其多模态内容。"""

    role: Literal["system", "user", "assistant"]  # 谁发的
    content: list[MessageContent] = Field(..., min_length=1)  # 内容块列表


# =============================================================================
# 大语言模型服务提供方协议
# =============================================================================


class LLMProvider(Protocol):
    """约定导航智能体调用模型时需要的方法。"""


    async def generate_response(self, messages: list[Message]) -> LLMResponse:

        ...


    def get_provider_name(self) -> str | None:

        ...


    def get_model_name(self) -> str:

        ...


    def get_target_size(self) -> int:

        ...


# =============================================================================
# 异常类型
# =============================================================================


class LLMError(Exception):
    """模型服务调用失败的基础异常。"""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        cause: Exception | None = None,
    ) -> None:

        self.provider = provider
        self.model = model
        self.cause = cause
        super().__init__(message)


class LLMParseError(LLMError):
    """模型响应无法解析为有效导航动作时抛出。"""

    def __init__(
        self,
        message: str,
        *,
        raw_output: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        usage: TokenUsage | None = None,
    ) -> None:

        self.raw_output = raw_output
        self.usage = usage
        super().__init__(message, provider=provider, model=model)


class CircuitBreakerOpenError(LLMError):
    """熔断器拒绝继续调用模型时抛出。"""

    def __init__(
        self,
        message: str,
        *,
        cooldown_remaining_seconds: float,
        provider: str | None = None,
    ) -> None:

        self.cooldown_remaining_seconds = cooldown_remaining_seconds
        super().__init__(message, provider=provider)
