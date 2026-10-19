"""按导航轨迹组装模型对话，并裁剪过旧图像以控制上下文长度。"""

# 消息顺序为系统指令、问题与缩略图，再按轨迹交替加入模型动作和观察结果。
# 预筛动作会追加候选区域提示；过旧的裁剪图可替换成文字占位符。

from __future__ import annotations

from dataclasses import dataclass, field

from giant.agent.trajectory import Trajectory, Turn
from giant.geometry.primitives import Region
from giant.llm.protocol import (
    BoundingBoxAction,
    ConchAction,
    FinalAnswerAction,
    Message,
    MessageContent,
    SelectAction,
    StepResponse,
)
from giant.prompts.builder import PromptBuilder


@dataclass
class ContextManager:
    """记录导航轨迹，并把历史动作与图像转换为下一轮模型消息。"""

    wsi_path: str
    question: str
    max_steps: int
    max_history_images: int | None = None
    system_prompt: str | None = None
    enable_conch: bool = False
    enforce_fixed_iterations: bool = False
    candidate_regions: list[Region] | None = None
    enable_neuro_symbolic: bool = False

    trajectory: Trajectory = field(init=False)
    _prompt_builder: PromptBuilder = field(init=False, repr=False)


    def __post_init__(self) -> None:

        self.trajectory = Trajectory(
            wsi_path=self.wsi_path,
            question=self.question,
        )
        self._prompt_builder = PromptBuilder(
            enforce_fixed_iterations=self.enforce_fixed_iterations
        )


    @property
    def current_step(self) -> int:

        count = sum(
            1
            for turn in self.trajectory.turns
            if isinstance(turn.response.action, (BoundingBoxAction, FinalAnswerAction))
        )
        return count + 1


    @property
    def is_final_step(self) -> bool:

        return self.current_step == self.max_steps


    def _format_candidate_hint(self) -> str | None:

        if not self.candidate_regions:
            return None

        centers = ", ".join(
            f"({r.x + r.width // 2}, {r.y + r.height // 2})"
            for r in self.candidate_regions
        )
        return (
            "Reference: pre-screened candidate centres (Level-0 coordinates, "
            f"ranked by PLIP suspicion score, APPROXIMATE locations): [{centers}]. "
            "These are regions that a pathology CLIP model flagged as potentially "
            "abnormal. Use them only to know WHICH area to look at — then set your "
            "own crop box by centring on the actual lesion you see in the image "
            "(with a margin of surrounding normal tissue), do NOT copy a coordinate "
            "as your box's corner or centre. Examine suspicious regions at low power "
            "(large crop) then high power (small crop) as needed. You are also free "
            "to crop anywhere else on the slide."
        )


    def add_turn(
        self,
        image_base64: str,
        response: StepResponse,
        region: Region | None = None,
        conch_scores: list[float] | None = None,
        expert_opinion: str | None = None,
        magnification: str | None = None,
        observation_task: str | None = None,
        evidence_ids: list[str] | None = None,
    ) -> None:

        turn = Turn(
            step_index=len(self.trajectory.turns),
            image_base64=image_base64,
            response=response,
            region=region,
            conch_scores=conch_scores,
            expert_opinion=expert_opinion,
            magnification=magnification,
            observation_task=observation_task,
            evidence_ids=list(evidence_ids or []),
        )
        self.trajectory.turns.append(turn)


        if isinstance(response.action, FinalAnswerAction):
            self.trajectory.final_answer = response.action.answer_text

    def get_messages(self, thumbnail_base64: str) -> list[Message]:
        """按系统、初始问题及历史动作顺序构建下一次模型调用的消息。"""
        messages: list[Message] = []


        messages.append(
            self._prompt_builder.build_system_message(
                system_prompt=self.system_prompt,
                enable_conch=self.enable_conch,
                enable_neuro_symbolic=self.enable_neuro_symbolic,
            )
        )


        messages.append(
            self._prompt_builder.build_user_message(
                question=self.question,
                step=1,
                max_steps=self.max_steps,
                context_images=[thumbnail_base64],
                candidate_hint=self._format_candidate_hint(),
            )
        )


        step = 1
        for turn in self.trajectory.turns:
            action = turn.response.action


            if isinstance(action, SelectAction):
                messages.append(self._build_assistant_message(turn))
                messages.append(self._build_select_complete_message(turn))
                continue


            messages.append(self._build_assistant_message(turn))


            if isinstance(action, FinalAnswerAction):
                break

            # 步数守卫：已经到最大步数，不再继续
            if step >= self.max_steps:
                break


            if isinstance(action, BoundingBoxAction):
                step += 1
                messages.append(
                    self._build_user_message_for_turn(
                        turn=turn,
                        step=step,
                    )
                )
                continue


            if isinstance(action, ConchAction):
                step += 1
                messages.append(
                    self._build_user_message_for_conch_turn(
                        turn=turn,
                        step=step,
                    )
                )


        if self.max_history_images is not None:
            messages = self._apply_image_pruning(messages, thumbnail_base64)

        return messages


    def _build_assistant_message(self, turn: Turn) -> Message:


        action = turn.response.action
        if hasattr(action, "action_type"):
            if action.action_type == "crop":
                action_text = (
                    f"crop(x={action.x}, y={action.y}, "
                    f"width={action.width}, height={action.height})"
                )
            elif action.action_type == "conch":
                action_text = f"conch(hypotheses={list(action.hypotheses)!r})"
            elif action.action_type == "select":
                action_text = f"select(candidates={list(action.candidates)!r})"
            else:
                action_text = f'answer("{action.answer_text}")'
        else:
            action_text = str(action)

        text = f"Reasoning: {turn.response.reasoning}\n\nAction: {action_text}"

        return Message(
            role="assistant",
            content=[MessageContent(type="text", text=text)],
        )


    def _build_select_complete_message(self, turn: Turn) -> Message:

        action = turn.response.action
        centers_text = ""
        if isinstance(action, SelectAction) and action.candidates:
            centers_text = ", ".join(f"({x}, {y})" for x, y in action.candidates)
        text = (
            "PLIP pre-screening is complete. Candidate regions to examine "
            f"(Level-0 centres, APPROXIMATE locations): [{centers_text}]. "
            "Now crop to examine them — start with the most suspicious candidate "
            "at low power (large crop), then zoom to high power (small crop). "
            "Decide your own crop box from what you see in the image (centre on the "
            "lesion, include a margin of normal tissue); do NOT copy a coordinate as "
            "your box's corner or centre."
        )
        return Message(
            role="user",
            content=[MessageContent(type="text", text=text)],
        )


    def _build_user_message_for_turn(self, turn: Turn, step: int) -> Message:


        last_region = None
        if turn.region is not None:
            r = turn.region
            last_region = f"({r.x}, {r.y}, {r.width}, {r.height})"

        msg = self._prompt_builder.build_user_message(
            question=self.question,
            step=step,
            max_steps=self.max_steps,
            context_images=[turn.image_base64],
            last_region=last_region,
        )


        # 让千问导航和最终仲裁时能看到专家的意见，而不是只靠它自己看图下结论。
        if turn.expert_opinion:
            if (
                not msg.content
                or msg.content[0].type != "text"
                or msg.content[0].text is None
            ):
                raise ValueError("Expected user message to start with text content")
            new_content: list[MessageContent] = [
                MessageContent(
                    type="text",
                    text=(
                        f"[Patho-R1 专家结论] {turn.expert_opinion}\n\n"
                        f"{msg.content[0].text}"
                    ),
                ),
                *msg.content[1:],
            ]
            msg = Message(role=msg.role, content=new_content)

        return msg


    def _build_user_message_for_conch_turn(self, turn: Turn, step: int) -> Message:
        action = turn.response.action
        if not isinstance(action, ConchAction):
            raise TypeError(f"Expected ConchAction, got {type(action).__name__}")


        if not self.enable_conch:
            conch_summary = (
                "CONCH is disabled for this run. Do not request conch; choose "
                "crop(...) or answer(...)."
            )
        else:

            conch_scores = turn.conch_scores
            if conch_scores is None or len(conch_scores) != len(action.hypotheses):
                conch_summary = "CONCH scores unavailable."
            else:
                lines = ["CONCH similarity scores (higher = more similar):"]
                for hypothesis, score in zip(
                    action.hypotheses, conch_scores, strict=True
                ):
                    lines.append(f"- {hypothesis}: {score:.4f}")
                conch_summary = "\n".join(lines)

        last_region = None
        if turn.region is not None:
            r = turn.region
            last_region = f"({r.x}, {r.y}, {r.width}, {r.height})"


        msg = self._prompt_builder.build_user_message(
            question=self.question,
            step=step,
            max_steps=self.max_steps,
            context_images=[turn.image_base64],
            last_region=last_region,
        )


        if (
            not msg.content
            or msg.content[0].type != "text"
            or msg.content[0].text is None
        ):
            raise ValueError("Expected user message to start with text content")
        new_content: list[MessageContent] = [
            MessageContent(
                type="text",
                text=f"{conch_summary}\n\n{msg.content[0].text}",
            ),
            *msg.content[1:],
        ]
        return Message(role=msg.role, content=new_content)


    # 被裁掉的图用占位文本替换，防止上下文窗口被撑爆。
    def _apply_image_pruning(
        self,
        messages: list[Message],
        thumbnail_base64: str,
    ) -> list[Message]:
        """保留缩略图和最近几张裁剪图，旧图改为文字占位符。"""
        if self.max_history_images is None:
            return messages

        user_message_indices = [
            i for i, msg in enumerate(messages) if msg.role == "user"
        ]
        if not user_message_indices:
            return messages

        # 首条用户消息始终包含缩略图，不应裁剪。
        crop_user_message_indices = user_message_indices[1:]
        if len(crop_user_message_indices) <= self.max_history_images:
            return messages

        indices_to_prune = set(crop_user_message_indices[: -self.max_history_images])

        pruned_messages: list[Message] = []
        # 用户消息中的索引从 0 开始：0 为缩略图提示，其余为裁剪图提示。
        user_msg_index = 0

        for i, msg in enumerate(messages):
            if msg.role != "user":
                pruned_messages.append(msg)
                continue

            if user_msg_index == 0:
                pruned_messages.append(msg)
                user_msg_index += 1
                continue

            if i in indices_to_prune:
                pruned_messages.append(
                    self._prune_images_from_message(msg, user_msg_index)
                )
            else:
                pruned_messages.append(msg)

            user_msg_index += 1

        return pruned_messages


    def _prune_images_from_message(self, msg: Message, step_index: int) -> Message:

        new_content: list[MessageContent] = []

        for content in msg.content:
            if content.type == "image":
                # 用占位文本替换。
                placeholder = (
                    f"[Image from Step {step_index + 1} removed to save context]"
                )
                new_content.append(MessageContent(type="text", text=placeholder))
            else:
                new_content.append(content)

        return Message(role=msg.role, content=new_content)
