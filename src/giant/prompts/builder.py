"""根据当前步骤、图像与历史区域构建发送给模型的消息。"""

from giant.llm.protocol import Message, MessageContent
from giant.prompts.templates import (
    CONCH_ACTION_PROMPT,
    FINAL_STEP_PROMPT,
    INITIAL_USER_PROMPT,
    SUBSEQUENT_USER_PROMPT,
    SUBSEQUENT_USER_PROMPT_FIXED_ITERATIONS,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_FIXED_ITERATIONS_CONSTRAINTS,
    SYSTEM_PROMPT_NEURO_SYMBOLIC,
)


class PromptBuilder:
    """将导航规则、当前步骤和图像拼成模型消息。"""

    def __init__(self, *, enforce_fixed_iterations: bool = False) -> None:
        self._enforce_fixed_iterations = enforce_fixed_iterations

    def build_system_message(
        self,
        *,
        system_prompt: str | None = None,
        enable_conch: bool = False,
        enable_neuro_symbolic: bool = False,
    ) -> Message:

        prompt = system_prompt or SYSTEM_PROMPT
        if self._enforce_fixed_iterations:
            prompt = f"{prompt}\n\n{SYSTEM_PROMPT_FIXED_ITERATIONS_CONSTRAINTS}"
        if enable_conch:
            prompt = f"{prompt}\n\n{CONCH_ACTION_PROMPT}"
        if enable_neuro_symbolic:
            prompt = f"{prompt}\n\n{SYSTEM_PROMPT_NEURO_SYMBOLIC}"

        return Message(
            role="system",
            content=[
                MessageContent(
                    type="text",
                    text=prompt,
                )
            ],
        )

    def build_user_message(
        self,
        question: str,
        step: int,
        max_steps: int,
        context_images: list[str],
        last_region: str | None = None,
        candidate_hint: str | None = None,
    ) -> Message:
        """组合问题、当前步数、上一步区域及观察图像。"""
        # 校验输入。
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if step < 1:
            raise ValueError("step must be at least 1")
        if step > max_steps:
            raise ValueError(f"step ({step}) cannot exceed max_steps ({max_steps})")

        # 构建文本内容。
        text = self._build_prompt_text(
            question, step, max_steps, last_region, candidate_hint
        )

        # 构建消息内容列表。
        content: list[MessageContent] = [
            MessageContent(type="text", text=text),
        ]

        # 添加图像。
        for image_base64 in context_images:
            content.append(
                MessageContent(
                    type="image",
                    image_base64=image_base64,
                    media_type="image/jpeg",
                )
            )

        return Message(role="user", content=content)

    def _build_prompt_text(
        self,
        question: str,
        step: int,
        max_steps: int,
        last_region: str | None,
        candidate_hint: str | None,
    ) -> str:

        remaining_crops = max_steps - step

        # 最后一步必须给出答案。
        if step == max_steps:
            return FINAL_STEP_PROMPT.format(
                question=question,
                step=step,
                max_steps=max_steps,
            )

        # 初始步骤（第 1 步）。
        if step == 1:
            text = INITIAL_USER_PROMPT.format(
                question=question,
                step=step,
                max_steps=max_steps,
                remaining_crops=remaining_crops,
            )
            if candidate_hint:
                text += f"\n\n{candidate_hint}"
            if self._enforce_fixed_iterations:
                text += (
                    "\n\nIMPORTANT: Do NOT answer yet. You must answer only on the "
                    "final step."
                )
            return text


        template = (
            SUBSEQUENT_USER_PROMPT_FIXED_ITERATIONS
            if self._enforce_fixed_iterations
            else SUBSEQUENT_USER_PROMPT
        )
        return template.format(
            question=question,
            step=step,
            max_steps=max_steps,
            remaining_crops=remaining_crops,
            last_region=last_region or "unknown",
        )
