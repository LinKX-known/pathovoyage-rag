"""向病理专科模型发送裁剪图像，并将专家结论转为中文。"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO

from openai import AsyncOpenAI
from PIL import Image

from giant.config import settings
from giant.llm.model_registry import DEFAULT_QWEN_MODEL
from giant.utils.logging import get_logger

logger = get_logger(__name__)


# 通用反偏误规则：直接给结论 + 一句依据、有证据才判有、拿不准判无/不确定。
_CONSULT_PROMPT = """You are a pathology expert. Examine this H&E-stained tissue crop and answer the question below.

Rules:
- Give a direct conclusion first, then one or two sentences of key microscopic evidence. Do not list steps, do not write a long report, do not give follow-up recommendations.
- Make a definitive claim only when you have clear microscopic evidence; if evidence is insufficient or ambiguous, say so explicitly instead of guessing.

Observation task at the current magnification:
{observation_task}

Diagnostic criteria reference:
{criteria}

Question: {question}"""

# 器官上下文前置模板：明确这是哪个器官的原发灶，严禁跨器官误判（尤其乳腺）。
_TISSUE_CONTEXT_PREFIX = (
    "切片部位/器官信息：{context}。\n"

)


# （输出「不不不不」「对称/对称」「细胞/细胞」这类死循环）。0.3 的采样能打破循环。
_PATHO_R1_TEMPERATURE = 0.3


@dataclass(frozen=True)
class PathoR1Result:
    """保存专科会诊的结论、费用和错误信息。"""

    opinion: str


class PathoR1Client:
    """调用专科模型分析裁剪图，并在需要时翻译结论。"""

    def __init__(
        self,
        *,
        base_url: str,
        model: str = "patho-r1-7b",
        api_key: str = "EMPTY",
        question: str,
        criteria: str = "",
        tissue_context: str | None = None,
    ) -> None:


        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=120.0)
        self.model = model

        self.question = question

        self.criteria = criteria

        self.tissue_context = tissue_context

        self._translator_client = AsyncOpenAI(
            api_key=settings.require_dashscope_key(),
            base_url=settings.DASHSCOPE_BASE_URL,
        )
        self._translator_model = DEFAULT_QWEN_MODEL

    @staticmethod
    def _is_mostly_chinese(text: str) -> bool:
        """判断文本是否已主要是中文（>30% 汉字），用于跳过翻译。"""
        cjk = sum(1 for c in text if "一" <= c <= "鿿")
        return cjk > 0 and cjk / max(len(text), 1) > 0.3

    async def _translate_to_chinese(self, text: str) -> str:

        if not text or self._is_mostly_chinese(text):
            return text
        try:
            resp = await self._translator_client.chat.completions.create(
                model=self._translator_model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "把下面这段病理诊断文字翻译成简体中文，保留医学专业术语"
                            "（如 mucinous adenocarcinoma→黏液腺癌、invasion depth→浸润深度、"
                            "lymphovascular invasion→脉管瘤栓）。只输出译文，不要解释：\n\n"
                            f"{text}"
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=2048,
            )
            return (resp.choices[0].message.content or text).strip()
        except Exception as e:
            logger.warning("Patho-R1 translation failed: %s", e)
            return text

    async def consult(
        self,
        image: Image.Image,
        observation_task: str = "",
        tissue_context: str | None = None,
        criteria: str | None = None,
    ) -> PathoR1Result:
        """发送裁剪图像与观察任务，返回专科诊断及调用统计。

        criteria 非空时覆盖实例级判据——逐 crop 会按当前倍率注入更贴合的判据片段。
        """
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

        task = observation_task or "(observe the field and answer the question)"
        criteria_text = criteria or self.criteria or "(none; use general diagnostic standards)"
        prompt = _CONSULT_PROMPT.format(
            question=self.question, criteria=criteria_text, observation_task=task
        )
        context = tissue_context or self.tissue_context
        if context:
            prompt = _TISSUE_CONTEXT_PREFIX.format(context=context) + prompt

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=_PATHO_R1_TEMPERATURE,
            max_tokens=256,
        )

        raw = response.choices[0].message.content or ""

        opinion_zh = await self._translate_to_chinese(raw)
        return PathoR1Result(opinion=opinion_zh)
