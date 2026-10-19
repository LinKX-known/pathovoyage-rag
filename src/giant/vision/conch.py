"""定义可选图文评分器接口及未配置时的错误行为。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from PIL import Image


class ConchUnavailableError(RuntimeError):
    """请求未配置的图文评分器时抛出。"""


class ConchScorer(Protocol):
    """约定候选诊断图文评分器的接口。"""

    def score_hypotheses(
        self, image: Image.Image, hypotheses: list[str]
    ) -> list[float]:
        """按输入顺序返回各候选诊断与图像的相似度分数。"""


class UnconfiguredConchScorer:
    """在评分器未配置时给出明确错误。"""

    def score_hypotheses(
        self, image: Image.Image, hypotheses: list[str]
    ) -> list[float]:
        raise ConchUnavailableError(
            "CONCH scoring was requested but no conch_scorer is configured. "
            "Provide an implementation via AgentConfig.conch_scorer."
        )
