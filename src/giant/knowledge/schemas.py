"""定义知识检索的证据条目、证据包与语料统计。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# 证据在流水线中出现的位置决定注入预算与提示词措辞。
Stage = Literal["planning", "consult", "arbitration"]


class Passage(BaseModel, frozen=True):
    """一条可引用的知识片段（教科书/指南的一个 snippet）。"""

    id: str = Field(..., min_length=1, description="Unique snippet id")
    text: str = Field(..., min_length=1, description="Snippet content")
    title: str = Field(default="", description="Source book or guideline title")
    source: str = Field(default="textbook", description="textbook | guideline | case")
    section: str | None = Field(default=None, description="Chapter or section if known")
    score: float = Field(default=0.0, description="Retrieval score (BM25 by default)")
    meta: dict[str, str] = Field(
        default_factory=dict, description="Extra provenance (edition, url, ...)"
    )


class EvidenceBundle(BaseModel, frozen=True):
    """一次检索的完整结果，供提示词注入与轨迹落库使用。"""

    query: str = Field(default="", description="Query sent to the retriever")
    stage: Stage = Field(default="planning", description="Pipeline stage")
    passages: list[Passage] = Field(default_factory=list)
    discriminators: list[str] = Field(
        default_factory=list,
        description="本体扩展得到的「必须与之区分」条目（KG-RAG，P1 启用）",
    )
    corpus_version: str = Field(default="", description="Corpus fingerprint")

    @property
    def is_empty(self) -> bool:
        return not self.passages

    def citation_ids(self) -> list[str]:
        """按注入顺序返回证据编号 E1..En，供仲裁环节引用校验使用。"""
        return [f"E{i + 1}" for i in range(len(self.passages))]

    def with_stage(self, stage: Stage) -> "EvidenceBundle":
        return self.model_copy(update={"stage": stage})


class KnowledgeStats(BaseModel, frozen=True):
    """语料与索引的自查信息，供 `giant knowledge status` 展示。"""

    corpus_path: str
    corpus_version: str
    n_passages: int
    titles: list[str] = Field(default_factory=list)
    total_chars: int = 0
    n_dropped_noise: int = Field(
        default=0, description="因 PDF 抽取噪声被丢弃的原始片段数"
    )


class CitationReport(BaseModel, frozen=True):
    """仲裁结论对证据的引用情况，用于量化「可解释性」而不是口头宣称。"""

    used: list[str] = Field(default_factory=list, description="被引用的证据编号")
    unused: list[str] = Field(default_factory=list, description="已提供但未被引用")
    invalid: list[str] = Field(
        default_factory=list, description="引用了但不存在的编号（幻觉信号）"
    )
    coverage: float = Field(
        default=0.0, ge=0.0, le=1.0, description="被引用证据占已提供证据的比例"
    )

    @property
    def has_citations(self) -> bool:
        return bool(self.used)


class WeightedEvidence(BaseModel, frozen=True):
    """仲裁阶段对单条专家结论的加权记录（结构化的「证据秤」）。"""

    region: str = Field(default="", description="区域坐标或描述")
    opinion: str = Field(default="", description="专家结论原文（截断）")
    weight: float = Field(default=0.0, ge=0.0, description="有效权重")
    why: str = Field(default="", description="为什么给这个权重")


class ArbitrationResult(BaseModel, frozen=True):
    """结构化仲裁输出：答案 + 加权证据 + 冲突 + 少数派 + 引用。"""

    answer: str = Field(default="", description="统一的最终结论")
    weighted_evidence: list[WeightedEvidence] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list, description="区域/证据间矛盾")
    minority: list[str] = Field(default_factory=list, description="少数派结论及其区域")
    citations: list[str] = Field(default_factory=list, description='引用的证据编号（E1…）')
    structured: bool = Field(
        default=False, description="是否成功解析为结构化输出"
    )
    raw: str = Field(default="", description="模型原始输出，便于排查")
