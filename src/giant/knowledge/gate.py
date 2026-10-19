"""检索闸门：判断「检索回来的片段到底能不能回答这个问题」。

存在的理由（实测）：BM25 只看词重合，问「组织学亚型」会命中肛管 AIN、甲状腺、卵巢的
段落。这些噪声一旦被注入提示词，就会污染判据卡与仲裁依据——注入了错的知识，
比不注入更糟。

闸门只做两件事：给整包证据打一个相关性标签，并把明显无关的片段挡在注入之前。
默认是**确定性规则**（不额外调用 LLM），因此不增加成本、结果可复现；
需要更细的判断时，再把 LLM 版评估接到同一个接口上（见 docs/RAG_INTEGRATION.md）。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from giant.knowledge.lexicon import expand_query, has_cjk
from giant.knowledge.schemas import EvidenceBundle, Passage
from giant.knowledge.tokenize import tokenize

Relevance = Literal["relevant", "partially_relevant", "irrelevant"]

# 判定阈值：问题锚点词在片段里的命中个数。
_RELEVANT_AT = 3
_PARTIAL_AT = 2
# 人工策展的判据卡本身就是为这些问题写的，给一次额外命中。
_CURATED_BONUS = 1

# 语料是结直肠专用的（见 data/corpus/sources.json）。问其它器官时，
# 通用病理词（adenocarcinoma / breast / immunohistochemistry）会误判成相关，
# 因此先在「域」这一层做拦截：明确问别的器官、且没提结直肠 → 直接判无关。
_IN_SCOPE_TERMS: tuple[str, ...] = (
    "结直肠",
    "结肠",
    "直肠",
    "回盲部",
    "阑尾",
    "colorectal",
    "colon",
    "colonic",
    "rectal",
    "rectum",
)
_OUT_OF_SCOPE_ORGANS: tuple[str, ...] = (
    "胃",
    "食管",
    "小肠",
    "十二指肠",
    "肺",
    "乳腺",
    "肝",
    "胰腺",
    "胆管",
    "肾",
    "膀胱",
    "前列腺",
    "卵巢",
    "子宫内膜",
    "宫颈",
    "甲状腺",
    "皮肤",
    "软组织",
    "鼻咽",
)


class GateDecision(BaseModel, frozen=True):
    """闸门判定结果，写进轨迹用于事后审计「为什么注入了/没注入」。"""

    relevance: Relevance = "irrelevant"
    best_coverage: int = 0
    kept: int = 0
    dropped: int = 0
    anchors: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    @property
    def allows_injection(self) -> bool:
        return self.relevance != "irrelevant" and self.kept > 0


def _anchor_terms(question: str) -> list[str]:
    """把问题拆成用于判定的锚点词（中文先展开成英文，再统一分词）。"""
    expanded = question
    if has_cjk(question):
        expanded, _hits = expand_query(question)
    return sorted(set(tokenize(expanded)))


def _coverage(passage: Passage, anchors: list[str]) -> int:
    tokens = set(tokenize(passage.text + " " + passage.title))
    hits = sum(1 for anchor in anchors if anchor in tokens)
    if passage.source == "curated" and hits > 0:
        hits += _CURATED_BONUS
    return hits


def _scope_violation(question: str) -> str | None:
    """问题是否落在语料域之外（明确问了别的器官、又没提结直肠）。"""
    text = question.lower()
    if any(term in text for term in _IN_SCOPE_TERMS):
        return None
    for organ in _OUT_OF_SCOPE_ORGANS:
        if organ in text:
            return f"out_of_scope_organ={organ}"
    return None


def assess(
    bundle: EvidenceBundle,
    question: str,
    *,
    relevant_at: int = _RELEVANT_AT,
    partial_at: int = _PARTIAL_AT,
) -> GateDecision:
    """给证据包打相关性标签，并给出被判为「可注入」的片段数。"""
    anchors = _anchor_terms(question)
    if bundle.is_empty:
        return GateDecision(
            relevance="irrelevant", anchors=anchors, reasons=["retrieval empty"]
        )

    violation = _scope_violation(question)
    if violation is not None:
        return GateDecision(
            relevance="irrelevant",
            anchors=anchors[:12],
            reasons=[violation, "corpus_scope=colorectal"],
        )

    coverages = [_coverage(passage, anchors) for passage in bundle.passages]
    best = max(coverages)
    kept = sum(1 for coverage in coverages if coverage >= partial_at)

    if best >= relevant_at:
        relevance: Relevance = "relevant"
    elif best >= partial_at:
        relevance = "partially_relevant"
    else:
        relevance = "irrelevant"

    reasons = [
        f"best_coverage={best}",
        f"kept={kept}/{len(coverages)}",
        f"anchors={len(anchors)}",
    ]
    return GateDecision(
        relevance=relevance,
        best_coverage=best,
        kept=kept,
        dropped=len(coverages) - kept,
        anchors=anchors[:12],
        reasons=reasons,
    )


def apply(
    bundle: EvidenceBundle,
    question: str,
    *,
    relevant_at: int = _RELEVANT_AT,
    partial_at: int = _PARTIAL_AT,
) -> tuple[EvidenceBundle, GateDecision]:
    """按闸门结果过滤证据包；整包无关时返回空包（调用方据此跳过注入）。"""
    decision = assess(bundle, question, relevant_at=relevant_at, partial_at=partial_at)
    if not decision.allows_injection:
        return (
            EvidenceBundle(
                query=bundle.query,
                stage=bundle.stage,
                corpus_version=bundle.corpus_version,
            ),
            decision,
        )

    kept = [
        passage
        for passage in bundle.passages
        if _coverage(passage, decision.anchors) >= partial_at
    ]
    return (
        bundle.model_copy(update={"passages": kept}),
        decision.model_copy(update={"kept": len(kept)}),
    )
