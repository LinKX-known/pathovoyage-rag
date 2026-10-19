"""把检索到的知识与术语表转成 PLIP 的检索文本（确定性，不依赖云端模型）。

为什么需要这一层：

原实现里 PLIP 的检索文本完全由千问现场生成，一旦规划失败就回退到一句写死的
``colorectal adenocarcinoma tumor epithelium, H&E``——**所有问题共用同一句**，
于是「有没有脉管瘤栓」这类问题会被 PLIP 当成「给我找肿瘤」，预筛退化成无效步骤。

本模块给出三级、全确定性的构造策略（优先级从高到低）：

1. **策展判据卡里的英文短语**：卡里本来就写着 ``lymphovascular invasion, LVI``、
   ``circumferential resection margin CRM`` 这类标准表述，CLIP 对短语的匹配远好于
   零散单词；按问题锚点词与短语的词重叠度打分选取。
2. **术语表展开的英文词**：命中即用，每个中文术语构成一个检索维度。
3. **最后的固定兜底**：连术语都没命中时才使用。

全部是纯本地计算，同一问题永远得到同一结果——这对复现实验很重要。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from giant.knowledge.lexicon import expand_query
from giant.knowledge.schemas import EvidenceBundle, Passage
from giant.knowledge.tokenize import tokenize

# 连术语表都没命中时的最后兜底（保留原行为，作为安全网）。
LAST_RESORT_QUERY = "colorectal adenocarcinoma tumor epithelium, H&E"

# 从英文文本里抽取 2~5 词的短语：CLIP 对短语的响应比单词好，但也怕长句。
_PHRASE_RE = re.compile(r"[A-Za-z][A-Za-z\-]+(?:\s+[A-Za-z][A-Za-z\-]+){1,4}")
_ABBREV_RE = re.compile(r"\b[A-Z]{2,5}\b")


@dataclass(frozen=True)
class PlipQuery:
    """一个 PLIP 检索维度及其来源，来源用于审计「这段文本是怎么来的」。"""

    text: str
    weight: float
    source: str


def _english_terms(text: str) -> set[str]:
    """抽取英文关键词（含 CRM/LVI 这类缩写）。"""
    lowered = text.lower()
    terms = {token for token in re.findall(r"[a-z]{3,}", lowered)}
    terms.update(match.lower() for match in _ABBREV_RE.findall(text))
    return terms


def _passage_phrases(passage: Passage, limit: int = 6) -> list[str]:
    """从一个证据片段里抽出可用的英文短语。"""
    phrases: list[str] = []
    for match in _PHRASE_RE.finditer(passage.text):
        phrase = " ".join(match.group(0).split())
        if len(phrase) < 12 or len(phrase) > 80:
            continue
        if phrase.lower() not in {existing.lower() for existing in phrases}:
            phrases.append(phrase)
        if len(phrases) >= limit:
            break
    return phrases


def _score_phrase(phrase: str, anchors: set[str]) -> float:
    """短语与问题锚点的重叠度。

    只重叠一个词是不够的：``perineural invasion`` 与 ``depth of invasion`` 只共享
    ``invasion``，若放行就会把「浸润深度」错配成「神经侵犯」。因此要求至少命中
    两个锚点，重叠度用于在达标的短语之间排序。
    """
    tokens = _english_terms(phrase)
    if not tokens:
        return 0.0
    overlap = len(tokens & anchors)
    if overlap < 2:
        return 0.0
    return overlap / max(len(tokens), 1) + 0.15 * overlap


def _dedupe_words(text: str) -> str:
    """去掉重复词：``surgical margin resection margin`` → ``surgical margin resection``。"""
    seen: list[str] = []
    for word in text.split():
        if word.lower() not in {existing.lower() for existing in seen}:
            seen.append(word)
    return " ".join(seen)


def build_dimensions(
    question: str,
    evidence: EvidenceBundle | None = None,
    *,
    max_dimensions: int = 3,
    min_score: float = 0.2,
) -> list[PlipQuery]:
    """为 PLIP 构造检索维度；返回空列表表示完全无法构造（调用方用最后兜底）。"""
    _expanded, hits = expand_query(question)
    anchors = _english_terms(_expanded)

    # ① 策展判据卡优先：其中的英文短语是人工核对过的标准表述
    curated: list[tuple[float, str]] = []
    for passage in (evidence.passages if evidence else []):
        if passage.source != "curated":
            continue
        for phrase in _passage_phrases(passage):
            score = _score_phrase(phrase, anchors)
            if score >= min_score:
                curated.append((score, phrase))

    if curated:
        curated.sort(key=lambda item: item[0], reverse=True)
        picked: list[str] = []
        for _score, phrase in curated:
            if phrase.lower() in {item.lower() for item in picked}:
                continue
            picked.append(phrase)
            if len(picked) >= max_dimensions:
                break
        weight = round(1.0 / len(picked), 4)
        return [PlipQuery(text=text, weight=weight, source="curated") for text in picked]

    # ② 术语表展开：每个中文术语一个维度
    from giant.knowledge.lexicon import _LEXICON  # 局部导入，避免对外暴露

    lexicon_dims: list[str] = []
    for term in hits:
        english = _dedupe_words(_LEXICON.get(term, "").strip())
        if english and english not in lexicon_dims:
            lexicon_dims.append(english)
        if len(lexicon_dims) >= max_dimensions:
            break
    if lexicon_dims:
        weight = round(1.0 / len(lexicon_dims), 4)
        return [
            PlipQuery(text=text, weight=weight, source="lexicon") for text in lexicon_dims
        ]

    return []


def build_query_text(question: str, evidence: EvidenceBundle | None = None) -> str:
    """把维度拼成单条检索文本（给只有单文本接口的旧逻辑用）。"""
    dims = build_dimensions(question, evidence)
    if not dims:
        return LAST_RESORT_QUERY
    return "; ".join(dim.text for dim in dims)


def dimension_texts(dims: list[PlipQuery]) -> list[tuple[str, float]]:
    """转成 ``prescreen_candidates(dimensions=...)`` 需要的 (文本, 权重) 列表。"""
    return [(dim.text, dim.weight) for dim in dims]


__all__ = [
    "LAST_RESORT_QUERY",
    "PlipQuery",
    "build_dimensions",
    "build_query_text",
    "dimension_texts",
]
