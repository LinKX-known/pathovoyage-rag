"""把检索结果压缩成可注入提示词的「判据卡」与「证据块」。

两条硬约束：
1. Patho-R1 是本地 7B（max_tokens=256），判据卡必须极短（默认 ≤900 字符）；
2. 仲裁环节是最终答案的出口，需要完整编号与出处，便于事后做引用校验。
"""

from __future__ import annotations

import re

from giant.knowledge.schemas import CitationReport, EvidenceBundle

_DEFAULT_CARD_CHARS = 900
_DEFAULT_EVIDENCE_CHARS = 2400

_CITATION_RE = re.compile(r"\[(E\d+)\]")


def _clip(text: str, budget: int) -> str:
    """按词边界截断到预算内。"""
    text = " ".join(text.split())
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    window = text[:budget]
    cut = window.rfind(" ")
    if cut < budget * 0.6:
        return window.rstrip() + "…"
    return window[:cut].rstrip() + "…"


def criteria_card(
    bundle: EvidenceBundle,
    *,
    max_chars: int = _DEFAULT_CARD_CHARS,
    max_items: int = 3,
) -> str:
    """生成给专家模型（Patho-R1）的短判据卡；无证据时返回空串。"""
    if bundle.is_empty:
        return ""

    header = "# 参考判据（检索自病理教科书，属于证据而非诊断结论）"
    lines: list[str] = [header]
    used = len(header)

    for index, passage in enumerate(bundle.passages[:max_items]):
        remaining = max_chars - used
        if remaining < 80:
            break
        snippet = _clip(passage.text, remaining - 12)
        if not snippet:
            break
        line = f"- [E{index + 1}] {snippet}"
        lines.append(line)
        used += len(line)

    if bundle.discriminators:
        for item in bundle.discriminators[:3]:
            remaining = max_chars - used
            if remaining < 40:
                break
            line = f"- 需鉴别：{_clip(item, remaining - 10)}"
            lines.append(line)
            used += len(line)

    lines.append("请以上述判据核对镜下所见；判据不足时明确说明，不要臆测。")
    return "\n".join(lines)


def evidence_block(
    bundle: EvidenceBundle,
    *,
    max_chars: int = _DEFAULT_EVIDENCE_CHARS,
    max_items: int = 6,
) -> str:
    """生成给仲裁模型的编号证据块，供其逐条引用 [E1]…。"""
    if bundle.is_empty:
        return ""

    chunks: list[str] = []
    used = 0
    for index, passage in enumerate(bundle.passages[:max_items]):
        remaining = max_chars - used
        if remaining < 120:
            break
        source = passage.title or passage.source
        body = _clip(passage.text, remaining - 60)
        if not body:
            break
        chunk = f"[E{index + 1}] {source}\n{body}"
        chunks.append(chunk)
        used += len(chunk)

    return "\n\n".join(chunks)


def citation_check(answer: str, bundle: EvidenceBundle) -> CitationReport:
    """统计仲裁结论对 `[E#]` 证据的引用情况。

    这是「可解释性」唯一可量化的口径：提供了 N 条证据，结论引用了其中几条、
    有没有引用不存在的编号（幻觉信号）。
    """
    provided = bundle.citation_ids()
    found = {match.group(1) for match in _CITATION_RE.finditer(answer or "")}
    used = [citation for citation in provided if citation in found]
    unused = [citation for citation in provided if citation not in found]
    invalid = sorted(found - set(provided))
    coverage = (len(used) / len(provided)) if provided else 0.0
    return CitationReport(
        used=used, unused=unused, invalid=invalid, coverage=round(coverage, 3)
    )
