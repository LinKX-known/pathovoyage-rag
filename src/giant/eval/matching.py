"""把模型的自由文本答案对齐到选择题选项（A/B/C/D）。

为什么需要独立一层：我们的 agent 输出的是中文/英文诊断结论（"诊断为黏液腺癌"），
而基准的判分口径是选项字母。这层匹配的错判会直接变成准确率的假象，
所以每一档都记录匹配方式（method），便于回溯「这个准确率是怎么来的」。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PUNCT_RE = re.compile(r"[\s\u3000,;:.!?，。；：、！？（）()\[\]{}<>《》\"'“”‘’|/\\_-]+")
_LETTER_RE = re.compile(r"^\s*(?:option|answer|答案|选项)?\s*[:：]?\s*([A-Da-d])\s*[.。)]?\s*$")


@dataclass(frozen=True)
class MatchResult:
    """一次匹配的结果；letter 为 None 表示无法对齐（判为错答）。"""

    letter: str | None
    method: str
    score: float = 0.0


def normalize(text: str) -> str:
    """去标点、去空白、统一大小写，用于稳健比较。"""
    return _PUNCT_RE.sub("", (text or "").lower())


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(token) > 1}


def _dice(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return 2 * len(left & right) / (len(left) + len(right))


def match_option(
    answer: str,
    options: dict[str, str],
    *,
    min_dice: float = 0.6,
) -> MatchResult:
    """按「字母 → 精确 → 子串 → 词重叠」四档依次尝试对齐。"""
    if not answer or not options:
        return MatchResult(None, "empty")

    # ① 模型直接回了字母
    letter_match = _LETTER_RE.match(answer)
    if letter_match:
        letter = letter_match.group(1).upper()
        if letter in options:
            return MatchResult(letter, "letter", 1.0)

    normalized_answer = normalize(answer)

    # ② 与某个选项完全一致
    for letter, text in options.items():
        if normalized_answer == normalize(text):
            return MatchResult(letter, "exact", 1.0)

    # ③ 子串包含：长选项优先，避免「Moderately differentiated」抢走
    #    「Moderately differentiated adenocarcinoma」的匹配
    for letter, text in sorted(options.items(), key=lambda kv: len(kv[1]), reverse=True):
        normalized_option = normalize(text)
        if len(normalized_option) < 3:
            continue
        if normalized_option in normalized_answer:
            return MatchResult(letter, "substring", 1.0)

    # ④ 词重叠（Dice 系数）
    answer_tokens = _tokens(answer)
    best_letter, best_score = None, 0.0
    for letter, text in options.items():
        score = _dice(answer_tokens, _tokens(text))
        if score > best_score:
            best_letter, best_score = letter, score
    if best_letter is not None and best_score >= min_dice:
        return MatchResult(best_letter, "tokens", round(best_score, 3))

    return MatchResult(None, "no_match", round(best_score, 3))
