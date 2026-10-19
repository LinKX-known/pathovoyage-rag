"""轻量英文分词：小写、切分、去停用词、极简词干还原。

刻意不引入 nltk/spacy —— 检索层要保持「无模型、无网络、可复现」，
临床语料的 BM25 收益在细粒度词干化上很有限，用规则足够。
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")

_STOPWORDS: frozenset[str] = frozenset(
    """
    a an the of in on for to and or with without by is are was were be been being
    that this these those as at from into during over under between within its it
    can may might should would could not no nor but if then than more most less
    least such via per also which who whom whose there their they we you your our
    has have had do does did done both each other some any all only very much many
    up down out off again further once here when where why how what while about
    against before after above below both few own same so too s t just don now
    """.split()
)

# 临床语料里高频但几乎不带来区分度的词。
_DOMAIN_NOISE: frozenset[str] = frozenset(
    {"patient", "patients", "figure", "fig", "table", "chapter", "see", "note"}
)


def _stem(token: str) -> str:
    """极简词干还原：只处理复数与常见后缀，避免过度合并。"""
    if len(token) <= 4:
        return token
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    if token.endswith(("sses", "shes", "ches", "xes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def _cjk_bigrams(run: str) -> list[str]:
    """中文按字符二元组切分：无需分词器即可获得可检索的索引项。"""
    if len(run) == 1:
        return [run]
    return [run[index : index + 2] for index in range(len(run) - 1)]


def tokenize(text: str) -> list[str]:
    """把文本切成可索引的词元序列（英文词 + 中文字符二元组）。"""
    tokens: list[str] = []
    lowered = text.lower()
    for raw in _TOKEN_RE.findall(lowered):
        if len(raw) < 2:
            continue
        if raw in _STOPWORDS or raw in _DOMAIN_NOISE:
            continue
        tokens.append(_stem(raw))
    for run in _CJK_RUN_RE.findall(text):
        tokens.extend(_cjk_bigrams(run))
    return tokens


def tokenize_query(text: str) -> list[str]:
    """查询侧分词：与索引侧保持一致，便于命中。"""
    return tokenize(text)
