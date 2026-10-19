"""自包含的 Okapi BM25 检索器（无第三方依赖、索引在内存中构建）。

为什么不用 rank_bm25 / pyserini：
- pyserini 需要 JVM，且与项目现有依赖冲突；
- 本语料规模（数千段）用纯 Python 倒排索引，构建 <2s、查询 <50ms，足够 P0；
- 无模型、无网络，检索结果完全可复现（这点对临床可审计性很重要）。
"""

from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict

from giant.knowledge.schemas import Passage
from giant.knowledge.tokenize import tokenize


class BM25Index:
    """在给定语料上构建 BM25 倒排索引并提供 top-k 检索。"""

    def __init__(
        self,
        passages: list[Passage],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._passages = list(passages)
        self._k1 = k1
        self._b = b

        postings: dict[str, dict[int, int]] = defaultdict(dict)
        doc_len: list[int] = []

        for doc_id, passage in enumerate(self._passages):
            # 标题也参与索引：课文名（如 Robbins 章节名）常含关键实体。
            tokens = tokenize(passage.text) + tokenize(passage.title)
            counts = Counter(tokens)
            for token, tf in counts.items():
                postings[token][doc_id] = tf
            doc_len.append(max(len(tokens), 1))

        self._postings = dict(postings)
        self._doc_len = doc_len
        self._n_docs = len(self._passages)
        self._avgdl = (sum(doc_len) / self._n_docs) if self._n_docs else 1.0
        self._idf = {
            token: math.log(1.0 + (self._n_docs - len(docs) + 0.5) / (len(docs) + 0.5))
            for token, docs in self._postings.items()
        }

    def __len__(self) -> int:
        return self._n_docs

    @property
    def vocabulary_size(self) -> int:
        return len(self._postings)

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        min_score: float = 0.0,
        min_coverage: float = 0.0,
    ) -> list[tuple[Passage, float]]:
        """返回 (片段, 分数) 列表，按分数降序。

        min_coverage 要求片段至少命中查询词的这一比例（0~1）。中文问题是多概念的，
        只命中一个泛词（如 "invasion"）的片段即使 BM25 分数不低也往往无关，
        用覆盖率门槛把它们挡掉。
        """
        if k <= 0 or not self._n_docs:
            return []

        # 查询词去重：重复出现不重复累加，避免长问句里高频词主导排序。
        query_tokens = list(dict.fromkeys(tokenize(query)))
        if not query_tokens:
            return []

        scores: dict[int, float] = defaultdict(float)
        matched: dict[int, int] = defaultdict(int)
        for token in query_tokens:
            docs = self._postings.get(token)
            if not docs:
                continue
            idf = self._idf[token]
            for doc_id, tf in docs.items():
                dl = self._doc_len[doc_id]
                denom = tf + self._k1 * (1.0 - self._b + self._b * dl / self._avgdl)
                scores[doc_id] += idf * tf * (self._k1 + 1.0) / denom
                matched[doc_id] += 1

        if not scores:
            return []

        min_hits = max(1, math.ceil(min_coverage * len(query_tokens)))
        candidates = [
            (doc_id, score)
            for doc_id, score in scores.items()
            if score >= min_score and matched[doc_id] >= min_hits
        ]
        if not candidates:
            return []

        top = heapq.nlargest(k, candidates, key=lambda item: item[1])
        results: list[tuple[Passage, float]] = []
        for doc_id, score in top:
            passage = self._passages[doc_id]
            results.append((passage.model_copy(update={"score": round(score, 4)}), score))
        return results
