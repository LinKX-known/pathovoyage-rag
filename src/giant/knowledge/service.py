"""知识层门面：加载语料、执行检索、返回带引用的证据包。

设计约束：
- **可注入**：runner 通过 AgentConfig 接收一个 KnowledgeService 实例（与 conch_scorer
  相同的注入方式），因此 RAG 关闭时主链路零改动、零 import 成本；
- **懒加载**：语料与索引在首次检索时才构建，CLI 启动不受影响；
- **可复现**：检索不依赖任何 API，同一语料 + 同一问题 → 同一结果。
"""

from __future__ import annotations

from pathlib import Path

from giant.knowledge.bm25 import BM25Index
from giant.knowledge.corpus import compute_version, read_corpus
from giant.knowledge.lexicon import expand_query, has_cjk
from giant.knowledge.schemas import EvidenceBundle, KnowledgeStats, Passage, Stage

_DEFAULT_TOP_K = 6

# 相对分数截断：低于最高分这一比例的命中视为噪声。比「覆盖率门槛」更语言无关，
# 中英混排的查询（中文问题 + 英文术语展开）不会因为分母膨胀而整体被剔空。
_DEFAULT_RELATIVE_CUTOFF = 0.25


class KnowledgeError(RuntimeError):
    """语料缺失或索引构建失败时抛出，附带可操作的修复提示。"""


class KnowledgeService:
    """病理教科书语料的检索服务（BM25 + 中英术语展开）。"""

    def __init__(
        self,
        corpus_path: str | Path,
        *,
        top_k: int = _DEFAULT_TOP_K,
        min_score: float = 0.0,
        min_coverage: float = 0.0,
        relative_cutoff: float = _DEFAULT_RELATIVE_CUTOFF,
    ) -> None:
        self._corpus_path = Path(corpus_path).expanduser()
        self._top_k = max(1, top_k)
        self._min_score = min_score
        self._min_coverage = min_coverage
        self._relative_cutoff = relative_cutoff
        self._passages: list[Passage] | None = None
        self._index: BM25Index | None = None
        self._version: str = ""
        self._cache: dict[tuple[str, str, int], EvidenceBundle] = {}

    # ------------------------------------------------------------------ 构造

    @classmethod
    def from_settings(cls) -> "KnowledgeService":
        """按 .env 配置创建服务；配置项在 giant.config 中定义。"""
        from giant.config import settings

        return cls(
            settings.RAG_CORPUS_PATH,
            top_k=settings.RAG_TOP_K,
            min_score=settings.RAG_MIN_SCORE,
            min_coverage=settings.RAG_MIN_COVERAGE,
            relative_cutoff=settings.RAG_RELATIVE_CUTOFF,
        )

    # ------------------------------------------------------------------ 加载

    def _ensure_ready(self) -> None:
        if self._index is not None:
            return
        if not self._corpus_path.exists():
            raise KnowledgeError(
                f"知识语料不存在: {self._corpus_path}。"
                "先运行 `giant knowledge build` 生成语料（详见 docs/RAG_INTEGRATION.md）。"
            )
        passages = read_corpus(self._corpus_path)
        if not passages:
            raise KnowledgeError(f"知识语料为空: {self._corpus_path}")
        self._passages = passages
        self._version = compute_version(passages)
        self._index = BM25Index(passages)

    # ------------------------------------------------------------------ 查询

    @property
    def corpus_version(self) -> str:
        self._ensure_ready()
        return self._version

    @property
    def stats(self) -> KnowledgeStats:
        self._ensure_ready()
        assert self._passages is not None
        return KnowledgeStats(
            corpus_path=str(self._corpus_path),
            corpus_version=self._version,
            n_passages=len(self._passages),
            titles=sorted({p.title for p in self._passages if p.title}),
            total_chars=sum(len(p.text) for p in self._passages),
        )

    def retrieve(
        self,
        query: str,
        *,
        stage: Stage = "planning",
        k: int | None = None,
    ) -> EvidenceBundle:
        """检索并打包为 EvidenceBundle；中文问题会先做术语展开。

        相同 (query, stage, k) 的检索在进程内只算一次——预筛与规划阶段会各要一次
        完全相同的证据，缓存既省时间也保证两边看到的是同一份结果。
        """
        self._ensure_ready()
        assert self._index is not None

        cache_key = (query, stage, k or self._top_k)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        if not query or not query.strip():
            return EvidenceBundle(query=query, stage=stage, corpus_version=self._version)

        expanded = query
        if has_cjk(query):
            expanded, _hits = expand_query(query)

        hits = self._index.search(
            expanded,
            k=max(self._top_k, k or self._top_k) * 2,
            min_score=self._min_score,
            min_coverage=self._min_coverage,
        )
        if hits:
            best = max(score for _passage, score in hits)
            threshold = best * self._relative_cutoff
            hits = [(passage, score) for passage, score in hits if score >= threshold]

        bundle = EvidenceBundle(
            query=expanded,
            stage=stage,
            passages=[passage for passage, _score in hits[: (k or self._top_k)]],
            corpus_version=self._version,
        )
        self._cache[cache_key] = bundle
        return bundle

    def search(self, query: str, *, k: int = 5) -> list[Passage]:
        """供 CLI 调试使用的裸检索。"""
        return self.retrieve(query, k=k).passages
