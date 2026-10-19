"""语料构建与读取：把 MedRAG textbooks 的原始 jsonl 策展成本项目可用的语料。

原始片段有两个致命问题，构建阶段必须处理：

1. **PDF 抽取噪声**：Robbins 约 16% 的片段词间空格丢失
   （如 ``Rupturedovarianmucinoustumorsmayseedtheperitoneum``），这类片段
   在 BM25 里几乎不可能被命中，留着只会稀释 idf。
2. **片段过长且跨章节**：平均 1.6KB，一段里混着肺、乳腺、卵巢多个主题，
   用 1.6KB 的块做检索精度极差。

因此做四件事：噪声过滤 → 句子级滑窗切分（默认 600 字符）→ 语料指纹 → 落盘。
「从哪些书、哪些章节取」由 ``selection.py`` + ``data/corpus/sources.json`` 决定。
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Iterable, Sequence

from giant.knowledge.schemas import KnowledgeStats, Passage

_DEFAULT_WINDOW_CHARS = 600
_DEFAULT_MIN_CHARS = 100

# 噪声判据：空格占比过低 = 词被粘在一起；词长中位数过大 = 同样是粘连。
_NOISE_SPACE_RATIO = 0.08
_NOISE_MAX_MEDIAN_TOKEN = 16
_NOISE_MIN_WORDS = 5

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _is_noisy(text: str) -> bool:
    """判断片段是否因 PDF 抽取而失去词边界。

    中文段落天然没有空格分隔，不适用这条判据（人工策展的中文判据卡会走这里）。
    """
    stripped = text.strip()
    if not stripped:
        return True
    if _CJK_RE.search(stripped):
        return False
    space_ratio = stripped.count(" ") / len(stripped)
    if space_ratio < _NOISE_SPACE_RATIO:
        return True
    words = stripped.split()
    if len(words) < _NOISE_MIN_WORDS:
        return True
    return statistics.median(len(word) for word in words) > _NOISE_MAX_MEDIAN_TOKEN


def _hard_wrap(text: str, max_chars: int) -> list[str]:
    """对超长句子按词强制换行。"""
    pieces: list[str] = []
    buffer = ""
    for word in text.split():
        candidate = f"{buffer} {word}".strip()
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        if buffer:
            pieces.append(buffer)
        buffer = word[:max_chars]
    if buffer:
        pieces.append(buffer)
    return pieces


def _split_windows(text: str, *, max_chars: int, min_chars: int) -> list[str]:
    """把长片段切成句子级滑窗，窗口内保持语义完整。"""
    units: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            units.extend(_hard_wrap(sentence, max_chars))
        else:
            units.append(sentence)

    windows: list[str] = []
    buffer = ""
    for unit in units:
        candidate = f"{buffer} {unit}".strip()
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        if buffer:
            windows.append(buffer)
        buffer = unit
    if buffer:
        windows.append(buffer)

    return [window for window in windows if len(window) >= min_chars]


def _iter_raw_snippets(paths: Iterable[Path]) -> Iterable[dict]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def compute_version(passages: Sequence[Passage]) -> str:
    """语料指纹：id + 文本长度 + 标题的有序哈希，写入轨迹用于复现。"""
    digest = hashlib.sha256()
    for passage in passages:
        digest.update(f"{passage.id}|{len(passage.text)}|{passage.title}\n".encode())
    return digest.hexdigest()[:12]


def build_corpus_from_items(
    items: Iterable[dict],
    out_path: Path,
    *,
    window_chars: int = _DEFAULT_WINDOW_CHARS,
    min_chars: int = _DEFAULT_MIN_CHARS,
) -> KnowledgeStats:
    """把（已选好的）原始片段噪声过滤、切窗后落盘，返回统计信息。"""
    passages: list[Passage] = []
    seen_ids: set[str] = set()
    n_noise = 0

    for item in items:
        raw_id = str(item.get("id") or "")
        text = str(item.get("content") or item.get("contents") or "").strip()
        if not raw_id or len(text) < min_chars:
            continue
        if _is_noisy(text):
            n_noise += 1
            continue

        windows = _split_windows(text, max_chars=window_chars, min_chars=min_chars)
        for index, window in enumerate(windows):
            window_id = f"{raw_id}#w{index}"
            if window_id in seen_ids:
                continue
            seen_ids.add(window_id)
            passages.append(
                Passage(
                    id=window_id,
                    text=window,
                    title=str(item.get("title") or ""),
                    source=str(item.get("_source") or "textbook"),
                    meta={"raw_id": raw_id, "window": str(index)},
                )
            )

    passages.sort(key=lambda passage: passage.id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for passage in passages:
            handle.write(
                json.dumps(
                    {
                        "id": passage.id,
                        "text": passage.text,
                        "title": passage.title,
                        "source": passage.source,
                        "meta": passage.meta,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return KnowledgeStats(
        corpus_path=str(out_path),
        corpus_version=compute_version(passages),
        n_passages=len(passages),
        titles=sorted({p.title for p in passages if p.title}),
        total_chars=sum(len(p.text) for p in passages),
        n_dropped_noise=n_noise,
    )


def build_corpus(
    raw_paths: Sequence[Path],
    out_path: Path,
    *,
    keywords: Sequence[str] | None = None,
    window_chars: int = _DEFAULT_WINDOW_CHARS,
    min_chars: int = _DEFAULT_MIN_CHARS,
    source: str = "textbook",
) -> KnowledgeStats:
    """从整份 jsonl 直接构建（按关键词粗筛），供快速试跑用。

    正式语料请用 ``selection.py`` 的 spec 方式（章节级选择），精度高得多。
    """
    pattern = (
        re.compile("|".join(re.escape(word) for word in keywords), re.IGNORECASE)
        if keywords
        else None
    )

    def _items() -> Iterable[dict]:
        for item in _iter_raw_snippets(raw_paths):
            text = str(item.get("content") or item.get("contents") or "")
            if pattern is not None and not pattern.search(text):
                continue
            yield {**item, "_source": source}

    return build_corpus_from_items(
        _items(), out_path, window_chars=window_chars, min_chars=min_chars
    )


def read_corpus(path: Path) -> list[Passage]:
    """读取策展后的语料文件。"""
    passages: list[Passage] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            passages.append(
                Passage(
                    id=str(item["id"]),
                    text=str(item["text"]),
                    title=str(item.get("title") or ""),
                    source=str(item.get("source") or "textbook"),
                    meta=dict(item.get("meta") or {}),
                )
            )
    return passages
