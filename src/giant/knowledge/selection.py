"""语料选择规则：用显式的 spec 描述「从哪本书的哪一段取知识」。

为什么不直接全量入库：医学教科书按器官分章，全量入库会让「黏液腺癌」命中肺癌
分类表（同一批词在不同章节反复出现）。而按关键词全局过滤又会误伤
（``colon`` 出现在「直肠活检查淀粉样变」这类无关句子里）。

真正可靠的信号是 **章节位置**：结直肠章节是结直肠词频在文档顺序上连续密集的一段。
本模块把这份人工核对过的选择固化进 ``data/corpus/sources.json``，让语料可复现、可审计。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class SourceSpec:
    """一本书的选取规则：按 chunk id 区间，或按正则筛选，或全量。"""

    file: str
    title: str
    source: str = "textbook"
    subdir: str = "rag_raw"
    id_range: tuple[int, int] | None = None
    regex: str | None = None

    def __post_init__(self) -> None:
        if self.id_range is not None and self.id_range[0] > self.id_range[1]:
            raise ValueError(f"{self.file}: id_range 上下界颠倒")


def _numeric_id(raw_id: str) -> int | None:
    """chunk id 形如 ``Pathology_Robbins_1234``，取末尾数字用于区间选择。"""
    suffix = raw_id.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else None


def load_spec(path: Path) -> tuple[list[SourceSpec], dict]:
    """读取 spec 文件，返回 (源列表, 全局参数)。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    specs = [
        SourceSpec(
            file=str(item["file"]),
            title=str(item.get("title") or Path(str(item["file"])).stem),
            source=str(item.get("source") or "textbook"),
            subdir=str(item.get("subdir") or "rag_raw"),
            id_range=tuple(item["id_range"]) if item.get("id_range") else None,  # type: ignore[arg-type]
            regex=item.get("regex"),
        )
        for item in data.get("sources", [])
    ]
    if not specs:
        raise ValueError(f"spec 中没有可用语源: {path}")
    return specs, {k: v for k, v in data.items() if k != "sources"}


def select_items(base_dirs: dict[str, Path], specs: Iterable[SourceSpec]) -> Iterator[dict]:
    """按规则逐条产出原始片段（附加 title/source 供后续构建）。

    base_dirs 提供若干候选目录（如 {"rag_raw": ..., "corpus": ...}），
    由每条 spec 的 subdir 指定从哪个目录取文件。
    """
    for spec in specs:
        base = base_dirs.get(spec.subdir)
        if base is None:
            continue
        path = base / spec.file
        if not path.exists():
            continue
        pattern = re.compile(spec.regex, re.IGNORECASE) if spec.regex else None
        low, high = spec.id_range if spec.id_range else (None, None)

        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw_id = str(item.get("id") or "")
                if not raw_id:
                    continue
                if low is not None and high is not None:
                    number = _numeric_id(raw_id)
                    if number is None or not (low <= number <= high):
                        continue
                elif pattern is not None:
                    text = str(item.get("content") or item.get("contents") or "")
                    if not pattern.search(text):
                        continue
                # 两条规则都不给 → 全量保留（用于人工策展的片段文件）

                yield {
                    **item,
                    "title": item.get("title") or spec.title,
                    "_source": spec.source,
                }
