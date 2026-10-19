"""评测指标：准确率、macro-F1、逐任务分解，以及多数类基线。

为什么单列基线：SlideBench 的结直肠子集类别极不均衡（HISTAI 三分类里恶性占 76%），
只报 accuracy 会把「全判恶性」糊成好成绩，所以基线必须同表呈现。
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class MetricSummary:
    n: int
    n_matched: int
    accuracy: float
    macro_f1: float
    n_invalid: int
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    majority_baseline: float = 0.0

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "n_matched": self.n_matched,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "n_invalid": self.n_invalid,
            "majority_baseline": self.majority_baseline,
            "per_class": self.per_class,
        }


def _pairs(records: Iterable[Mapping]) -> list[tuple[str | None, str]]:
    return [(record.get("pred"), record["gold"]) for record in records]


def accuracy(records: Sequence[Mapping]) -> float:
    pairs = _pairs(records)
    if not pairs:
        return 0.0
    return sum(1 for pred, gold in pairs if pred == gold) / len(pairs)


def macro_f1(records: Sequence[Mapping]) -> float:
    """逐类 F1 的算术平均（未预测到的类 F1 记 0）。"""
    pairs = _pairs(records)
    if not pairs:
        return 0.0
    labels = sorted({gold for _pred, gold in pairs})
    scores: list[float] = []
    for label in labels:
        tp = sum(1 for pred, gold in pairs if pred == label and gold == label)
        fp = sum(1 for pred, gold in pairs if pred == label and gold != label)
        fn = sum(1 for pred, gold in pairs if pred != label and gold == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(f1)
    return sum(scores) / len(scores)


def per_class_report(records: Sequence[Mapping]) -> dict[str, dict[str, float]]:
    pairs = _pairs(records)
    report: dict[str, dict[str, float]] = {}
    for label in sorted({gold for _pred, gold in pairs}):
        total = sum(1 for _pred, gold in pairs if gold == label)
        correct = sum(1 for pred, gold in pairs if gold == label and pred == label)
        report[label] = {
            "support": float(total),
            "recall": round(correct / total, 4) if total else 0.0,
        }
    return report


def majority_baseline(records: Sequence[Mapping]) -> float:
    """常量预测（永远答最常见的类）的准确率——低于它的模型等于没学到东西。"""
    golds = [record["gold"] for record in records]
    if not golds:
        return 0.0
    counts = collections.Counter(golds)
    return max(counts.values()) / len(golds)


def summarize(records: Sequence[Mapping]) -> MetricSummary:
    matched = [pred for pair in _pairs(records) for pred in [pair[0]] if pred is not None]
    return MetricSummary(
        n=len(records),
        n_matched=len(matched),
        accuracy=round(accuracy(records), 4),
        macro_f1=round(macro_f1(records), 4),
        n_invalid=len(records) - len(matched),
        per_class=per_class_report(records),
        majority_baseline=round(majority_baseline(records), 4),
    )


def group_by(records: Sequence[Mapping], key: str) -> dict[str, MetricSummary]:
    """按某个字段（如 task）分组统计。"""
    buckets: dict[str, list[Mapping]] = {}
    for record in records:
        buckets.setdefault(str(record.get(key) or "unknown"), []).append(record)
    return {name: summarize(items) for name, items in sorted(buckets.items())}
