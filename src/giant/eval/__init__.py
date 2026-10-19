"""评测支持：答案与选项对齐、指标统计、SlideBench 批量评测的公共部件。"""

from giant.eval.matching import MatchResult, match_option, normalize
from giant.eval.metrics import (
    MetricSummary,
    accuracy,
    group_by,
    macro_f1,
    majority_baseline,
    per_class_report,
    summarize,
)

__all__ = [
    "MatchResult",
    "MetricSummary",
    "accuracy",
    "group_by",
    "macro_f1",
    "majority_baseline",
    "match_option",
    "normalize",
    "per_class_report",
    "summarize",
]
