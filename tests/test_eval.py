"""评测层测试：答案对齐与指标统计（纯离线，不依赖 WSI / API）。"""

from __future__ import annotations

import pytest

from giant.eval.matching import match_option, normalize
from giant.eval.metrics import (
    accuracy,
    group_by,
    macro_f1,
    majority_baseline,
    summarize,
)

_OPTIONS = {
    "A": "Diverticulitis",
    "B": "Adenocarcinoma",
    "C": "Tubular adenoma",
    "D": "Normal mucosa",
}


def test_match_letter_only() -> None:
    assert match_option("B", _OPTIONS).letter == "B"
    assert match_option("答案：C", _OPTIONS).letter == "C"
    assert match_option("option D.", _OPTIONS).letter == "D"


def test_match_exact_text() -> None:
    result = match_option("Adenocarcinoma", _OPTIONS)
    assert result.letter == "B" and result.method == "exact"


def test_match_substring_chinese_answer() -> None:
    options = {"A": "黏液腺癌", "B": "高分化腺癌"}
    result = match_option("镜下诊断为黏液腺癌，伴黏液湖形成", options)
    assert result.letter == "A" and result.method == "substring"


def test_match_prefers_exact_over_longer_substring() -> None:
    options = {
        "A": "Moderately differentiated",
        "B": "Moderately differentiated adenocarcinoma",
    }
    assert match_option("Moderately differentiated adenocarcinoma", options).letter == "B"
    assert match_option("Moderately differentiated", options).letter == "A"


def test_match_token_overlap() -> None:
    options = {"A": "Well differentiated", "B": "Poorly differentiated"}
    result = match_option("This tumour is poorly differentiated with high grade nuclei", options)
    assert result.letter == "B"


def test_match_returns_none_when_unrelated() -> None:
    result = match_option("I cannot determine from this image.", _OPTIONS)
    assert result.letter is None and result.method == "no_match"


def test_match_handles_empty_inputs() -> None:
    assert match_option("", _OPTIONS).letter is None
    assert match_option("A", {}).letter is None


def test_normalize_strips_punct_and_case() -> None:
    assert normalize("Stage III, R0.") == "stageiiir0"


# --------------------------------------------------------------------------- 指标


def _records(pairs: list[tuple[str | None, str]], task: str = "Grading") -> list[dict]:
    return [{"pred": pred, "gold": gold, "task": task} for pred, gold in pairs]


def test_accuracy_and_invalid_count() -> None:
    records = _records([("A", "A"), ("B", "C"), (None, "A"), ("A", "A")])
    summary = summarize(records)
    assert summary.n == 4
    assert summary.n_invalid == 1
    assert summary.accuracy == pytest.approx(0.5)


def test_macro_f1_penalises_ignoring_minority_class() -> None:
    # 8 个 A 类答对、2 个 B 类被判成 A → accuracy 0.8，但 A 有 2 个假阳、
    # B 的 F1 = 0，macro-F1 只有 (0.889 + 0.0) / 2 ≈ 0.444，暴露了类别不均衡下的虚高
    records = _records([("A", "A")] * 8 + [("A", "B")] * 2)
    assert accuracy(records) == pytest.approx(0.8)
    assert macro_f1(records) == pytest.approx(0.4444, abs=1e-3)
    assert majority_baseline(records) == pytest.approx(0.8)


def test_per_class_recall_in_summary() -> None:
    records = _records([("A", "A"), ("A", "A"), ("A", "B")])
    summary = summarize(records)
    assert summary.per_class["A"]["recall"] == pytest.approx(1.0)
    assert summary.per_class["B"]["recall"] == pytest.approx(0.0)


def test_group_by_task() -> None:
    records = _records([("A", "A")], task="Grading") + _records(
        [("B", "A")], task="Staging"
    )
    grouped = group_by(records, "task")
    assert grouped["Grading"].accuracy == pytest.approx(1.0)
    assert grouped["Staging"].accuracy == pytest.approx(0.0)


def test_empty_records_are_safe() -> None:
    summary = summarize([])
    assert summary.n == 0 and summary.accuracy == 0.0 and summary.macro_f1 == 0.0
