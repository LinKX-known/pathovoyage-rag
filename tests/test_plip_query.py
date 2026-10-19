"""PLIP 查询构造与逐 crop 判据切片的测试（纯离线，不需要 API 与 WSI）。"""

from __future__ import annotations

from giant.knowledge.cards import criteria_slice
from giant.knowledge.plip_query import (
    LAST_RESORT_QUERY,
    build_dimensions,
    build_query_text,
    dimension_texts,
)
from giant.knowledge.schemas import EvidenceBundle, Passage


def _curated_bundle(*texts: str) -> EvidenceBundle:
    return EvidenceBundle(
        passages=[
            Passage(id=f"c{i}", text=text, source="curated", title="Curated")
            for i, text in enumerate(texts)
        ]
    )


# --------------------------------------------------------------------------- 维度构造


def test_curated_phrase_preferred_for_vascular_invasion() -> None:
    bundle = _curated_bundle(
        "脉管侵犯（lymphovascular invasion, LVI）指肿瘤细胞出现在内皮衬覆的腔隙内。"
    )
    dims = build_dimensions("有没有脉管瘤栓", bundle)
    assert dims, "应能构造出维度"
    assert dims[0].source == "curated"
    assert "lymphovascular" in dims[0].text.lower()
    assert all("mucinous" not in dim.text.lower() for dim in dims)


def test_curated_phrase_preferred_for_tumor_budding() -> None:
    bundle = _curated_bundle(
        "肿瘤出芽（tumor budding）指浸润前沿单个或不超过 4 个细胞的肿瘤细胞簇。"
    )
    dims = build_dimensions("肿瘤出芽怎么分级", bundle)
    assert dims[0].source == "curated"
    assert "budding" in dims[0].text.lower()


def test_fallback_to_lexicon_when_no_curated_phrase() -> None:
    # 空证据：只能靠术语表；「切缘」在判据卡里没有现成的英文短语
    dims = build_dimensions("切缘是否干净（R0）", EvidenceBundle())
    assert dims and dims[0].source == "lexicon"
    assert "margin" in dims[0].text.lower()


def test_dimensions_are_normalised_to_one() -> None:
    dims = build_dimensions("组织学亚型是什么？", EvidenceBundle())
    assert dims
    assert abs(sum(dim.weight for dim in dims) - 1.0) < 1e-6


def test_no_signal_returns_empty_and_last_resort_text() -> None:
    # 纯英文且不含任何病理术语 → 构造不出维度
    assert build_dimensions("hello there friend", EvidenceBundle()) == []
    assert build_query_text("hello there friend", EvidenceBundle()) == LAST_RESORT_QUERY


def test_dimension_texts_shape() -> None:
    dims = build_dimensions("有没有脉管瘤栓", EvidenceBundle())
    pairs = dimension_texts(dims)
    assert all(isinstance(text, str) and isinstance(weight, float) for text, weight in pairs)


# --------------------------------------------------------------------------- 逐 crop 判据切片


def test_criteria_slice_differs_by_magnification() -> None:
    bundle = _curated_bundle(
        "低倍看组织结构与生长模式（architectural pattern, growth pattern）。",
        "高倍看核异型与核分裂象（nuclear atypia, mitotic figures）。",
    )
    low = criteria_slice(bundle, "low")
    high = criteria_slice(bundle, "high")
    assert "architectural" in low.lower() or "结构" in low
    assert "nuclear" in high.lower() or "核" in high
    assert low != high


def test_criteria_slice_respects_budget() -> None:
    bundle = _curated_bundle("核异型" + "描述" * 400)
    sliced = criteria_slice(bundle, "high", max_chars=120)
    assert 0 < len(sliced) <= 120


def test_criteria_slice_empty_inputs() -> None:
    assert criteria_slice(None, "low") == ""
    assert criteria_slice(EvidenceBundle(), "high") == ""


# --------------------------------------------------------------------------- 回归保护


def test_depth_question_does_not_mismatch_perineural_invasion() -> None:
    """回归：只共享 invasion 一个词的短语不得被选中（曾把「浸润深度」错配成「神经侵犯」）。"""
    bundle = _curated_bundle(
        "神经侵犯（perineural invasion, PNI）指肿瘤细胞沿神经束膜浸润。",
        "浸润深度（depth of invasion）指肿瘤穿透肠壁的层次。",
    )
    dims = build_dimensions("浸润深度到哪一层", bundle)
    assert dims, "应能构造出维度"
    assert "perineural" not in dims[0].text.lower()


def test_lexicon_dimensions_have_no_duplicate_words() -> None:
    dims = build_dimensions("切缘是否干净（R0）", EvidenceBundle())
    text = dims[0].text.lower()
    words = text.split()
    assert len(words) == len(set(words)), f"存在重复词: {text}"
