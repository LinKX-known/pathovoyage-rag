"""知识层（RAG）测试：不依赖任何 API Key、不依赖 WSI 文件即可运行。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from giant.agent.trajectory import Trajectory, Turn
from giant.knowledge.bm25 import BM25Index
from giant.knowledge.cards import citation_check, criteria_card, evidence_block
from giant.knowledge.corpus import build_corpus_from_items, read_corpus
from giant.knowledge.lexicon import expand_query, has_cjk
from giant.knowledge.schemas import EvidenceBundle, Passage
from giant.knowledge.selection import SourceSpec, load_spec, select_items
from giant.knowledge.service import KnowledgeError, KnowledgeService
from giant.knowledge.tokenize import tokenize
from giant.llm.protocol import FinalAnswerAction, StepResponse

# --------------------------------------------------------------------------- 分词


def test_tokenize_drops_stopwords_and_stems_plurals() -> None:
    tokens = tokenize("The tumors of the colon are invasive")
    assert "the" not in tokens and "of" not in tokens
    assert "tumor" in tokens  # 复数被还原
    assert "colon" in tokens


def test_tokenize_emits_cjk_bigrams() -> None:
    tokens = tokenize("黏液腺癌")
    assert "黏液" in tokens and "液腺" in tokens and "腺癌" in tokens


# --------------------------------------------------------------------------- 术语表


def test_expand_query_prefers_longest_term() -> None:
    query, hits = expand_query("黏液腺癌的组织学亚型是什么")
    assert "mucinous" in query and "adenocarcinoma" in query
    # 「腺癌」「癌」被更长的「黏液腺癌」覆盖，不重复展开
    assert "黏液腺癌" in hits
    assert "腺癌" not in hits


def test_expand_query_passthrough_for_english() -> None:
    assert has_cjk("colorectal adenocarcinoma") is False
    query, hits = expand_query("colorectal adenocarcinoma")
    assert hits == [] and query == "colorectal adenocarcinoma"


# --------------------------------------------------------------------------- BM25


def _passages(*texts: str) -> list[Passage]:
    return [Passage(id=f"p{i}", text=text, title="T") for i, text in enumerate(texts)]


def test_bm25_ranks_relevant_passage_first() -> None:
    index = BM25Index(
        _passages(
            "Mucinous adenocarcinoma contains extracellular mucin pools.",
            "Breast fibroadenoma is a benign biphasic tumor.",
            "Lung adenocarcinoma shows gland formation and mucin production.",
        )
    )
    results = index.search("mucinous adenocarcinoma mucin pool", k=2)
    assert results and results[0][0].id == "p0"
    assert results[0][1] > 0


def test_bm25_coverage_gate_filters_weak_hits() -> None:
    index = BM25Index(_passages("invasion of the extracellular matrix", "tumor budding"))
    # 覆盖率 1.0 要求全部查询词命中，只有第一段命中 invasion/matrix
    results = index.search("invasion matrix budding", k=5, min_coverage=1.0)
    assert [passage.id for passage, _ in results] == []


# --------------------------------------------------------------------------- 语料构建


def test_build_corpus_drops_pdf_noise_and_splits_windows(tmp_path: Path) -> None:
    items = [
        # 词粘连（PDF 噪声）：空格占比过低，应被丢弃
        {"id": "noise_1", "content": "Rupturedovarianmucinoustumorsmayseedtheperitoneum" * 3},
        # 正常长文本：应被切成多个 ≤600 字符的窗口
        {
            "id": "good_1",
            "content": " ".join(["Colorectal adenocarcinoma shows gland formation."] * 40),
            "title": "Pathoma",
        },
    ]
    out = tmp_path / "corpus.jsonl"
    stats = build_corpus_from_items(items, out, window_chars=300, min_chars=40)

    assert stats.n_dropped_noise == 1
    assert stats.n_passages > 1
    passages = read_corpus(out)
    assert all(len(passage.text) <= 300 for passage in passages)
    assert all(passage.id.startswith("good_1#w") for passage in passages)
    assert stats.corpus_version


def test_read_corpus_roundtrip_keeps_meta(tmp_path: Path) -> None:
    out = tmp_path / "c.jsonl"
    build_corpus_from_items(
        [
            {
                "id": "x_1",
                "content": "colorectal adenocarcinoma with abundant extracellular mucin "
                * 4,
                "title": "T",
            }
        ],
        out,
        min_chars=40,
    )
    [passage] = read_corpus(out)
    assert passage.meta == {"raw_id": "x_1", "window": "0"}


# --------------------------------------------------------------------------- 选择规则


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_select_items_by_id_range_and_regex(tmp_path: Path) -> None:
    raw = tmp_path / "rag_raw"
    _write_jsonl(
        raw / "Book.jsonl",
        [
            {"id": "Book_1", "content": "colorectal polyp"},
            {"id": "Book_2", "content": "unrelated topic"},
            {"id": "Book_3", "content": "colorectal carcinoma"},
        ],
    )
    specs = [SourceSpec(file="Book.jsonl", title="Book", id_range=(2, 3))]
    selected = list(select_items({"rag_raw": raw}, specs))
    assert [item["id"] for item in selected] == ["Book_2", "Book_3"]

    specs = [SourceSpec(file="Book.jsonl", title="Book", regex="polyp")]
    selected = list(select_items({"rag_raw": raw}, specs))
    assert [item["id"] for item in selected] == ["Book_1"]


def test_load_spec_rejects_reversed_range(tmp_path: Path) -> None:
    spec = tmp_path / "sources.json"
    spec.write_text(
        json.dumps({"sources": [{"file": "x.jsonl", "id_range": [9, 1]}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_spec(spec)


def test_select_items_takes_all_when_no_rule(tmp_path: Path) -> None:
    curated = tmp_path / "corpus"
    _write_jsonl(curated / "curated.jsonl", [{"id": "c1", "content": "判据一段"}])
    specs = [SourceSpec(file="curated.jsonl", title="Curated", source="curated", subdir="corpus")]
    selected = list(select_items({"corpus": curated}, specs))
    assert len(selected) == 1 and selected[0]["_source"] == "curated"


# --------------------------------------------------------------------------- 服务


@pytest.fixture()
def service(tmp_path: Path) -> KnowledgeService:
    corpus = tmp_path / "corpus.jsonl"
    build_corpus_from_items(
        [
            {
                "id": "curated_1",
                "content": "黏液腺癌要求细胞外黏液占肿瘤体积 50% 以上，黏液池内漂浮肿瘤细胞簇。",
                "title": "Curated",
                "_source": "curated",
            },
            {
                "id": "book_1",
                "content": "Mucinous adenocarcinoma of the colon shows abundant extracellular mucin.",
                "title": "Pathoma",
            },
        ],
        corpus,
        min_chars=20,
    )
    return KnowledgeService(corpus, top_k=2)


def test_service_retrieve_hits_curated_criteria(service: KnowledgeService) -> None:
    bundle = service.retrieve("黏液腺癌的组织学亚型")
    assert isinstance(bundle, EvidenceBundle)
    assert bundle.passages[0].id.startswith("curated_1")
    assert bundle.corpus_version
    assert bundle.stage == "planning"


def test_service_raises_actionable_error_for_missing_corpus(tmp_path: Path) -> None:
    service = KnowledgeService(tmp_path / "missing.jsonl")
    with pytest.raises(KnowledgeError) as excinfo:
        service.retrieve("任何问题")
    assert "giant knowledge build" in str(excinfo.value)


def test_service_empty_query_returns_empty_bundle(service: KnowledgeService) -> None:
    assert service.retrieve("   ").is_empty


# --------------------------------------------------------------------------- 提示词压缩


def test_criteria_card_respects_budget_and_numbering() -> None:
    bundle = EvidenceBundle(
        query="q",
        passages=[
            Passage(id=f"p{i}", text="word " * 500, title="T") for i in range(3)
        ],
    )
    card = criteria_card(bundle, max_chars=400, max_items=3)
    assert len(card) <= 460  # 预算 + 结尾提示语
    assert "[E1]" in card


def test_evidence_block_empty_when_no_passages() -> None:
    assert evidence_block(EvidenceBundle(query="q")) == ""


# --------------------------------------------------------------------------- 轨迹兼容


def test_trajectory_and_turn_defaults_keep_backward_compat() -> None:
    trajectory = Trajectory(wsi_path="slide.svs", question="q")
    assert trajectory.rag_evidence == [] and trajectory.rag_corpus_version == ""

    turn = Turn(
        step_index=0,
        image_base64="",
        response=StepResponse(
            reasoning="r", action=FinalAnswerAction(answer_text="a")
        ),
    )
    assert turn.evidence_ids == []


def test_trajectory_accepts_rag_evidence() -> None:
    trajectory = Trajectory(
        wsi_path="slide.svs",
        question="q",
        rag_evidence=[Passage(id="p1", text="t")],
        rag_corpus_version="v1",
    )
    assert trajectory.rag_evidence[0].id == "p1"


# --------------------------------------------------------------------------- 检索闸门


def test_gate_marks_relevant_for_curated_criteria() -> None:
    from giant.knowledge.gate import assess

    bundle = EvidenceBundle(
        passages=[
            Passage(
                id="c1",
                text="黏液腺癌要求细胞外黏液占肿瘤体积 50% 以上。",
                source="curated",
            )
        ]
    )
    decision = assess(bundle, "黏液腺癌的组织学亚型是什么")
    assert decision.relevance in {"relevant", "partially_relevant"}
    assert decision.allows_injection


def test_gate_drops_cross_organ_noise() -> None:
    from giant.knowledge.gate import apply

    bundle = EvidenceBundle(
        passages=[
            Passage(
                id="curated_ok",
                text="黏液腺癌要求细胞外黏液占肿瘤体积 50% 以上。",
                source="curated",
            ),
            Passage(
                id="noise",
                text="Aneurysms and dissections result from structural weakness of "
                "the vessel wall caused by loss of smooth muscle cells.",
                source="textbook",
                title="Pathology_Robbins",
            ),
        ]
    )
    gated, decision = apply(bundle, "黏液腺癌的组织学亚型是什么")
    assert [passage.id for passage in gated.passages] == ["curated_ok"]
    assert decision.dropped == 1


def test_gate_blocks_injection_when_irrelevant() -> None:
    from giant.knowledge.gate import apply

    bundle = EvidenceBundle(
        passages=[
            Passage(id="x", text="Breast fibroadenoma is a benign tumor.", title="T")
        ]
    )
    gated, decision = apply(bundle, "有没有脉管瘤栓")
    assert decision.allows_injection is False
    assert gated.is_empty


def test_gate_blocks_out_of_scope_organ_question() -> None:
    from giant.knowledge.gate import assess

    bundle = EvidenceBundle(
        passages=[
            Passage(
                id="p",
                text="colorectal adenocarcinoma histologic subtype classification",
                title="T",
            )
        ]
    )
    # 语料是结直肠专用的；问肺时即使词面重合也必须拦下来
    decision = assess(bundle, "肺腺癌的组织学亚型是什么")
    assert decision.relevance == "irrelevant"
    assert decision.allows_injection is False
    assert any("out_of_scope_organ" in reason for reason in decision.reasons)


def test_gate_allows_colorectal_question_with_other_organ() -> None:
    from giant.knowledge.gate import assess

    bundle = EvidenceBundle(
        passages=[
            Passage(
                id="p",
                text="结直肠癌肝转移的切缘与 R0 判定，mucinous adenocarcinoma。",
                source="curated",
            )
        ]
    )
    decision = assess(bundle, "结直肠癌肝转移的切缘要求是什么")
    assert decision.allows_injection is True


# --------------------------------------------------------------------------- 引用统计


def test_citation_check_counts_used_unused_and_invalid() -> None:
    from giant.knowledge.schemas import CitationReport

    bundle = EvidenceBundle(
        passages=[Passage(id=f"p{i}", text="t") for i in range(3)]
    )
    answer = "结论 A [E1]，依据 [E3]；另外还有 [E9]。"
    report = citation_check(answer, bundle)
    assert isinstance(report, CitationReport)
    assert report.used == ["E1", "E3"]
    assert report.unused == ["E2"]
    assert report.invalid == ["E9"]
    assert report.coverage == pytest.approx(round(2 / 3, 3))


def test_citation_check_without_citations() -> None:
    bundle = EvidenceBundle(passages=[Passage(id="p1", text="t")])
    report = citation_check("一句没有任何编号的结论", bundle)
    assert report.has_citations is False and report.unused == ["E1"]


# --------------------------------------------------------------------------- 结构化仲裁解析


def test_parse_arbitration_extracts_structured_fields() -> None:
    from giant.llm.plip_planner import _parse_arbitration

    raw = """```json
{"answer": "浸润性黏液腺癌",
 "weighted_evidence": [{"region": "(1,2,3,4)", "opinion": "黏液湖>50%",
                        "weight": 0.8, "why": "多区域一致"}],
 "conflicts": ["区域 A 判高分化，区域 B 判黏液腺癌"],
 "minority": ["区域 A 的高分化结论"],
 "citations": ["E1", "E2", "not-an-id"]}
```"""
    result = _parse_arbitration(raw)
    assert result.structured is True
    assert result.answer == "浸润性黏液腺癌"
    assert result.weighted_evidence[0].weight == pytest.approx(0.8)
    assert result.citations == ["E1", "E2"]
    assert len(result.conflicts) == 1


def test_parse_arbitration_falls_back_to_plain_text() -> None:
    from giant.llm.plip_planner import _parse_arbitration

    result = _parse_arbitration("这是一段自由文本结论，没有 JSON。")
    assert result.structured is False
    assert result.answer == "这是一段自由文本结论，没有 JSON。"
    assert result.weighted_evidence == []


def test_parse_arbitration_clamps_weight() -> None:
    from giant.llm.plip_planner import _parse_arbitration

    raw = '{"answer": "A", "weighted_evidence": [{"weight": 3.7}, {"weight": "bad"}]}'
    result = _parse_arbitration(raw)
    assert [item.weight for item in result.weighted_evidence] == [1.0, 0.0]


# --------------------------------------------------------------------------- 提示词注入（离线验证）


class _FakeQwenProvider:
    """捕获提示词的假客户端：验证证据真的被写进了送给模型的文本。"""

    captured: list[str] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    async def complete_text(self, *, system: str, user: str, max_tokens: int) -> str:
        type(self).captured.append(user)
        return (
            '{"criteria": "c", "low_task": "l", "medium_task": "m", '
            '"high_task": "h"}'
        )


@pytest.fixture()
def bundle() -> EvidenceBundle:
    return EvidenceBundle(
        query="q",
        passages=[
            Passage(
                id="curated_1",
                text="黏液腺癌要求细胞外黏液占肿瘤体积 50% 以上。",
                title="Curated",
            )
        ],
        corpus_version="v1",
    )


def test_plan_observation_prompt_contains_evidence(
    monkeypatch: pytest.MonkeyPatch, bundle: EvidenceBundle
) -> None:
    import asyncio

    from giant.llm import plip_planner

    _FakeQwenProvider.captured = []
    monkeypatch.setattr(plip_planner, "QwenProvider", _FakeQwenProvider)

    plan = asyncio.run(plip_planner.plan_observation("亚型是什么", evidence=bundle))

    assert plan.low_task == "l"
    prompt = _FakeQwenProvider.captured[0]
    assert "[E1]" in prompt and "黏液腺癌" in prompt


def test_arbitration_prompt_contains_evidence_block(
    monkeypatch: pytest.MonkeyPatch, bundle: EvidenceBundle
) -> None:
    import asyncio

    from giant.llm import plip_planner

    _FakeQwenProvider.captured = []
    monkeypatch.setattr(plip_planner, "QwenProvider", _FakeQwenProvider)

    asyncio.run(
        plip_planner.arbitrate_patho_r1("问题", ["专家结论一"], evidence=bundle)
    )

    prompt = _FakeQwenProvider.captured[0]
    assert "# 指南证据" in prompt and "[E1]" in prompt
    assert "标注对应编号" in prompt
