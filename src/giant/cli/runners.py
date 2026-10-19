"""创建模型与智能体，执行单次或多次推理并汇总答案。"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from giant.utils.logging import get_logger

if TYPE_CHECKING:
    from giant.cli.main import Mode, Provider


@dataclass
class InferenceResult:
    """保存一次推理编排后的答案和运行统计。"""

    success: bool
    answer: str
    total_cost: float
    total_tokens: int = 0
    trajectory: Any | None = None
    error_message: str | None = None
    runs_answers: list[str] = field(default_factory=list)
    agreement: float = 1.0


def run_single_inference(
    *,
    wsi_path: Path,
    question: str,
    mode: Mode,
    provider: Provider,
    model: str,
    max_steps: int,
    strict_font_check: bool = False,
    enforce_fixed_iterations: bool = False,
    runs: int,
    budget_usd: float,
    verbose: int,
    prescreen: bool = False,
    prescreen_k: int = 5,
    neuro_symbolic: bool = False,
    patho_r1: bool = False,
    rag: bool = False,
    rag_corpus: Path | None = None,
    rag_k: int | None = None,
) -> InferenceResult:
    """创建模型服务并执行指定次数的推理，按预算汇总结果。"""
    from giant.cli.main import Mode

    if mode != Mode.giant:
        raise ValueError(f"Unsupported inference mode: {mode}")
    get_logger(__name__).info("Running inference", wsi=str(wsi_path), runs=runs)
    _ = verbose
    return _run_giant_mode(
        wsi_path=wsi_path,
        question=question,
        provider=provider,
        model=model,
        max_steps=max_steps,
        strict_font_check=strict_font_check,
        enforce_fixed_iterations=enforce_fixed_iterations,
        runs=runs,
        budget_usd=budget_usd,
        prescreen=prescreen,
        prescreen_k=prescreen_k,
        neuro_symbolic=neuro_symbolic,
        patho_r1=patho_r1,
        rag=rag,
        rag_corpus=rag_corpus,
        rag_k=rag_k,
    )


def _run_giant_mode(
    *,
    wsi_path: Path,
    question: str,
    provider: Provider,
    model: str,
    max_steps: int,
    strict_font_check: bool = False,
    enforce_fixed_iterations: bool = False,
    runs: int,
    budget_usd: float,
    prescreen: bool = False,
    prescreen_k: int = 5,
    neuro_symbolic: bool = False,
    patho_r1: bool = False,
    rag: bool = False,
    rag_corpus: Path | None = None,
    rag_k: int | None = None,
) -> InferenceResult:

    from giant.agent import AgentConfig, GIANTAgent
    from giant.llm import create_provider

    logger = get_logger(__name__)
    llm = create_provider(provider.value, model=model)

    # 知识层：只有显式开启时才构造，语料在首次检索时懒加载。
    knowledge = None
    if rag:
        from giant.config import settings
        from giant.knowledge.service import KnowledgeService

        knowledge = KnowledgeService(
            rag_corpus or settings.RAG_CORPUS_PATH,
            top_k=rag_k or settings.RAG_TOP_K,
            min_score=settings.RAG_MIN_SCORE,
            min_coverage=settings.RAG_MIN_COVERAGE,
            relative_cutoff=settings.RAG_RELATIVE_CUTOFF,
        )
        logger.info("知识检索已启用", corpus=str(rag_corpus or settings.RAG_CORPUS_PATH))
        if not prescreen and not patho_r1:
            # 避免「开了 RAG 却什么都没发生」的静默空转
            logger.warning(
                "RAG 已开启但未开 --prescreen / --patho-r1：证据只会写入轨迹，"
                "不会影响 PLIP 查询、观察判据与仲裁"
            )

    candidate_regions = None
    candidate_centers = None
    plip_plan = None
    if prescreen:
        from giant.llm.plip_planner import plan_retrieval
        from giant.vision.plip import prescreen_candidates

        # 问题驱动的预筛：先用千问规划检索计划（意图分类 + 维度拆解），再按维度
        # 召回候选框；规划失败则用「判据卡短语 → 术语表 → 固定兜底」的确定性维度。
        planning_evidence = None
        plip_source = "none"
        if knowledge is not None:
            try:
                bundle = knowledge.retrieve(question, stage="planning")
                from giant.knowledge.gate import apply as apply_gate

                gated, decision = apply_gate(bundle, question)
                if decision.allows_injection:
                    planning_evidence = gated
                    plip_source = "llm+evidence"
                else:
                    logger.info(
                        "预筛阶段闸门拦截：不注入证据",
                        relevance=decision.relevance,
                        reason=decision.reasons[:1],
                    )
            except Exception as e:  # 检索失败不阻断预筛
                logger.warning("知识检索失败，预筛跳过 RAG 注入", error=str(e))

        dimensions = None
        try:
            plan = asyncio.run(plan_retrieval(question, evidence=planning_evidence))
            dimensions = [(d.text, d.weight) for d in plan.dimensions]
            plip_plan = {
                "intent": plan.intent,
                "source": plip_source if planning_evidence is not None else "llm",
                "dimensions": [
                    {"name": d.name, "text": d.text, "weight": d.weight}
                    for d in plan.dimensions
                ],
            }
            logger.info(
                "PLIP 检索规划",
                intent=plan.intent,
                names=[d.name for d in plan.dimensions],
            )
        except Exception as e:  # 规划失败不阻断预筛，改用确定性维度
            logger.warning("PLIP 检索规划失败，改用确定性维度", error=str(e))
            try:
                from giant.knowledge.plip_query import build_dimensions

                fallback = build_dimensions(question, planning_evidence)
                if fallback:
                    dimensions = [(dim.text, dim.weight) for dim in fallback]
                    plip_plan = {
                        "intent": "fallback",
                        "source": "deterministic",
                        "dimensions": [
                            {"name": dim.source, "text": dim.text, "weight": dim.weight}
                            for dim in fallback
                        ],
                    }
            except Exception as inner:  # 连确定性维度都失败时才走 PLIP 默认打分
                logger.warning("确定性维度构造失败，回退默认打分", error=str(inner))

        candidate_regions, candidate_centers = prescreen_candidates(
            wsi_path, k=prescreen_k, dimensions=dimensions
        )

    async def run_once(*, config: AgentConfig) -> Any:
        agent = GIANTAgent(
            wsi_path=wsi_path,
            question=question,
            llm_provider=llm,
            config=config,
        )
        return await agent.run()

    async def run_all() -> list[Any]:
        results: list[Any] = []
        total = 0.0

        for _i in range(runs):
            remaining_budget = None
            if budget_usd > 0:
                remaining_budget = max(budget_usd - total, 0.0)
                if remaining_budget <= 0:
                    break

            result = await run_once(
                config=AgentConfig(
                    max_steps=max_steps,
                    budget_usd=remaining_budget,
                    strict_font_check=strict_font_check,
                    enforce_fixed_iterations=enforce_fixed_iterations,
                    candidate_regions=candidate_regions,
                    candidate_centers=candidate_centers,
                    enable_neuro_symbolic=neuro_symbolic,
                    enable_patho_r1=patho_r1,
                    plip_plan=plip_plan,
                    knowledge=knowledge,
                )
            )
            results.append(result)
            total += result.total_cost

            if budget_usd > 0 and total >= budget_usd:
                break

        return results

    all_results = asyncio.run(run_all())
    total_cost = sum(r.total_cost for r in all_results)
    total_tokens = sum(r.total_tokens for r in all_results)

    return _summarize_runs(
        run_results=all_results,
        total_cost=total_cost,
        total_tokens=total_tokens,
    )


def _summarize_runs(
    *,
    run_results: list[Any],
    total_cost: float,
    total_tokens: int,
) -> InferenceResult:
    if not run_results:
        return InferenceResult(
            success=False,
            answer="",
            total_cost=total_cost,
            total_tokens=total_tokens,
            trajectory=None,
            error_message="No runs executed",
        )

    candidates = [r for r in run_results if r.success and r.answer]
    if not candidates:
        candidates = [r for r in run_results if r.answer] or run_results

    answers = [r.answer for r in candidates if r.answer]
    final_answer = answers[0] if answers else ""
    agreement = 1.0
    winning = candidates[0]

    if answers:
        counts = Counter(answers)
        max_count = max(counts.values())
        winners = {a for a, c in counts.items() if c == max_count}
        final_answer = next(a for a in answers if a in winners)
        agreement = counts[final_answer] / len(answers)
        winning = next(r for r in candidates if r.answer == final_answer)

    return InferenceResult(
        success=winning.success,
        answer=final_answer,
        total_cost=total_cost,
        total_tokens=total_tokens,
        trajectory=winning.trajectory,
        error_message=winning.error_message,
        runs_answers=answers,
        agreement=agreement,
    )
