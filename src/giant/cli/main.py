"""解析命令行参数，启动单张切片的推理并输出结果。"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from giant import __version__
from giant.config import settings
from giant.llm.model_registry import get_default_model
from giant.utils.logging import configure_logging, get_logger


app = typer.Typer(
    name="giant",
    help="Pathovoyage 病理智航：多尺度动态探索病理诊断智能体",
    add_completion=False,
    pretty_exceptions_show_locals=False,  # 报错时不打印局部变量（简洁）
)


class Mode(str, Enum):
    """限制命令行可选的推理模式。"""

    giant = "giant"


class Provider(str, Enum):
    """限制命令行可选的模型服务提供方。"""

    openai = "openai"
    anthropic = "anthropic"
    qwen = "qwen"


# =============================================================================
# 命令
# =============================================================================


@app.command()
def version(
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON")] = False,
) -> None:

    if json_output:
        typer.echo(json.dumps({"version": __version__}))
    else:
        typer.echo(f"giant {__version__}")


# =============================================================================
# 知识层（RAG）
# =============================================================================

_DEFAULT_RAG_CORPUS = Path("data/corpus/textbooks_colorectal.jsonl")

knowledge_app = typer.Typer(
    name="knowledge",
    help="病理知识语料（RAG）：构建索引、查看状态、试检索",
    add_completion=False,
)
app.add_typer(knowledge_app, name="knowledge")


@knowledge_app.command("build")
def knowledge_build(
    spec: Annotated[
        Path, typer.Option("--spec", help="语料策展规则（sources.json）")
    ] = Path("data/corpus/sources.json"),
    raw_dir: Annotated[
        Path, typer.Option("--raw-dir", help="MedRAG textbooks 原始 jsonl 目录")
    ] = Path("data/rag_raw"),
    out: Annotated[
        Path, typer.Option("--out", help="输出语料路径")
    ] = _DEFAULT_RAG_CORPUS,
) -> None:
    """按 spec 选择语源、去噪、切窗，生成可检索的语料文件。"""
    from giant.knowledge.corpus import build_corpus_from_items
    from giant.knowledge.selection import load_spec, select_items

    specs, meta = load_spec(spec)
    dirs = {"rag_raw": raw_dir, "corpus": spec.parent}
    items = list(select_items(dirs, specs))
    if not items:
        typer.echo("没有选中任何片段：检查 --spec / --raw-dir 是否正确", err=True)
        raise typer.Exit(1)

    stats = build_corpus_from_items(
        items,
        out,
        window_chars=int(meta.get("window_chars", 600)),
        min_chars=int(meta.get("min_chars", 100)),
    )
    typer.echo(f"语料: {stats.corpus_path}")
    typer.echo(f"片段: {stats.n_passages}（丢弃噪声片段 {stats.n_dropped_noise}）")
    typer.echo(f"字符: {stats.total_chars}")
    typer.echo(f"指纹: {stats.corpus_version}")
    typer.echo(f"语源: {', '.join(stats.titles)}")


@knowledge_app.command("status")
def knowledge_status(
    corpus: Annotated[Path, typer.Option("--corpus", help="语料路径")] = _DEFAULT_RAG_CORPUS,
) -> None:
    """查看语料规模、语源与指纹。"""
    from giant.knowledge.service import KnowledgeService

    stats = KnowledgeService(corpus).stats
    typer.echo(f"语料: {stats.corpus_path}")
    typer.echo(f"片段: {stats.n_passages}")
    typer.echo(f"字符: {stats.total_chars}")
    typer.echo(f"指纹: {stats.corpus_version}")
    typer.echo(f"语源: {', '.join(stats.titles)}")


@knowledge_app.command("search")
def knowledge_search(
    query: Annotated[str, typer.Argument(help="检索问题（中文会自动展开为英文术语）")],
    k: Annotated[int, typer.Option("--k", help="返回条数")] = 5,
    corpus: Annotated[Path, typer.Option("--corpus", help="语料路径")] = _DEFAULT_RAG_CORPUS,
) -> None:
    """试检索：用于人工检查语料与排序质量。"""
    from giant.knowledge.service import KnowledgeService

    service = KnowledgeService(corpus, top_k=k)
    passages = service.search(query, k=k)
    if not passages:
        typer.echo("没有命中。语料可能不覆盖该问题，或阈值过严。")
        return
    for index, passage in enumerate(passages, start=1):
        text = " ".join(passage.text.split())
        typer.echo(f"[E{index}] {passage.score:.2f}  {passage.title} ({passage.source})")
        typer.echo(f"    {text[:220]}")


@app.command()
def run(
    wsi_path: Annotated[
        Path,
        typer.Argument(
            exists=True,  # 必须已存在
            file_okay=True,  # 必须是文件
            dir_okay=False,  # 不能是目录
            readable=True,  # 必须可读
            help="Path to WSI file (.svs, .ndpi, .tiff)",
        ),
    ],
    question: Annotated[
        str, typer.Option("--question", "-q", help="Question to answer about the slide")
    ],
    mode: Annotated[
        Mode, typer.Option("--mode", "-m", help="Evaluation mode")
    ] = Mode.giant,
    provider: Annotated[
        Provider, typer.Option("--provider", "-p", help="LLM provider")
    ] = Provider.openai,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Model name (see docs/models/model-registry.md)"),
    ] = None,
    max_steps: Annotated[
        int, typer.Option("--max-steps", "-T", help="Max navigation steps")
    ] = settings.MAX_ITERATIONS,
    strict_font_check: Annotated[
        bool,
        typer.Option(
            "--strict-font-check/--no-strict-font-check",
            help="Fail if TrueType fonts are unavailable for axis labels",
        ),
    ] = False,
    enforce_fixed_iterations: Annotated[
        bool,
        typer.Option(
            "--enforce-fixed-iterations/--no-enforce-fixed-iterations",
            help="Reject early answers (paper-fidelity fixed-iteration mode)",
        ),
    ] = False,
    runs: Annotated[
        int, typer.Option("--runs", "-r", help="Number of runs for majority voting")
    ] = 1,
    budget_usd: Annotated[
        float,
        typer.Option(
            "--budget-usd",
            help="Stop early if total cost exceeds this USD budget (0 disables)",
        ),
    ] = 0.0,
    prescreen: Annotated[
        bool,
        typer.Option(
            "--prescreen/--no-prescreen",
            help="Pre-screen candidate regions with PLIP (hybrid localization)",
        ),
    ] = False,
    prescreen_k: Annotated[
        int, typer.Option("--prescreen-k", help="Number of candidate regions to seed")
    ] = 5,
    neuro_symbolic: Annotated[
        bool,
        typer.Option(
            "--neuro-symbolic/--no-neuro-symbolic",
            help="Enable neuro-symbolic diagnosis (structured features + "
            "magnification-cognition FSM + L1/L2/L3 rule engine)",
        ),
    ] = False,
    patho_r1: Annotated[
        bool,
        typer.Option(
            "--patho-r1/--no-patho-r1",
            help="Enable Patho-R1 specialist reasoning on every crop "
            "(requires a locally-served Patho-R1 via PATHO_R1_BASE_URL)",
        ),
    ] = False,
    rag: Annotated[
        bool,
        typer.Option(
            "--rag/--no-rag",
            help="Enable knowledge retrieval (guides/criteria) injection into "
            "planning and arbitration; build the corpus with `giant knowledge build`",
        ),
    ] = False,
    rag_corpus: Annotated[
        Path | None,
        typer.Option("--rag-corpus", help="Path to the curated knowledge corpus (.jsonl)"),
    ] = None,
    rag_k: Annotated[
        int | None,
        typer.Option("--rag-k", help="Number of evidence passages to retrieve"),
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Save trajectory to JSON")
    ] = None,
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose", "-v", count=True, help="Increase verbosity (-v, -vv, -vvv)"
        ),
    ] = 0,
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON")] = False,
) -> None:
    """按命令行选项运行切片推理，并输出答案或保存轨迹。"""

    from giant.cli.runners import run_single_inference

    _configure_logging(verbose)
    logger = get_logger(__name__)

    effective_model = model or get_default_model(provider.value)

    logger.info(
        "Starting inference",
        wsi=str(wsi_path),
        mode=mode.value,
        provider=provider.value,
    )

    try:

        result = run_single_inference(
            wsi_path=wsi_path,
            question=question,
            mode=mode,
            provider=provider,
            model=effective_model,
            max_steps=max_steps,
            strict_font_check=strict_font_check,
            enforce_fixed_iterations=enforce_fixed_iterations,
            runs=runs,
            budget_usd=budget_usd,
            verbose=verbose,
            prescreen=prescreen,
            prescreen_k=prescreen_k,
            neuro_symbolic=neuro_symbolic,
            patho_r1=patho_r1,
            rag=rag,
            rag_corpus=rag_corpus,
            rag_k=rag_k,
        )

        turns_count = _trajectory_turn_count(result.trajectory)  # 走了几步


        if output is not None:
            trajectory_dict = _trajectory_to_dict(result.trajectory)
            artifact = {
                **trajectory_dict,
                "answer": result.answer,  # 最终答案
                "success": result.success,  # 是否成功
                "error_message": result.error_message,  # 失败原因（诊断用）
                "total_cost": result.total_cost,  # 总花费（美元）
                "mode": mode.value,
                "provider": provider.value,
                "model": effective_model,
                "runs": runs,  # 投票跑了几次
                "agreement": result.agreement,  # 多次投票的一致性
                "runs_answers": result.runs_answers,  # 每次的答案
            }
            output.write_text(json.dumps(artifact, indent=2))
            logger.info("Run artifact saved", path=str(output))


        if json_output:
            output_data = {
                "success": result.success,
                "answer": result.answer,
                "error_message": result.error_message,
                "total_cost": result.total_cost,
                "agreement": result.agreement,
                "turns": turns_count,
            }
            typer.echo(json.dumps(output_data, indent=2))
        else:
            typer.echo(f"\nAnswer: {result.answer}")
            typer.echo(f"Cost: ${result.total_cost:.4f}")
            if runs > 1:  # 只有多轮投票才显示一致性
                typer.echo(f"Agreement: {result.agreement:.0%}")
            if turns_count:
                typer.echo(f"Turns: {turns_count}")
            if result.error_message:  # 失败时打印具体原因，避免「无答案」无从排查
                typer.echo(f"Error: {result.error_message}")

        # 成功返回 0，失败返回 1（作为进程退出码）
        raise typer.Exit(0 if result.success else 1)

    except typer.Exit:
        raise  # 正常退出，直接上抛
    except Exception as e:
        # 任何异常：记日志 + 打印错误 + 退出码 1
        logger.exception("Inference failed")
        if json_output:
            typer.echo(json.dumps({"error": str(e)}))
        else:
            typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:

    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


def _configure_logging(verbose: int) -> None:

    if verbose == 0:
        level = "WARNING"
    elif verbose == 1:
        level = "INFO"
    else:  # 详细日志级别至少为 2。
        level = "DEBUG"

    configure_logging(level=level)


def _trajectory_to_dict(trajectory: object | None) -> dict[str, object]:
    if trajectory is None:
        return {}
    if isinstance(trajectory, dict):
        return dict(trajectory)


    model_dump = getattr(trajectory, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    return {}


def _trajectory_turn_count(trajectory: object | None) -> int:
    turns = getattr(trajectory, "turns", None)
    return len(turns) if isinstance(turns, list) else 0


if __name__ == "__main__":
    app()
