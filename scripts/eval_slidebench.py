"""在 SlideBench 的结直肠子集上批量评测本项目 pipeline，产出可复现的准确率表。

设计要点：
- **断点续跑**：结果按题逐行追加到 JSONL，API 中途失败重跑不会重算已完成题；
- **成本与耗时逐题记录**：基准评测最容易失控的是账单，必须能按题追溯；
- **指标含基线**：accuracy 之外给 macro-F1 与多数类基线，避免类别不均衡造成的虚高；
- **可解释性字段随结果落盘**：检索闸门判定、仲裁引用统计、结构化标记。

用法：
    # 先空跑，检查切片是否齐、题目是否会构造正确
    python scripts/eval_slidebench.py --subset tcga-crc --dry-run

    # 20 题标定单题成本
    python scripts/eval_slidebench.py --subset tcga-crc --limit 20 --tag pilot

    # 全量（RAG 开关各跑一遍，得到配对对比）
    python scripts/eval_slidebench.py --subset tcga-crc --tag rag_off
    python scripts/eval_slidebench.py --subset tcga-crc --rag --tag rag_on
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROMPT_SUFFIX = """

Options:
{options}

请只依据切片所见作答；如果信息不足，给出最可能的一项。
最后一行只输出所选选项的字母（例如：B）。
"""


def _load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_question(item: dict) -> str:
    """把「题目 + 选项」拼成喂给 pipeline 的问题文本。"""
    rendered = "\n".join(
        f"{letter}. {text}" for letter, text in sorted(item["options"].items())
    )
    return f"{item['question']}{PROMPT_SUFFIX.format(options=rendered)}"


def _extract_diagnostics(result: object) -> dict:
    """从 RunResult 里提取轻量字段（绝不含 base64 图像）。"""
    trajectory = getattr(result, "trajectory", None)
    if trajectory is None:
        return {}
    turns = getattr(trajectory, "turns", []) or []
    arbitration = getattr(trajectory, "patho_r1_arbitration", None) or {}
    return {
        "turns": len(turns),
        "rag_gate": getattr(trajectory, "rag_gate", []) or [],
        "final_citation": getattr(trajectory, "final_citation", None),
        "arbitration_structured": arbitration.get("structured"),
        "arbitration_conflicts": len(arbitration.get("conflicts") or []),
        "corpus_version": getattr(trajectory, "rag_corpus_version", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="tcga-crc", help="子集名（data/slidebench/subsets/*.jsonl）")
    parser.add_argument("--subset-file", type=Path, default=None)
    parser.add_argument("--slides-dir", type=Path, default=Path("data/slidebench/slides"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/slidebench/results"))
    parser.add_argument("--tag", default="default", help="实验标签，区分 rag_on / rag_off 等")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tasks", default="", help="只跑指定任务，逗号分隔（如 Grading,Staging）")
    parser.add_argument("--provider", default="qwen", choices=["qwen", "openai", "anthropic"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--budget-usd", type=float, default=0.0)
    parser.add_argument("--prescreen", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--neuro-symbolic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patho-r1", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rag", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rag-corpus", type=Path, default=None)
    parser.add_argument("--rag-k", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    subset_path = args.subset_file or Path(f"data/slidebench/subsets/{args.subset}.jsonl")
    if not subset_path.exists():
        print(f"子集不存在：{subset_path}（先跑 scripts/prepare_slidebench.py）")
        return 1
    items = _load_jsonl(subset_path)
    if args.tasks:
        wanted = {task.strip() for task in args.tasks.split(",") if task.strip()}
        items = [item for item in items if item["task"] in wanted]
    if args.limit > 0:
        items = items[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / f"{args.subset}.{args.tag}.jsonl"
    done_ids = {record["id"] for record in _load_jsonl(results_path)} if results_path.exists() else set()

    plan: list[tuple[dict, Path]] = []
    missing: list[str] = []
    for item in items:
        slide_path = args.slides_dir / f"{item['slide']}.svs"
        if not slide_path.exists():
            missing.append(item["slide"])
            continue
        plan.append((item, slide_path))

    todo = [(item, path) for item, path in plan if item["id"] not in done_ids]
    print(
        f"题目 {len(items)} 条；切片就绪 {len(plan)}；已完成 {len(plan) - len(todo)}；"
        f"本次待跑 {len(todo)}"
    )
    if missing:
        print(f"缺切片 {len(set(missing))} 张（如 {sorted(set(missing))[:3]}），这些题会被跳过")
    if not todo:
        print("没有待跑的题，直接汇总已有结果")
    if args.dry_run:
        for item, path in todo[:5]:
            print(f"\n--- {item['id']} [{item['task']}] {path.name}")
            print(build_question(item)[:400])
        print(f"\n（dry-run 只展示前 5 条，实际待跑 {len(todo)} 条）" )
        return 0

    from giant.cli.main import Mode, Provider
    from giant.cli.runners import run_single_inference
    from giant.eval.matching import match_option

    started = time.time()
    with results_path.open("a", encoding="utf-8") as handle:
        for index, (item, slide_path) in enumerate(todo, start=1):
            question = build_question(item)
            record = {
                "id": item["id"],
                "subset": item["subset"],
                "slide": item["slide"],
                "task": item["task"],
                "gold": item["answer_letter"],
                "gold_text": item["answer_text"],
            }
            tick = time.time()
            try:
                result = run_single_inference(
                    wsi_path=slide_path,
                    question=question,
                    mode=Mode.giant,
                    provider=Provider(args.provider),
                    model=args.model or "qwen-vl-max",
                    max_steps=args.max_steps,
                    strict_font_check=False,
                    enforce_fixed_iterations=False,
                    runs=args.runs,
                    budget_usd=args.budget_usd,
                    verbose=0,
                    prescreen=args.prescreen,
                    prescreen_k=5,
                    neuro_symbolic=args.neuro_symbolic,
                    patho_r1=args.patho_r1,
                    rag=args.rag,
                    rag_corpus=args.rag_corpus,
                    rag_k=args.rag_k,
                )
                match = match_option(result.answer, item["options"])
                record.update(
                    {
                        "pred": match.letter,
                        "method": match.method,
                        "match_score": match.score,
                        "answer_text": (result.answer or "")[:800],
                        "cost_usd": round(result.total_cost, 6),
                        "success": result.success,
                        "error": result.error_message,
                        **_extract_diagnostics(result),
                    }
                )
            except Exception as error:  # 单题失败不应中断整批
                record.update({"pred": None, "method": "exception", "error": str(error)[:300]})
            record["elapsed_s"] = round(time.time() - tick, 1)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{index}/{len(todo)}] {item['id']} gold={record['gold']} "
                f"pred={record['pred']} ({record['method']}) "
                f"{record['elapsed_s']}s ${record.get('cost_usd', 0):.4f}"
            )

    records = _load_jsonl(results_path)
    summary = _summarize(records, subset=args.subset, tag=args.tag, out_dir=args.out_dir)
    _print_summary(summary)
    print(f"\n总耗时 {(time.time() - started) / 60:.1f} 分钟，结果：{results_path}")
    return 0


def _summarize(records: list[dict], *, subset: str, tag: str, out_dir: Path) -> dict:
    from giant.eval.metrics import group_by, summarize

    overall = summarize(records).as_dict()
    by_task = {task: summary.as_dict() for task, summary in group_by(records, "task").items()}
    citations = [
        record["final_citation"]["coverage"]
        for record in records
        if isinstance(record.get("final_citation"), dict)
    ]
    gated = [
        1
        for record in records
        for gate in (record.get("rag_gate") or [])
        if gate.get("kept", 0) == 0
    ]
    payload = {
        "subset": subset,
        "tag": tag,
        "overall": overall,
        "by_task": by_task,
        "cost_usd_total": round(sum(record.get("cost_usd") or 0 for record in records), 4),
        "elapsed_s_total": round(sum(record.get("elapsed_s") or 0 for record in records), 1),
        "citation_coverage_mean": round(sum(citations) / len(citations), 4) if citations else None,
        "gate_blocked_events": len(gated),
    }

    (out_dir / f"{subset}.{tag}.summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / f"{subset}.{tag}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "task", "slide", "gold", "gold_text", "pred", "method", "answer_text", "cost_usd", "elapsed_s", "error"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)
    return payload


def _print_summary(summary: dict) -> None:
    overall = summary["overall"]
    print("\n=== 总体 ===")
    print(
        f"N={overall['n']}  acc={overall['accuracy']:.4f}  macro-F1={overall['macro_f1']:.4f}  "
        f"未匹配={overall['n_invalid']}  多数类基线={overall['majority_baseline']:.4f}"
    )
    print(f"总成本 ${summary['cost_usd_total']}  总耗时 {summary['elapsed_s_total'] / 60:.1f} 分钟")
    if summary["citation_coverage_mean"] is not None:
        print(
            f"证据引用覆盖率均值={summary['citation_coverage_mean']}  "
            f"闸门拦截事件={summary['gate_blocked_events']}"
        )
    print("\n=== 分任务 ===")
    for task, values in sorted(summary["by_task"].items(), key=lambda kv: -kv[1]["n"]):
        print(
            f"  {task:<40} N={values['n']:<4} acc={values['accuracy']:.3f} "
            f"macroF1={values['macro_f1']:.3f}"
        )


if __name__ == "__main__":
    sys.exit(main())
