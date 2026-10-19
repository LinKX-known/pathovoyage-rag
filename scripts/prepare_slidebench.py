"""把 SlideBench 的 VQA CSV 预处理成项目可消费的 JSONL 子集。

背景：SlideBench-VQA 的问答表只包含「题目 + 选项 + 答案字母 + 切片 ID」，
切片像素本身需要另外下载（TCGA 从 GDC、HISTAI 从 huggingface 数据集）。
本脚本只做「题目侧的准备工作」，不涉及图像，因此可以在任何机器上离线跑。

用法：
    python scripts/prepare_slidebench.py --subset tcga-crc
    python scripts/prepare_slidebench.py --subset histai-colorectum
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import json
from pathlib import Path

DATA_DIR = Path("data/slidebench")
OUT_DIR = Path("data/slidebench/subsets")

# TCGA 的结直肠两套：结肠腺癌（COAD）与直肠腺癌（READ）。
_TCGA_CRC_TUMORS = {"COAD", "READ"}


def _norm_slide_tcga(raw: str) -> str:
    return raw.strip()


def _norm_slide_histai(raw: str) -> str:
    """HISTAI 的 Slide 列是 "['HISTAI-breast/case_0783_slide_H&E_0']" 这种字符串。"""
    text = raw.strip()
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, list) and parsed:
            return str(parsed[0]).strip()
    return text.strip("[]'\"")


def build_tcga_crc() -> list[dict]:
    rows = list(
        csv.DictReader((DATA_DIR / "SlideBench-VQA-TCGA.csv").open(encoding="utf-8"))
    )
    items: list[dict] = []
    for row in rows:
        tumor = (row.get("Tumor") or "").strip()
        if tumor not in _TCGA_CRC_TUMORS:
            continue
        options = {
            letter: (row.get(letter) or "").strip()
            for letter in ("A", "B", "C", "D")
            if (row.get(letter) or "").strip()
        }
        items.append(
            {
                "id": f"tcga-crc-{row['ID']}",
                "subset": "tcga-crc",
                "slide": _norm_slide_tcga(row["Slide"]),
                "organ": "colorectum",
                "disease": tumor,
                "task": (row.get("Narrow Category") or "").strip(),
                "broad_task": (row.get("Broad Category") or "").strip(),
                "question": (row.get("Question") or "").strip(),
                "options": options,
                "answer_letter": (row.get("Answer") or "").strip(),
                "answer_text": options.get((row.get("Answer") or "").strip(), ""),
            }
        )
    return items


def build_histai_colorectum() -> list[dict]:
    rows = list(
        csv.DictReader((DATA_DIR / "SlideBench-VQA-Histai.csv").open(encoding="utf-8"))
    )
    items: list[dict] = []
    for index, row in enumerate(rows):
        organ = (row.get("Organ") or "").strip()
        if not organ.lower().startswith("color"):
            continue
        options = {
            letter: (row.get(letter) or "").strip()
            for letter in ("A", "B", "C")
            if (row.get(letter) or "").strip()
        }
        letter = (row.get("Answer") or "").strip()
        items.append(
            {
                "id": f"histai-colorectum-{index}",
                "subset": "histai-colorectum",
                "slide": _norm_slide_histai(row["Slide"]),
                "organ": "colorectum",
                "disease": "",
                "task": options.get(letter, ""),  # HISTAI 的「任务」就是三分类答案本身
                "broad_task": "Disease Classification",
                "question": (row.get("Question") or "").strip(),
                "options": options,
                "answer_letter": letter,
                "answer_text": options.get(letter, ""),
            }
        )
    return items


def summarize(items: list[dict]) -> dict:
    return {
        "n_questions": len(items),
        "n_slides": len({item["slide"] for item in items}),
        "by_task": dict(collections.Counter(item["task"] for item in items)),
        "by_answer": dict(collections.Counter(item["answer_text"] for item in items)),
        "by_disease": dict(collections.Counter(item["disease"] for item in items)),
        "n_options": dict(collections.Counter(str(len(item["options"])) for item in items)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        choices=["tcga-crc", "histai-colorectum", "all"],
        default="all",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    builders = {
        "tcga-crc": build_tcga_crc,
        "histai-colorectum": build_histai_colorectum,
    }
    targets = builders if args.subset == "all" else {args.subset: builders[args.subset]}

    for name, builder in targets.items():
        items = builder()
        out = OUT_DIR / f"{name}.jsonl"
        with out.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        slides = OUT_DIR / f"{name}_slides.txt"
        slides.write_text(
            "\n".join(sorted({item["slide"] for item in items})) + "\n", encoding="utf-8"
        )
        stats = summarize(items)
        (OUT_DIR / f"{name}_stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print(f"\n=== {name} ===")
        print(f"题目: {stats['n_questions']}  切片: {stats['n_slides']}")
        print(f"选项数分布: {stats['n_options']}")
        print("按任务:")
        for task, count in sorted(stats["by_task"].items(), key=lambda kv: -kv[1]):
            print(f"  {task}: {count}")
        print("按答案:")
        for answer, count in sorted(stats["by_answer"].items(), key=lambda kv: -kv[1]):
            print(f"  {answer}: {count}")
        if any(stats["by_disease"].values()):
            print("按癌种:", stats["by_disease"])
        print(f"写出: {out} / {slides} / stats.json")


if __name__ == "__main__":
    main()
