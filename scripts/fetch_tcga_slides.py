"""按 SlideBench 的切片清单，从 GDC 下载 TCGA-COAD/READ 的诊断切片（.svs）。

为什么需要专门的脚本：SlideBench 只给题目 CSV + CONCH patch 特征，不含 WSI 像素；
而本项目的 agent 靠 openslide 在多尺度上真实裁图，必须有原始切片。

用法：
    # 1) 生成清单（不下载，先看体积；87 张约 46.8 GB）
    python scripts/fetch_tcga_slides.py manifest

    # 2) 分批下载（可断点续传、下完校验 md5）
    python scripts/fetch_tcga_slides.py download --limit 10
    python scripts/fetch_tcga_slides.py download            # 续下剩余

依赖只用标准库，AutoDL 上无需额外装包。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

GDC_API = "https://api.gdc.cancer.gov/files"
GDC_DATA = "https://api.gdc.cancer.gov/data"

SLIDE_LIST = Path("data/slidebench/subsets/tcga-crc_slides.txt")
MANIFEST = Path("data/slidebench/gdc_manifest.csv")
OUT_DIR = Path("data/slidebench/slides")

CHUNK = 4 * 1024 * 1024


def _gdc_query(project: str) -> list[dict]:
    """取一个项目下全部诊断切片（Diagnostic Slide）的文件元数据。"""
    payload = {
        "filters": {
            "op": "and",
            "content": [
                {
                    "op": "=",
                    "content": {"field": "cases.project.project_id", "value": project},
                },
                {
                    "op": "=",
                    "content": {"field": "data_type", "value": "Slide Image"},
                },
                {
                    "op": "=",
                    "content": {
                        "field": "experimental_strategy",
                        "value": "Diagnostic Slide",
                    },
                },
            ],
        },
        "fields": "file_id,file_name,file_size,md5sum",
        "size": "3000",
        "format": "JSON",
    }
    request = urllib.request.Request(
        GDC_API,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)["data"]["hits"]


def cmd_manifest(_args: argparse.Namespace) -> int:
    slides = [line.strip() for line in SLIDE_LIST.read_text().splitlines() if line.strip()]
    wanted = set(slides)
    if not wanted:
        print(f"切片清单为空：{SLIDE_LIST}")
        return 1

    hits: list[dict] = []
    for project in ("TCGA-COAD", "TCGA-READ"):
        project_hits = _gdc_query(project)
        print(f"{project}: 诊断切片 {len(project_hits)} 张")
        hits.extend(project_hits)

    rows: list[dict] = []
    matched: set[str] = set()
    for hit in hits:
        for slide in wanted:
            if hit["file_name"].startswith(slide):
                rows.append(
                    {
                        "slide": slide,
                        "file_id": hit["file_id"],
                        "file_name": hit["file_name"],
                        "size_bytes": hit["file_size"],
                        "md5": hit.get("md5sum") or "",
                    }
                )
                matched.add(slide)
                break

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total = sum(row["size_bytes"] for row in rows)
    missing = sorted(wanted - matched)
    print(f"\n清单: {MANIFEST}")
    print(f"匹配 {len(rows)}/{len(wanted)} 张，合计 {total / 1e9:.1f} GB")
    if rows:
        sizes = sorted(row["size_bytes"] for row in rows)
        print(
            f"单张: 平均 {total / len(rows) / 1e6:.0f} MB，"
            f"中位 {sizes[len(sizes) // 2] / 1e6:.0f} MB，最大 {sizes[-1] / 1e6:.0f} MB"
        )
    if missing:
        print(f"未匹配 {len(missing)} 张：{missing[:5]}")
    return 0


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_one(row: dict, *, out_dir: Path, retries: int = 3) -> tuple[str, str, int]:
    """下载单个切片，支持断点续传；返回 (slide, 状态, 字节数)。"""
    destination = out_dir / f"{row['slide']}.svs"
    part = destination.with_suffix(".svs.part")
    expected = int(row["size_bytes"])

    if destination.exists() and destination.stat().st_size == expected:
        return row["slide"], "已存在", expected

    for attempt in range(1, retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        if offset >= expected and expected > 0:
            break
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = urllib.request.Request(
            f"{GDC_DATA}/{row['file_id']}", headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                mode = "ab" if offset and response.status == 206 else "wb"
                with part.open(mode) as handle:
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        handle.write(block)
            if part.stat().st_size >= expected:
                break
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            print(f"  重试 {attempt}/{retries} {row['slide']}: {error}")
            time.sleep(2 * attempt)
    else:
        return row["slide"], "失败", 0

    if row.get("md5"):
        actual = _md5(part)
        if actual != row["md5"]:
            part.unlink(missing_ok=True)
            return row["slide"], f"md5 不匹配({actual[:8]})", 0

    part.replace(destination)
    return row["slide"], "完成", destination.stat().st_size


def cmd_download(args: argparse.Namespace) -> int:
    if not MANIFEST.exists():
        print(f"清单不存在：{MANIFEST}，先运行 `manifest` 子命令")
        return 1
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    args.out.mkdir(parents=True, exist_ok=True)

    if args.max_gb > 0:
        budget = args.max_gb * 1e9
        kept, used = [], 0
        for row in rows:
            size = int(row["size_bytes"])
            if used + size > budget:
                continue
            kept.append(row)
            used += size
        print(f"按 --max-gb {args.max_gb} 截断：下载 {len(kept)}/{len(rows)} 张，约 {used / 1e9:.1f} GB")
        rows = kept
    if args.limit > 0:
        rows = rows[: args.limit]

    todo = [row for row in rows if not (args.out / f"{row['slide']}.svs").exists()]
    print(f"待下载 {len(todo)} 张（已存在 {len(rows) - len(todo)} 张），并发 {args.jobs}")
    if args.dry_run:
        for row in todo[:10]:
            print(f"  {row['slide']}  {int(row['size_bytes']) / 1e6:.0f} MB")
        if len(todo) > 10:
            print(f"  … 其余 {len(todo) - 10} 张")
        return 0

    done = failed = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {
            pool.submit(_download_one, row, out_dir=args.out, retries=args.retries): row
            for row in todo
        }
        for future in as_completed(futures):
            slide, status, size = future.result()
            symbol = "✓" if status in {"完成", "已存在"} else "✗"
            print(f"  {symbol} {slide}  {status}  {size / 1e6:.0f} MB")
            done += status in {"完成", "已存在"}
            failed += status not in {"完成", "已存在"}

    elapsed = time.time() - started
    print(f"\n完成 {done}，失败 {failed}，耗时 {elapsed / 60:.1f} 分钟")
    if failed:
        print("失败的切片重跑同一命令即可续传（.part 保留）")
    return 0 if failed == 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest", help="生成下载清单（查 GDC 元数据）")
    manifest.set_defaults(func=cmd_manifest)

    download = sub.add_parser("download", help="按清单下载切片")
    download.add_argument("--out", type=Path, default=OUT_DIR)
    download.add_argument("--limit", type=int, default=0, help="本次最多下载几张")
    download.add_argument(
        "--max-gb", type=float, default=0.0, help="按体积上限截断（0 表示不限）"
    )
    download.add_argument("--jobs", type=int, default=4)
    download.add_argument("--retries", type=int, default=3)
    download.add_argument("--dry-run", action="store_true")
    download.set_defaults(func=cmd_download)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
