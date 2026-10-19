# Pathovoyage + RAG（独立增强版）

本仓库是 [yt-Dong123/Pathovoyage_backend](https://github.com/yt-Dong123/Pathovoyage_backend)
在提交 `e1715df` 之上的**独立增强分支**，原仓库未做任何修改，也不含本仓库的提交历史。

## 相对原版新增了什么

| 模块 | 内容 |
|---|---|
| `src/giant/knowledge/` | MedRAG 教科书语料检索：BM25 倒排索引、中英病理术语表、语料策展与句子级切窗、**检索相关性闸门**、判据卡与编号证据块 |
| `src/giant/eval/` | 评测层：答案 → 选项四档匹配（字母/精确/子串/词重叠）、accuracy / macro-F1 / 多数类基线 / 逐任务分解 |
| `src/giant/agent/` `cli/` | 规划与仲裁环节注入证据；轨迹记录 `rag_evidence` / `rag_gate` / `final_citation`；新增 `--rag` 开关与 `giant knowledge build|status|search` 子命令 |
| `scripts/` | `prepare_slidebench.py`（题目子集）、`fetch_tcga_slides.py`（GDC 切片下载，断点续传+md5）、`eval_slidebench.py`（批量评测，断点续跑+成本统计） |
| `docs/` | `RAG_INTEGRATION.md`（RAG 设计与实现状态）、`BENCHMARK.md`（SlideBench 评测手册） |
| `tests/` | 43 个纯离线测试（不依赖 API Key 与 WSI） |
| `data/corpus/` | 结直肠语料成品 541 片段 + 策展规则 `sources.json` + 人工判据卡（标注待专家复核） |
| `data/slidebench/subsets/` | SlideBench 结直肠子集：TCGA COAD+READ（231 题 / 87 张切片）、HISTAI Colorectum（150 题） |

## 未包含（体积原因）

- **TCGA 原始切片**（46.8 GB）：用 `python scripts/fetch_tcga_slides.py manifest && ... download` 现下；
- **MedRAG 原始教科书 jsonl**（40 MB）：成品语料已随仓库提供，无需原始文件。

## 快速开始

```bash
pip install -e ".[plip]"
python -m pytest tests -q                     # 43 passed
giant knowledge status                        # 语料指纹与片段数
giant knowledge search "肿瘤出芽怎么分级"       # 试检索
```

评测流程见 `docs/BENCHMARK.md`。

## 说明

- 本项目为科研原型，诊断结论不可用于临床。
- 语料来自 MedRAG textbooks（`huggingface.co/MedRAG`），商用需替换为授权来源；
  `data/corpus/curated_colorectal.jsonl` 中的判据卡为人工整理，**待病理专家复核**。
