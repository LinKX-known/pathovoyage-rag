# 在 SlideBench 结直肠子集上评测（TCGA COAD + READ）

> 目标：在 AutoDL 单卡 4090 上，用本项目的多尺度导航 + Patho-R1 + RAG 管线，
> 跑出 SlideBench 结直肠子集的准确率，并给出可复现、可审计的评测记录。

---

## 1. 数据事实（2026-10 从官方 CSV / GDC 实测）

| 项 | 值 |
|---|---|
| 题目数 | **231**（COAD 157、READ 74） |
| 切片数 | **87**（全部 `-01Z-` 诊断切片） |
| 题型 | 4 选 1 |
| 切片总体积 | **46.8 GB**（平均 538 MB，中位 356 MB，最大 2.3 GB） |

任务分布（`Narrow Category`）：

| 任务 | 题数 | 我们的管线能否应对 |
|---|---|---|
| Grading | 51 | ✅ 判据在库里（RAG 主战场） |
| Staging | 46 | ⚠️ 单张切片通常判不了 pT/N，需谨慎解读 |
| Disease Classification | 45 | ✅ |
| Tissue Architecture and Arrangement | 29 | ✅ |
| Histopathological Changes | 25 | ✅ |
| Tumor Characteristics | 11 | ✅ |
| Treatment Guidance | 11 | ❌ 纯知识题，结构上不可答，应单列 |
| Differential Diagnosis | 6 | ✅（RAG 鉴别点有用） |
| 其它（Detection/Cyto/Risk/Prognosis） | 7 | ⚠️ 逐个看 |

**关键前提**：SlideBench 的发布物只有「题目 CSV + CONCH 512 维 patch 特征」，
**不含 WSI 像素**。本项目的 agent 靠 openslide 真裁图，所以必须自己下切片：
TCGA 从 GDC（公开），HISTAI 子集需另申请权限。

---

## 2. AutoDL 环境

```bash
# 镜像：PyTorch 2.x / CUDA 12.1+（4090 是 sm_89，不需要 cu128）
# 数据盘：≥ 60 GB（切片 46.8 GB + 模型与索引）
conda create -n pathovoyage python=3.11 -y && conda activate pathovoyage
cd Pathovoyage_backend
pip install -e .                      # 基础依赖（含 openslide-python / openslide-bin）
pip install -e ".[plip]"              # PLIP 需要 torch + transformers
huggingface-cli download vinid/plip   # 预筛权重（代码里 local_files_only=True，不会自动下）
```

起 Patho-R1 的本地 vLLM 服务（7B 约 15.2 GB，4090 24GB 可容）：

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model WenchuanZhang/Patho-R1-7B \
  --served-model-name patho-r1-7b \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --port 18000
```

`.env`：

```ini
DASHSCOPE_API_KEY=sk-...
PATHO_R1_BASE_URL=http://localhost:18000/v1
PATHO_R1_MODEL=patho-r1-7b
RAG_ENABLED=false
```

---

## 3. 三步跑通

```bash
# ① 题目侧准备（已跑过，产物在 data/slidebench/subsets/）
python scripts/prepare_slidebench.py --subset tcga-crc

# ② 切片下载（分批；断点续传；下完校验 md5）
python scripts/fetch_tcga_slides.py manifest                 # 87/87, 46.8 GB
python scripts/fetch_tcga_slides.py download --max-gb 40     # 先拉 40GB 内的
python scripts/fetch_tcga_slides.py download                 # 续下剩余

# ③ 评测：先空跑 → 20 题标定 → 全量两遍（RAG 关/开）
python scripts/eval_slidebench.py --subset tcga-crc --dry-run
python scripts/eval_slidebench.py --subset tcga-crc --limit 20 --max-steps 8 --tag pilot
python scripts/eval_slidebench.py --subset tcga-crc --max-steps 8 --tag rag_off
python scripts/eval_slidebench.py --subset tcga-crc --max-steps 8 --rag --tag rag_on
```

结果（断言 JSONL 可断点续跑）：

```
data/slidebench/results/tcga-crc.<tag>.jsonl      # 逐题记录（含成本/耗时/匹配方式/引用统计）
data/slidebench/results/tcga-crc.<tag>.csv        # 便于 Excel 查看
data/slidebench/results/tcga-crc.<tag>.summary.json
```

---

## 4. 指标口径

- **accuracy**：`pred == gold`，`pred` 由「答案 → 选项」匹配得到；
- **macro-F1**：逐类 F1 平均，**必须与 accuracy 同表**（README 里写清类别不均衡）；
- **majority_baseline**：永远答最常见类的准确率，低于它等于没学到；
- **逐任务分解**：8 类任务分别报，避免知识题掩盖形态题的表现；
- **引用覆盖率**：`final_citation.coverage` 均值，衡量可解释性（RAG 开时才有）；
- **未匹配数**：`n_invalid`，匹配失败的题按错答计，但要单独可见。

答案对齐四档（`src/giant/eval/matching.py`）：字母 → 精确 → 子串（长选项优先）→ 词重叠（Dice ≥ 0.6），
每题的 `method` 与 `match_score` 都落盘，便于抽查判分是否公平。

---

## 5. 成本与时间

| 项 | 估算 |
|---|---|
| 单题 VLM 调用 | `max_steps` 次（每次带 1 张 1000px 图，约 2–3k tokens） |
| 单题 Patho-R1（开 `--patho-r1`） | 每个 crop 一次本地 7B，4090 上约 2–4 s |
| 231 题 × 8 步 | 本地推理约 1.5–2.5 小时；云端 VLM 调用按官方价另计 |
| 建议 | 先 20 题 `--tag pilot` 标定「单题成本 / 单题耗时」，再乘 231 外推 |

**别一上来就全量 + 20 步**：231 × 20 = 4620 次 crop，本地 7B 推理就要 4 小时以上，
云端账单同步放大。以 `--max-steps 8` 做首轮，确认管线正确后再加步数。

---

## 6. 结果怎么读（诚实版）

1. **先看 `n_invalid`**：匹配失败率高说明答案格式不对（比如模型只写结论不给字母），
   这时准确率反映的是「匹配器」而不是「诊断能力」。
2. **再看分任务**：`Treatment Guidance`（11 题）结构上不可答，应剔除或单列；
   `Staging`（46 题）需要 T/N 的镜下依据，单切片常常不足，低分是预期内的。
3. **RAG 的作用点**：主要在 `Grading`（判据卡里有分级标准）与 `Differential Diagnosis`
   （鉴别点），这两组应做配对比较（`rag_off` vs `rag_on`），其余组预期差异不大。
4. **引用覆盖率**：如果 RAG 开着但 coverage 接近 0，说明证据没被用起来——
   这时应回到 `rag_gate` 记录，看是闸门拦掉了还是模型忽略了证据。

---

## 7. 复现记录

每次评测必须记录：`subset`、`tag`、`model`、`max_steps`、`prescreen/patho_r1/rag` 开关、
`rag_corpus_version`（轨迹里的语料指纹）、以及 git commit。这些字段大多已随结果落盘，
git commit 建议写进 `tag`（例如 `rag_on@e1715df`）。
