# Pathovoyage × RAG 接入设计（v0.1）

> 目标：在不破坏现有导航主循环（`agent/runner.py`）与三供应商客户端的前提下，为诊断链路引入
> **可检索、可引用、可审计** 的病理知识，提升判据质量与最终诊断的可辩护性。

---

## 0. TL;DR

1. **可行，但不要把你给的那个仓库当库装进去**——它和本项目在依赖、领域、许可证上都是冲突的。
2. 你给的 `SNOWTEAM2023/MedRAG`（WWW'25）是 **症状/EHR → 疾病** 的知识图谱诱导 RAG 研究脚本；
   与它同名的还有一个 **MedRAG Toolkit**（ACL'24 Findings / MIRAGE），那个才提供
   PubMed / StatPearls / Textbooks / Wikipedia 语料与 4 种检索器。
3. 推荐拆成三层知识：**K-RAG（指南/教科书检索）+ KG-RAG（结直肠病理本体）+ C-RAG（相似病例）**。
4. 第一期只做 **K-RAG 的词法检索 + 2 个接入点**（规划、仲裁），1–2 天可跑通，默认关闭、零回归。

---

## 1. 两个 MedRAG 的辨析

| | `SNOWTEAM2023/MedRAG`（你给的链接） | `Teddy-XiongGZ/MedRAG` Toolkit |
|---|---|---|
| 论文 | WWW'25 *KG-Elicited Reasoning for Healthcare Copilot* | ACL'24 Findings *Benchmarking RAG for Medicine* |
| 输入 | 患者主诉/症状/活动受限（DDXPlus 合成 EHR） | 医学考题、临床问题 |
| 检索对象 | **相似病例**（EHR embedding）+ 疾病-表现 **知识图谱** | **文献片段**（PubMed/StatPearls/Textbooks/Wikipedia） |
| 检索器 | 向量相似 + KG 邻居 | BM25 / Contriever / SPECTER / MedCPT |
| 输出 | 鉴别诊断 + 治疗建议 + **追问问题** | 带 snippet 出处的答案 |
| 代码形态 | 研究脚本（`main.py`、`KG_Retrieve.py`），无包结构 | 可复用的 `MedRAG` 类 |
| 许可 | **CC BY-NC 4.0（禁商用）** | 代码开源；语料各自许可（StatPearls 为 CC BY-NC-ND） |
| 对本项目价值 | **推理范式**：KG 扩展 + 鉴别对照 + 追问 | **语料与检索器** |

结论：**范式借 SNOWTEAM，语料/检索器参考 Toolkit，落地代码自己写。**

---

## 2. 为什么不能直接把 SNOWTEAM 的仓库装进来

| 冲突项 | 它的要求 | 本项目要求 | 后果 |
|---|---|---|---|
| torch | `2.4.1`（默认 cu121 轮子） | `>=2.7`（cu128，RTX 50 系 sm_120 必需） | 装上即降级，PLIP 跑不动 |
| numpy | `1.26.4` | `>=2.2.6,<2.4` | 冲突，pip 解不出可行解 |
| transformers | `4.45.1` | `>=4.46` | 与 PLIP 加载路径可能不兼容 |
| faiss | `faiss-cpu 1.8.0` | 未引入 | 额外重依赖，只为 1 个检索器不值得 |
| 领域 | DDXPlus 51 个初级诊断 + 慢痛 EHR | 结直肠癌 **形态学亚型 / 分级 / 分期** | 它的 KG 节点对我们无意义 |
| 形态 | 脚本 + 全局 `authentication.py` 占位 key + `config.py` | 包化 `src/giant`，配置走 pydantic-settings | 无法 import 复用 |
| 许可 | CC BY-NC 4.0 | 平台化大概率商用 | **法律风险，必须绕开** |

**可行做法**：把它的 4 个「思路」重写成我们的模块（思路不受版权保护，代码与提示词文本重新撰写）。

---

## 3. 可借鉴的四个模式

1. **分层聚合建图**：`Level1 大类 ← Level2 亚类 ← Level3 具体诊断`，每个节点挂「表现/判据」描述。
   映射到病理：`结直肠癌 ← 腺癌/黏液腺癌/印戒细胞癌/髓样癌… ← WHO 具体亚型`，节点挂镜下判据。
2. **KG 诱导推理**：检索命中节点后，沿图扩展到**兄弟节点**，显式生成「必须与之区分的 N 个诊断 + 各自关键特征」。
   这正是病理鉴别诊断的形态（黏液腺癌 vs 黏液湖 vs 印戒细胞）。
3. **鉴别差异增强（diagnostic difference）**：把易混对做成成对条目塞进提示词，比泛泛的「诊断标准」有效。
4. **追问生成（follow-up questions）**：信息不足时先问 2–3 个澄清问题；平台侧可变成「待补充临床信息」清单。

---

## 4. 目标架构

```
                      ┌─────────────── KnowledgeService（懒加载、可注入） ───────────────┐
 User question ──────▶│ K-RAG  指南/教科书片段  BM25 + MedCPT dense → RRF 融合          │
                      │ KG-RAG 结直肠病理本体   节点命中 → 邻居扩展 → 鉴别点             │──┐
 Historical cases ───▶│ C-RAG  相似病例/轨迹    PLIP 或切片 embedding + 结构化过滤        │  │
                      └──────────────────────────────────────────────────────────────────┘  │
                                                                                          ▼
                                                                    EvidenceBundle（带 id/来源/分数）
                                                                                          │
        ┌───────────────────────────────┬─────────────────────────────┬───────────────────┘
        ▼                               ▼                             ▼
  ① 规划阶段                     ② 会诊阶段                    ③ 仲裁阶段
  plan_retrieval /             每个 crop 的 Patho-R1          arbitrate_patho_r1
  plan_observation             「判据卡」（≤150 词）          「指南证据」区块 + 引用校验
        │                               │                             │
        └───────────────────────────────┴─────────────────────────────┘
                                        ▼
                         Trajectory.rag_evidence[] + Turn.evidence_ids[]
                                        ▼
                            最终诊断（可追溯到具体指南条文）
```

### 4.1 五个接入点（文件级）

| # | 位置 | 现状 | 接入后 |
|---|---|---|---|
| ① | `llm/plip_planner.py:270 plan_retrieval` | 纯 LLM 把问题翻成检索文本；维度权重凭经验 | 意图分类后先检索该问题的病理判据，用 evidence 片段约束生成的检索维度（术语更接近病理语义，PLIP 召回更准） |
| ② | `llm/plip_planner.py:297 plan_observation` | `criteria` 由 Qwen 凭空生成，换题即漂 | `criteria` 来自指南 evidence，`ObservationPlan` 新增 `evidence: list[Evidence]`（默认空，向后兼容） |
| ③ | `llm/patho_r1_client.py:20 _CONSULT_PROMPT`（criteria 注入在 :127） | 每个 crop 带一份自造判据 | 改为**判据卡**：3–5 条指南要点 + 2 条鉴别点 + 当前倍率任务；Evidence id 随 `Turn` 落库 |
| ④ | `agent/runner.py:1224 _arbitrate_patho_r1` → `plip_planner.py:318 arbitrate_patho_r1` | 仲裁者只看专家文字 | 增加 `# 指南证据` 区块；要求每条结论标注 `[E1]`；输出后做**引用校验**（无出处的断言降权/剔除） |
| ⑤ | `agent/trajectory.py:66–77` + `cli/main.py:57 run` | 轨迹只有 plip_plan / patho_r1_plan / arbitration | 轨迹新增 `rag_evidence`；CLI 新增 `--rag/--no-rag`、`--rag-k`、`--rag-mode`，子命令 `giant knowledge build|status|search` |

### 4.2 为什么优先 ①③④ 而不是先接 ②

- ③（Patho-R1 判据）是**当前最弱的一环**：专家模型很吃「你让它看什么」，而现在这份「什么」是另一个 LLM 编的。
- ④（仲裁）是最终答案的出口，接证据收益最大且**一次调用**，成本可控。
- ②（每 crop 一次会诊）调用量 = 步数，注入必须极度压缩（≤200 tokens），否则延迟/成本线性上涨。

---

## 5. 数据结构（新增，全部 frozen pydantic）

```python
class Passage(BaseModel, frozen=True):
    id: str                      # "WHO-CRC-2019#3.2.1"
    title: str                   # "WHO Classification of Tumours: Digestive System"
    source: str                  # "who" | "csco" | "nccn" | "textbook" | "case"
    section: str | None = None
    text: str
    score: float = 0.0
    meta: dict[str, str] = {}    # edition / year / url

class EvidenceBundle(BaseModel, frozen=True):
    query: str
    stage: Literal["planning", "consult", "arbitration"]
    passages: list[Passage] = []
    discriminators: list[str] = []   # KG 产出的「必须区分」条目
    corpus_version: str = ""
```

`Turn` 增加 `evidence_ids: list[str] = []`；`Trajectory` 增加 `rag_evidence: list[Passage] = []`。

---

## 6. 模块布局

```
src/giant/knowledge/
  __init__.py      # KnowledgeService 门面（懒加载，runner 通过 AgentConfig 注入）
  schemas.py       # Passage / EvidenceBundle / KGNode
  corpus.py        # build_index(): data/corpus/*.jsonl → 分块 → 索引落盘
  retriever.py     # BM25Retriever | DenseRetriever(MedCPT) | HybridRRF
  kg.py            # ColorectalKG: data/kg/colorectal.yaml → 三元组 + 邻居扩展
  cards.py         # criteria_card(bundle, bucket) -> str（压缩到 token 预算内）
  service.py       # retrieve_for_planning / _consult / _arbitration（带 diskcache）
data/corpus/*.jsonl
data/kg/colorectal.yaml
```

`pyproject.toml` 增加可选 extra（沿用现有 `plip` extra 的做法）：

```toml
[project.optional-dependencies]
rag = ["rank_bm25>=0.2.2", "jieba>=0.42.1"]
```

dense 检索复用已有的 `torch`/`transformers`（`--extra plip` 已带），不引入 faiss：语料只有几千条时，
numpy 暴力内积足够（`np.argsort(-X @ q)`），避免新重依赖。

---

## 7. 检索策略与上下文预算

| 阶段 | 检索 | 注入量 | 备注 |
|---|---|---|---|
| 规划 | top-6 → 压成 5 条要点 | ≤1.5k tokens | 只跑一次 |
| 会诊（每 crop） | 不检索，复用规划阶段的**判据卡**，按倍率切段 | ≤200 tokens | 每 run 只算一次，随 crop 复用 |
| 仲裁 | top-8（问题+专家结论关键词） | ≤3k tokens | 只跑一次 |

- 融合：RRF（`score = Σ 1/(60+rank)`），BM25 与 dense 各取 top-20。
- 中文：BM25 需分词，P0 用 `jieba`；也可先用「字符 bigram」避免新依赖。
- 缓存：已有 `diskcache` 依赖，key = `(corpus_version, mode, stage, query_hash, k)`。
- 成本：本地检索 ≪ 一次 VLM 调用；真正增量是注入的 token（估算 +8%~15%/run，取决于 top-k）。

---

## 8. 分阶段实施

**P0（1–2 天，可立即验证价值）**
- 手工策展 30–80 条结直肠病理语料（WHO 亚型判据、CSCO/NCCN 要点、常见鉴别点），英文+中文对照。
- BM25 + jieba，`KnowledgeService.retrieve()` 打通。
- 接 ①③④：规划阶段注入、仲裁注入 + 引用校验；`--rag/--no-rag` 开关。
- 轨迹落 `rag_evidence`；`giant knowledge build|status`。

**P1（约 1 周）**
- MedCPT dense + RRF 融合；判据卡压缩逻辑；KG YAML（亚型层级 + 鉴别对）与邻居扩展。
- 接 ②（每 crop 判据卡）；引用校验指标化。

**P2（平台化）**
- C-RAG：历史切片/轨迹建索引（PLIP 图向量或轨迹摘要向量），相似病例 few-shot + 一致性校验。
- 追问节点（借 SNOWTEAM 的 follow-up 范式）：问题信息不足时输出「待补充临床信息」。
- 与 `refactor/langgraph-orchestration-v1` 对齐：RAG 作为一个独立节点注入，不改 State 契约。

---

## 9. 评测与回归

1. **配对对照**：同一批切片跑 `--rag` / `--no-rag`，比较 answer、agreement、cost、latency。
2. **引用忠实度**：仲裁输出里每条断言能否映射到 `[E#]`；无出处比例作为核心指标（目标 <10%）。
3. **检索质量**：人工标注 20 题 × top-6 命中率（P@3、MRR）。
4. **回归**：默认 `RAG_ENABLED=false`，`ObservationPlan`/`Turn` 新字段带默认值，确保 langgraph 分支的 parity 测试不受影响。

---

## 10. 合规与许可

| 资产 | 许可 | 结论 |
|---|---|---|
| SNOWTEAM2023/MedRAG 代码 | CC BY-NC 4.0（**禁商用**） | 只借思路，代码与提示词全部重写 |
| MedRAG Toolkit 语料 | 代码开源；Textbooks 源自 MedQA（原书版权）、StatPearls 为 CC BY-NC-ND、Wikipedia CC BY-SA、PubMed 摘要受 NLM 条款 | 研究可用；商用必须替换为授权语料 |
| 推荐自建语料 | WHO 消化系统肿瘤分类（授权副本）、CSCO/NCCN 指南（授权）、院内脱敏病理报告 | 商用唯一干净路径 |
| 患者数据（C-RAG） | — | 切片与报告需脱敏、访问控制、审计日志 |

---

## 11. 风险清单

| 风险 | 影响 | 缓解 |
|---|---|---|
| 检索噪声污染仲裁 | 引入错误结论 | `RAG_MIN_SCORE` 阈值 + 引用校验 + 默认关闭 |
| Patho-R1 上下文被撑爆（本地 7B，`max_tokens=256`） | 输出退化 | 判据卡硬性 ≤200 tokens，超长截断 |
| 语料版本漂移 | 结论不可复现 | `corpus_version` 写入轨迹 |
| 与 langgraph 分支冲突 | 合并困难 | RAG 做成注入依赖，不改 State 契约 |
| 商用许可 | 法务阻断 | 语料走自建/授权路线（见 §10） |

---

## 12. P0 实现状态（已完成）

### 12.1 新增文件

| 文件 | 作用 |
|---|---|
| `src/giant/knowledge/schemas.py` | `Passage` / `EvidenceBundle` / `KnowledgeStats`（frozen pydantic） |
| `src/giant/knowledge/tokenize.py` | 轻量分词：英文小写+去停用词+极简词干；中文按字符二元组 |
| `src/giant/knowledge/lexicon.py` | 中→英病理术语表（约 120 条），中文问题 → 英文检索词，离线可复现 |
| `src/giant/knowledge/bm25.py` | 自包含 Okapi BM25 倒排索引（无第三方依赖） |
| `src/giant/knowledge/corpus.py` | 去噪（词粘连检测）、句子级滑窗切分、语料指纹 |
| `src/giant/knowledge/selection.py` | 按 `id_range` / `regex` / 全量的语源选择规则 |
| `src/giant/knowledge/cards.py` | 判据卡（≤900 字符）与编号证据块（`[E1]…`） |
| `src/giant/knowledge/service.py` | `KnowledgeService` 门面：懒加载、检索、相对分数截断 |
| `data/corpus/sources.json` | 语料策展规则（可审计、可复现） |
| `data/corpus/curated_colorectal.jsonl` | 14 条人工判据卡（亚型/分级/分期/切缘/LVI/PNI/出芽/MMR），**待病理专家复核** |
| `data/corpus/textbooks_colorectal.jsonl` | 构建产物（541 片段 / 247k 字符 / 指纹 `6b21d2539d8e`） |
| `tests/test_knowledge.py` | 20 个离线测试（含提示词注入断言） |

### 12.2 接入点（修改的文件）

| 位置 | 改动 |
|---|---|
| `config.py` | 新增 `RAG_*` 配置（默认关闭） |
| `cli/main.py` | 新增 `giant knowledge build/status/search`；`run` 新增 `--rag/--no-rag`、`--rag-corpus`、`--rag-k` |
| `cli/runners.py` | 按开关构造 `KnowledgeService`，注入 `AgentConfig`；预筛前先检索术语 |
| `agent/runner.py` | `AgentConfig.knowledge`；`_retrieve_evidence()`；规划阶段检索并写入轨迹；仲裁前联合检索；每个 crop 的 `Turn.evidence_ids` |
| `llm/plip_planner.py` | `plan_retrieval/plan_observation/arbitrate_patho_r1` 接受 `evidence`，把证据注入提示词 |
| `agent/trajectory.py` | `Trajectory.rag_evidence`、`rag_corpus_version`；`Turn.evidence_ids` |

### 12.3 用法

```bash
# 1) 下载语料（已下载 3 个源文件到 data/rag_raw/）
#    来自 huggingface.co/MedRAG 的 textbooks 子集
# 2) 构建
giant knowledge build
# 3) 试检索（中文可用）
giant knowledge search "肿瘤出芽怎么分级" --k 3
# 4) 带知识检索跑诊断
giant run slide.svs -q "组织学亚型是什么？" -p qwen --patho-r1 --rag
```

### 12.4 检索实测

| 中文问题 | Top-1 命中 | 分数 |
|---|---|---|
| 切缘是否干净（R0） | Curated：R0/R1 定义 + 直肠 CRM ≤1 mm | 15.4 |
| 有没有脉管瘤栓 | Curated：LVI 定义与预后意义 | 11.8 |
| 肿瘤出芽怎么分级 | Curated：Bd1/Bd2/Bd3 热点计数 | 23.8 |
| 黏液腺癌的组织学亚型是什么 | Curated：WHO 第 5 版亚型清单 | 40.6 |
| T 分期看浸润深度吗 | Schwartz：T 分期是淋巴结转移最强预测因子 | 15.2 |

### 12.5 语料踩坑记录（重要）

1. **Robbins 的消化道章节在 PDF 抽取中损坏**：约 16% 片段词边界丢失（`Rupturedovarianmucinoustumors…`），
   BM25 无法命中。这类片段按「空格占比 < 8% 或词长中位数 > 16」判据丢弃，全书仅保留强特征句。
2. **关键词全局过滤不可靠**：`colon` 会命中「直肠活检查淀粉样变」。改用**章节位置定位**——
   结直肠词频在文档顺序上的连续密集段（Schwartz 结直肠章节的密度是其他区块的 60 倍）。
3. **中文段落天然无空格**，不能套用噪声判据；中文按字符二元组建索引，避免引入分词器。
4. **中英混排查询的覆盖率门槛会误杀**：改为「相对最高分 25%」截断，语言无关。
5. MedRAG 的 StatPearls / PubMed / Wikipedia 语料仅有元数据（9.9 万篇、23.9M 片段量级），
   本地不可行；P0 只用 textbooks。

### 12.6 已知限制 / P1 待办

- 只做 **BM25 词法检索**；dense（MedCPT）+ RRF 融合未实现。
- **② 每 crop 判据卡未接**：判据卡已能生成，但尚未逐 crop 注入 Patho-R1（预算与延迟需实测）。
- 引用校验只有提示词约束，尚无「无出处断言」的自动剔除与指标统计。
- 语料以 Schwartz 外科章节为主，**组织学描述偏薄**；P1 应补 WHO 分类原文或授权指南。
- 14 条 curated 判据需病理医生复核后方可作为临床依据。
- 端到端（含真实 LLM 调用）未验证：本机无 DashScope Key、无 Patho-R1 服务、未装 torch/PLIP。

### 12.7 验证记录

```
$ python -m pytest tests/test_knowledge.py -q
20 passed in 2.98s

$ python -m compileall -q src/giant          # exit 0
$ python -m giant.cli.main knowledge status
片段: 541 / 字符: 247348 / 指纹: 6b21d2539d8e
语源: Curated_CRC_Criteria, Pathology_Robbins, Pathoma_Husain, Surgery_Schwartz
```

验证用环境：`D:\Pathovoyage_backend\.venv`（Python 3.12.7，装 pydantic/pytest/typer/pydantic-settings/
structlog/rich/aiolimiter/tenacity/openai/anthropic/httpx/pillow/numpy/opencv-python-headless/scipy/
diskcache/scikit-image/openslide-python/openslide-bin），运行需 `PYTHONPATH=src`。

---

## 13. P0.5：检索闸门 + 结构化仲裁（架构与语料均未改动）

目标是在**不动主循环结构、不换语料**的前提下，提升两件事：准确率（不注入错知识）与
可解释性（结论可统计、可追溯）。

### 13.1 新增/改动

| 文件 | 改动 |
|---|---|
| `knowledge/gate.py`（新） | 确定性检索闸门：问题锚点覆盖 + **语料域范围**判定；不调用 LLM，零成本、可复现 |
| `knowledge/schemas.py` | `CitationReport` / `WeightedEvidence` / `ArbitrationResult` |
| `knowledge/cards.py` | `citation_check()`：统计 `[E#]` 的 used / unused / invalid / coverage |
| `llm/plip_planner.py` | `arbitrate_structured()`：要求 JSON 输出并解析；失败退化为纯文本；`arbitrate_patho_r1` 保留为兼容包装 |
| `agent/runner.py` | `_retrieve_evidence()` 检索后过闸门并记录判定；仲裁改用结构化结果；轨迹新增 `rag_gate` / `final_citation` |
| `agent/trajectory.py` | 新增 `rag_gate: list[dict]`、`final_citation: dict | None` |

### 13.2 闸门的实测行为（真实语料，541 片段）

| 问题 | 判定 | 拦截原因 | 注入条数 |
|---|---|---|---|
| 组织学亚型是什么？ | relevant | 锚点覆盖 5 | 3/6 |
| 肿瘤出芽怎么分级 | relevant | 锚点覆盖 6 | 5/6 |
| 切缘是否干净（R0） | relevant | 锚点覆盖 3 | 6/6 |
| 结直肠癌肝转移的切缘要求 | relevant | 含结直肠，放行 | 6/6 |
| 肺腺癌的 EGFR 突变率是多少？ | **irrelevant** | `out_of_scope_organ=肺` | 0 |
| 乳腺癌的分子分型有哪些？ | **irrelevant** | `out_of_scope_organ=乳腺` | 0 |
| 胃癌的 Lauren 分型 | **irrelevant** | `out_of_scope_organ=胃` | 0 |

关键教训：**只靠词面覆盖挡不住跨器官噪声**——「肺腺癌的组织学亚型」和
「结直肠腺癌的组织学亚型」共享 adenocarcinoma / histology / subtype 三个锚点。
必须叠加语料域范围判定（语料是结直肠专用的，这一点写在 `sources.json` 里）。

### 13.3 结构化仲裁的字段

```json
{
  "answer": "浸润性黏液腺癌",
  "weighted_evidence": [{"region": "(70000,65000,12000,12000)", "opinion": "黏液湖>50%",
                         "weight": 0.8, "why": "低倍+高倍共 3 个区域一致"}],
  "conflicts": ["区域 A 判高分化，区域 B 判黏液腺癌（肿瘤异质性）"],
  "minority": ["区域 A 的高分化结论"],
  "citations": ["E1", "E2"]
}
```

解析失败时自动退化为「原文即答案」（`structured=false`），不阻断流程；
`citations` 只接受 `E\\d+` 形式，非法编号被丢弃并计入 `invalid`。

### 13.4 引用统计（可解释性的量化口径）

```
answer: 浸润性黏液腺癌（依据 [E1] 的亚型清单与 [E2] 的分级标准）
→ used=[E1,E2]  unused=[E3,E4,E5,E6]  invalid=[]  coverage=0.333
```

`coverage` 与 `invalid` 是两个可直接进评测表的指标：前者衡量「证据被用起来没有」，
后者是**幻觉编号**信号（引用了不存在的 E9）。

### 13.5 尚未做（需要真实 API 才能验证）

- SED 第 1 步（逐条独立评估 agree/uncertain/disagree）尚未接入；
- 结构化 JSON 输出的**实际成功率**未知——离线只有解析器的单元测试；
- 证据充分性守卫（answer 前检查区域/倍数）仍是提示词约束，未变成代码；
- `--runs` 的多 rollout 仍只用于字符串投票，未做策略蒸馏。


