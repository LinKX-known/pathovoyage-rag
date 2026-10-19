<p align="center">
  <strong>🛰️ Pathovoyage · 病理智航</strong><br>
  <em>多尺度动态探索病理诊断智能体 —— 结直肠癌全切片图像（WSI）诊断</em>
</p>

<p align="center">
  <a href="#"><img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11+-blue.svg"></a>
  <a href="#"><img alt="LLM: Qwen-VL" src="https://img.shields.io/badge/LLM-Qwen--VL-615ced.svg"></a>
  <a href="#"><img alt="Specialist: Patho-R1" src="https://img.shields.io/badge/Specialist-Patho--R1--7B-1a9d6a.svg"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
</p>

---

## 这是什么

Pathovoyage（病理智航）是一个**多尺度动态探索的病理诊断智能体**：让视觉语言模型（VLM）像病理医生一样，在**十亿像素量级**的全切片图像（Whole Slide Image, WSI）上逐层放大、观察、推理，最终给出有依据、可解释的诊断。

它面向**结直肠癌**病理诊断，由四个阶段串联而成——**一个通用导航模型 + 一个专科专家模型 + 一次独立的文字仲裁**：

| 阶段 | 角色 | 说明 |
|---|---|---|
| **① PLIP 问题驱动预定位** | 病理版 CLIP（本地） | 先把临床问题规划成检索意图，再零样本打分筛出可疑候选区，解决 VLM 在全图上「框不准」的定位难题 |
| **② Qwen-VL 多尺度导航** | 通用 VLM（云端） | 在缩略图与 crop 之间反复平移、逐级放大，决定「下一步看哪里」 |
| **③ Patho-R1 专科会诊** | 病理专家 VLM（本地 vLLM） | 每一次 crop 都送病理专科模型独立判断，给出「这块到底是什么」的镜下结论 |
| **④ 千问纯文字仲裁** | 通用 VLM（云端） | 导航结束后，把各区域专家结论汇总成纯文字，单独问一次千问给出最终亚型 |

关键设计：**「看什么」和「下什么诊断」分离**。通用 VLM 负责空间导航（它擅长），病理专家模型负责镜下定性（它擅长），最后仲裁者**只看专家结论、看不见图像**（避免通用 VLM 自己的图像判断盖过专家共识）。

---

## 四阶段诊断流水线

```text
                        一张 WSI + 一个临床问题
                                 │
   ┌─────────────────────────────▼─────────────────────────────┐
   │ ① PLIP 预定位                                              │
   │   千问判意图 A/B/C → 翻译成检索文本 / 拆成 3-5 个加权维度    │
   │   → 滑窗 PLIP 打分（维度×权重求和）→ 贪心 NMS → top-K 候选框 │
   └─────────────────────────────┬─────────────────────────────┘
                                 │ 候选框（画在缩略图上 + 写进提示词）
   ┌─────────────────────────────▼─────────────────────────────┐
   │ ② Qwen-VL 多尺度导航                                       │
   │   select → crop → crop → … → answer                        │
   │   每一步由模型在 Level-0 坐标下自己决定裁剪框                │
   └─────────────────────────────┬─────────────────────────────┘
                                 │ 每一次 crop 的图
   ┌─────────────────────────────▼─────────────────────────────┐
   │ ③ Patho-R1 专科会诊（逐 crop）                              │
   │   按倍率档（低/中/高）选对应观察任务 → 本地 vLLM 给结论       │
   │   → 翻成中文 → 回灌给导航模型，也写进轨迹                    │
   └─────────────────────────────┬─────────────────────────────┘
                                 │ 导航结束，汇总全部专家结论（纯文字）
   ┌─────────────────────────────▼─────────────────────────────┐
   │ ④ 千问纯文字仲裁                                            │
   │   只看专家结论 → 给出一个统一亚型（覆盖导航模型的原答案）     │
   └─────────────────────────────┬─────────────────────────────┘
                                 │
                            最终诊断 + 可追溯轨迹
```

第 ③④ 阶段由 `--patho-r1` 开启；第 ④ 阶段是整条流水线的收口——这一设计针对的是一个具体的失败模式：**通用 VLM 在导航中一边看图一边总结时，它自己的图像观察经常盖过多数专家的结论**（在多张切片上实测过）。把仲裁者的眼睛蒙上，只给它文字，它就只能依据专家共识作答。

---

## PLIP 问题驱动预定位

纯 VLM 在十亿像素的全图上自己决定裁剪坐标时，普遍存在「框不准、漏病灶」的问题——这是通用 VLM 空间推理的硬边界。Pathovoyage 把「定位」交给确定性的病理感知模型 **PLIP**（病理版 CLIP），并且**让检索文本跟着用户问题走**，而不是永远只找「肿瘤」。

### 意图分类与分派

用户问题先判三类意图，再分派到两条不同的路径（[plip_planner.py](src/giant/llm/plip_planner.py)）：

| 意图 | 典型问题 | 处理方式 |
|---|---|---|
| **A 存在性检测** | 有没有脉管瘤栓？切缘干净吗？ | 翻译成**一句**20词以内英文视觉描述检索文本 |
| **B 分类/定性** | 组织学亚型是什么？分化程度如何？ | 拆成 **3-5 个加权诊断维度**，每维一句检索文本 |
| **C 程度/量化** | T 分期是多少？浸润多深？ | 与 A 共用翻译路径 |

B 类的权重由千问按临床重要性分配并归一化到 1.0（结直肠亚型题里「黏液特征」拿到最高权重）。提示词里带一张**临床关键词 → 视觉描述**的翻译对照表（切缘、脉管瘤栓、神经侵犯、黏液湖、坏死、肿瘤出芽、浸润深度、浆膜侵犯），另有一句保底检索文本兜底。



### 打分与召回

```text
缩略图 → 组织分割(TissueSegmentor) → 滑窗过滤（组织占比不足的窗口丢弃）
    → 一次批量前向，得到 (patch × 检索维度) 相似度矩阵
    → 每个 patch 的分数 = Σ(维度权重 × 该维度相似度)
    → 标准贪心 NMS（半径 = 1.5 × 窗口尺寸）→ top-K 候选框（Level-0 坐标）
```

注意是**一次批量前向**拿到整张相似度矩阵，而不是每个维度各跑一次 PLIP——所有维度的检索文本一次性送进模型。权重的作用是**加权求和成一个综合分数**，随后用这个分数做一次标准贪心 NMS 得到空间上不重叠的 top-K，而不是按维度分别召回再合并。

未提供维度时（规划失败的回退路径），打分退回固定的「肿瘤 − 正常」相似度差。

候选框以两种形式喂回给 VLM：**画到缩略图上**（视觉种子）+ **写进提示词**（文本种子）。候选框只负责「指个方向」，精细定位仍由 VLM 自己裁剪完成——这是最轻的定位接入方式，既解决了定位难题，又保留了智能体的自主探索能力。

---

## 多尺度动态导航

全切片图像有数十亿像素，远超 VLM 的视野。传统做法是看模糊缩略图或随机采样，会严重低估 VLM 的能力。Pathovoyage 的核心是让 VLM **主动导航**——像病理医生在镜下反复扫视、对比、确认：

```text
1. SELECT    →  PLIP 预筛，确定可疑候选区域
2. CROP      →  低倍下看组织结构（腺体、生长模式、浸润方式）
3. CROP      →  高倍下看细胞学特征（核异型、分裂象、染色质）
4. REASON    →  累积多区域证据，更新鉴别诊断
5. ANSWER    →  证据充分后给出最终诊断
```

### 三种工具（动作空间）

导航的每一步，VLM 通过判别联合类型（Pydantic）决定一个动作：

| 动作 | 含义 | 说明 |
|---|---|---|
| `select` | PLIP 预筛 | 开启 `--neuro-symbolic` + 预筛时第一步强制，不占步数 |
| `crop` | 放大观察 | 在 Level-0 全分辨率坐标下裁剪某区域 |
| `answer` | 给出诊断 | 给出最终诊断文本 |

> 另保留 `conch`（图像-文本检索打分）作为消融功能，默认关闭。

### 导航的关键工程细节

- **坐标系统**：统一使用 Level-0（切片原生全分辨率）像素空间，缩略图上的坐标轴刻度与候选框都映射回 Level-0，保证定位可复现。
- **步数预算**：`current_step` 只算 crop + answer（select/conch 不占步），`-T 20` ≈ 19 次 crop + 1 次 answer。
- **倍率分档**：按 crop 的 Level-0 长边自动分档，并据此选用 Patho-R1 的观察任务：

  | 长边尺寸 | 档位 | 对应物镜 |
  |---|---|---|
  | ≥ 8000 px | `low` | 4×/10× — 组织结构、生长模式、浸润方式 |
  | ≥ 4000 px | `medium` | 10× — 腺体排列、间质反应、浸润前沿 |
  | < 4000 px | `high` | 20×/40× — 核异型、核分裂、胞质细节 |

- **空白 crop 拦截**：裁剪框落到玻璃/白色背景上时会污染专家会诊与最终答案。每次 crop 后用 **HSV 饱和度**判据检测（组织 crop 的「饱和像素占比」实测 0.36–0.75，空白 crop 仅 0.0000–0.0001，阈值取 0.02），判为空白则不记录这一步，而是回灌一条纠错提示——**告知组织的外接框 + PLIP 候选中心点**，让模型下一轮裁到组织上。
- **冗余抑制**：用 IoU + 累计重叠拦截「重复 crop」和「逐步爬行」，避免死循环、省预算。
- **图像异常容错**：空白检测在图片解码失败时**保守放行**（宁可放过，不因图像异常把一次正常 crop 拦成无效、白跑几轮重试）。

---

## Patho-R1 专科会诊

通用 VLM 能把「看哪里」决策得很好，但在「这块到底是什么」的镜下定性上不如专科模型。Pathovoyage 因此在**每一次 crop** 上调一次病理专科模型 **Patho-R1-7B**（WenchuanZhang/Patho-R1-7B，基于 Qwen2.5-VL，本地 vLLM 以 OpenAI 兼容 API 部署）。

- **只吃已裁好的图**：Patho-R1 视觉定位能力弱，因此**不做定位**——只对导航模型已经裁好的 crop 做深度专科推理。两者的长短板正好互补。
- **按问题动态生成观察任务**：启动时先让千问按用户问题生成一份「诊断判据 + 低/中/高倍观察任务」（全英文，Patho-R1 偏英文），再按每一步的倍率档选择对应的观察任务。判据跟着问题走，换题不失效。
- **器官上下文纠偏**：从问题里匹配已知器官名（结直肠、胃、肺、乳腺……共 23 个），前置到提示词里，严禁跨器官误判。
- **温度 0.3**：Patho-R1-7B 在贪心解码下极易陷入重复崩溃（输出「不不不不」「细胞/细胞」这类死循环），0.3 的采样能打破循环。
- **中文回灌**：Patho-R1 偏英文输出，结果先由千问翻成简体中文再回灌给导航模型，也写进轨迹供可视化。

会诊结论会作为一条消息回灌给导航模型（`[Patho-R1 专家结论] …`），因此开启 `--neuro-symbolic` 时导航模型的推理只需写「倍率 + 规划」两段，镜下所见与鉴别诊断交给专家，避免两边重复劳动。

> **前置条件**：需要本地起 Patho-R1 的 vLLM 服务，并在 `.env` 里配置 `PATHO_R1_BASE_URL`（默认 `http://localhost:18000/v1`，通常通过 SSH 隧道转发）。服务不可达时会诊会失败并记 `[Patho-R1 会诊失败，未获得专家判断]`——该条会被仲裁环节剔除，不会污染最终结论。

---

## 独立仲裁（方案 B）

导航结束后有一个**独立的收口步骤**：把轨迹里所有 crop 的专家结论汇总成**纯文字**，单独调一次千问给出统一诊断，并**覆盖**导航模型的原答案。

```text
# 汇总格式（每条一行，只传文字，不传任何图）
- 区域 (70000, 65000, 12000, 12000)·low：浸润性黏液腺癌，黏液湖 >50%。
- 区域 (32000, 46000, 3000, 3000)·high：高分化腺癌，未见黏液湖。
```

仲裁提示词里写了几条铁律：

- **你看不到任何图像**，只能看到文字结论——禁止加入自己的图像观察、记忆或推测，写的每一条判断都必须能在专家结论里找到出处；
- 专家结论矛盾通常源于**肿瘤异质性**或不同区域取到了不同成分，必须权衡后给出**一个**统一结论，而不是罗列几种说法让人自己选；
- 若专家结论明显分成两派，**以多数为准**，并指出少数派结论出现在哪些区域；
- 只有当专家结论确实不足以回答该问题时，才回答「专家证据不足」，并说明还缺什么证据。


---

## 结构化诊断（`--neuro-symbolic`）

开启后，系统提示词会追加一段**神经符号纪律**，把导航行为约束到病理医生的读片习惯上：

1. **首步强制 select**：必须先用 PLIP 工具预筛候选区，之后才允许 crop/answer。候选区列表同时给在用户消息里作参考。
2. **推理四段结构**：`倍率` + `规划`（镜下所见与鉴别诊断由 Patho-R1 提供，不重复写）。
3. **术语规范**：使用规范中文病理术语（architectural → 组织结构，NOT 建筑学；desmoplasia → 促结缔组织增生；differential diagnosis → 鉴别诊断……）。
4. **导航纪律**：评估肿瘤范围用**一个**中倍 crop 看全，而不是一连串小视野高倍「爬行」；不在相似倍率下重复裁同一区域。
5. **覆盖要求**：回答前必须检查**至少 3 个不同的候选区**，每个都要有低倍概览 + 高倍特写——只查一两个区域不足以做出可辩护的诊断。
6. **裁剪框自主决定**：候选中心是 PLIP 给出的**近似位置提示**，不是精确裁剪目标；必须看着实际组织自己定框，居中并留 20-30% 的正常组织边缘以读取肿瘤-正常交界。

> 说明：这一层目前是**提示词层面**的行为约束（[templates.py](src/giant/prompts/templates.py) 的 `SYSTEM_PROMPT_NEURO_SYMBOLIC`），不是代码里的规则引擎。证据充分性的把关依靠「强制首步 select + 覆盖要求 + answer 前的专家会诊」。

---

## 系统架构

```text
                              ┌──────────────────────────────────────────────┐
                              │          Pathovoyage 诊断智能体               │
                              └──────────────────────────────────────────────┘
                                            │
     ┌──────────────────┬───────────────────┼───────────────────┬──────────────────┐
     ▼                  ▼                   ▼                   ▼                  ▼
┌───────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌─────────────┐
│ 预定位层   │   │  导航层      │   │  专科会诊层   │   │  仲裁层      │   │  视觉/几何   │
│           │   │              │   │              │   │              │   │             │
│ 意图 A/B/C │   │ ReAct 循环   │   │ Patho-R1-7B  │   │ 千问纯文字   │   │ PLIP 打分    │
│ 翻译/拆维  │──▶│ crop/answer  │──▶│ 本地 vLLM    │──▶│ 只看专家结论 │   │ NMS / 裁剪   │
│ 维度加权   │   │ 空白/冗余拦截 │   │ 中文化       │   │ 覆盖原答案   │   │ 坐标刻度     │
└───────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └─────────────┘
     │                  │                   │                   │                  │
     └──────────────────┴───────────────────┴───────────────────┴──────────────────┘
                                            │
                                            ▼
                          ┌──────────────────────────────────┐
                          │  Qwen-VL-Max (DashScope 云端)      │
                          │  + Patho-R1-7B (本地 vLLM)         │
                          └──────────────────────────────────┘
```

**设计原则**：**定位、观察、定性、裁决四件事由不同的模型分工**。通用 VLM 负责空间决策与语言组织，PLIP 提供确定性的定位锚点，Patho-R1 提供专科镜下定性，最后的仲裁者被剥夺视觉、只能依据专家文字结论——每一层的长短板互补，且每一步都留下可追溯的记录。

---

## 快速开始

### 1. 环境

- Python ≥ 3.11，包管理器 [uv](https://docs.astral.sh/uv/)
- 可访问 DashScope（阿里云百炼）+ HuggingFace
- （可选，`--patho-r1`）可访问本地部署的 Patho-R1-7B vLLM 服务

### 2. 安装

```bash
git clone <仓库地址> pathovoyage && cd pathovoyage

# 安装依赖（--extra plip 会额外装 torch + transformers，供 --prescreen 使用）
uv sync --extra plip

# 下载 PLIP 模型权重（仅 --prescreen 需要，约 600MB）
# 国内网络建议先切镜像：export HF_ENDPOINT=https://hf-mirror.com
uv run huggingface-cli download vinid/plip
```

> **显卡提示**：RTX 50 系（Blackwell, sm_120）需要 torch ≥ 2.7（CUDA 12.8），`pyproject.toml` 已把 torch 指向 CUDA 12.8 索引。不匹配时会回退到 CPU 运行——PLIP 是 ViT-B，小模型，CPU 也能跑，只是慢一些。

### 3. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入 DashScope Key（从 https://bailian.console.aliyun.com/ 获取）
# DASHSCOPE_API_KEY=sk-xxxxxxxx
```

若要用 `--patho-r1`，再补上本地服务地址（`config.py` 里已有默认值，通常无需修改）：

```bash
PATHO_R1_BASE_URL=http://localhost:18000/v1
PATHO_R1_MODEL=patho-r1-7b
```

### 4. 跑一张切片

推荐配置——**预筛 + 结构化诊断 + 专科会诊 → 仲裁**：

```bash
uv run giant run "WSI_DATA" \
  -q "QUESTION" \
  -p qwen -T 20 -o result.json \
  --prescreen --neuro-symbolic --patho-r1
```

跑完终端会打印：

```
Answer: <最终诊断>
Cost: $0.38
Turns: 11
```

> 开启 `--patho-r1` 会给每个 crop 增加一次本地 vLLM 调用，并在结尾增加一次千问仲裁调用，`Cost` 与总耗时都会相应上涨。

---

## CLI 参考

命令入口：`uv run giant <子命令>`（或安装后直接 `giant <子命令>`）。

| 命令 | 用途 |
|---|---|
| `giant run <wsi> -q <问题>` | 对一张 WSI 运行导航诊断 |
| `giant version` | 打印版本 |

### `run` 参数

| 参数 | 含义 | 默认 |
|---|---|---|
| `wsi_path` | WSI 文件绝对路径（`.svs`/`.ndpi`/`.tiff`） | 必填 |
| `-q / --question` | 诊断问题 | 必填 |
| `-m / --mode` | 单张切片诊断模式：`giant` | `giant` |
| `-p / --provider` | 模型供应商：`qwen` / `openai` / `anthropic` | `openai` |
| `--model` | 模型名 | 按供应商默认（qwen → `qwen-vl-max`） |
| `-T / --max-steps` | 最大导航步数 | 20 |
| `-o / --output` | 保存轨迹 JSON 路径 | 不存 |
| `--prescreen / --no-prescreen` | 开 PLIP 问题驱动预筛定位 | 关 |
| `--prescreen-k` | 候选区域数量 | 5 |
| `--neuro-symbolic / --no-neuro-symbolic` | 开结构化诊断纪律（强制 select + 覆盖要求 + 术语规范） | 关 |
| `--patho-r1 / --no-patho-r1` | 开 Patho-R1 专科会诊 + 结尾文字仲裁 | 关 |
| `-r / --runs` | 多数投票次数 | 1 |
| `--budget-usd` | 花费预算上限（美元，0 关闭） | 0 |
| `--enforce-fixed-iterations` | 走满步数、禁止提前回答（论文忠实模式） | 关 |
| `--strict-font-check` | 坐标刻度字体缺失时直接报错 | 关 |
| `-v` | 增加日志详细度（`-v`/`-vv`） | 0 |
| `--json` | 以 JSON 输出结果 | 关 |


### 支持的模型供应商

| 供应商 | 说明 | Key |
|---|---|---|
| `qwen` | DashScope Qwen-VL（本项目默认，国内直连） | `DASHSCOPE_API_KEY` |
| `openai` | OpenAI GPT 系列 | `OPENAI_API_KEY` |
| `anthropic` | Claude 系列 | `ANTHROPIC_API_KEY` |

---

## 项目结构

```text
src/giant/
├── agent/            # 导航核心：runner（ReAct 循环 + 空白/冗余拦截 + 仲裁）、
│                     #          context（对话状态 + 专家结论回灌）、trajectory（轨迹）
├── llm/              # LLM 层：qwen_client / openai_client / anthropic_client / protocol
│                     #          plip_planner（意图分类 + 维度拆解 + 观察计划 + 仲裁）
│                     #          patho_r1_client（Patho-R1 专科会诊 + 中文化）
│                     #          model_registry（模型注册表）、pricing（计费）、circuit_breaker
├── vision/           # 视觉：plip（问题驱动候选预筛）、segmentation（组织分割）、
│                     #       sampler、conch、aggregation
├── prompts/          # 提示词：templates（含神经符号纪律）、builder
├── geometry/         # 几何：坐标、裁剪、坐标轴刻度、候选框叠加
├── wsi/              # 切片读取（openslide）+ 异常
├── core/             # 裁剪引擎与 level 选择
├── utils/            # 日志
└── cli/              # 命令行入口与推理编排
```

---

## 环境变量

| 变量 | 用途 | 默认 |
|---|---|---|
| `DASHSCOPE_API_KEY` | Qwen-VL + 全部规划/翻译/仲裁调用 | 必填 |
| `DASHSCOPE_BASE_URL` | DashScope OpenAI 兼容端点 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `PATHO_R1_BASE_URL` | Patho-R1 vLLM 服务地址 | `http://localhost:18000/v1` |
| `PATHO_R1_MODEL` | Patho-R1 模型名 | `patho-r1-7b` |
| `OPENAI_API_KEY` | OpenAI 供应商 | — |
| `ANTHROPIC_API_KEY` | Anthropic 供应商 | — |
| `HUGGINGFACE_TOKEN` | 门控 HuggingFace 数据集 | — |
| `LOG_LEVEL` | 日志级别 | `INFO` |
| `LOG_FORMAT` | `console` 或 `json` | `console` |

其余可调参数（`MAX_ITERATIONS`、`THUMBNAIL_SIZE`、`WSI_LONG_SIDE_TARGET`、各家 RPM/温度、系统提示词覆盖等）见 [config.py](src/giant/config.py)。

---

## 参考

Pathovoyage 的多尺度导航思路参考了 GIANT（Gigapixel Image Agent for Navigating Tissue）的工作：

> *Buckley T. A. et al. "Navigating Gigapixel Pathology Images with Large Multimodal Models." arXiv:2511.19652, 2025.*

```bibtex
@article{buckley2025navigating,
  title={Navigating Gigapixel Pathology Images with Large Multimodal Models},
  author={Buckley, Thomas A. and Weihrauch, Kian R. and Latham, Katherine and
          Zhou, Andrew Z. and Manrai, Padmini A. and Manrai, Arjun K.},
  journal={arXiv preprint arXiv:2511.19652},
  year={2025}
}
```

- PLIP 预筛基于 [vinid/plip](https://huggingface.co/vinid/plip)（病理版 CLIP）。
- Patho-R1 专科会诊基于 [WenchuanZhang/Patho-R1-7B](https://huggingface.co/WenchuanZhang/Patho-R1-7B)。

---

<p align="center">
  <em>让 VLM 像病理医生一样，逐层放大、观察、推理，给出可解释的诊断。</em>
</p>
