"""把诊断问题规划成图文检索条件、分倍率观察任务及最终仲裁输入。"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from giant.knowledge.schemas import ArbitrationResult, WeightedEvidence
from giant.llm.qwen_client import QwenProvider
from giant.utils.logging import get_logger

logger = get_logger(__name__)

# 兜底检索文本：问题无法翻译成任何特定视觉描述时的保守默认。
FALLBACK_QUERY = "colorectal adenocarcinoma tumor epithelium, H&E"

INTENT_EXISTENCE = "existence"
INTENT_CLASSIFICATION = "classification"
INTENT_QUANTIFICATION = "quantification"

_SYSTEM_PROMPT = (
    "You are a pathology image-retrieval assistant. You classify a clinical "
    "question and turn it into one or more English visual descriptions that a "
    "pathology vision-language model can use to search whole-slide images."
)


@dataclass(frozen=True)
class RetrievalDimension:
    """保存一个检索维度的描述及其权重。"""

    name: str
    text: str
    weight: float


@dataclass(frozen=True)
class RetrievalPlan:
    """保存问题意图、检索文本和加权维度。"""

    intent: str
    dimensions: list[RetrievalDimension]


@dataclass(frozen=True)
class ObservationPlan:
    """保存不同放大倍率下的观察任务和诊断判据。"""

    criteria: str
    low_task: str
    medium_task: str
    high_task: str


_INTENT_CLASSIFY_PROMPT = """\
用户的问题是："{question}"

请判断这个问题属于以下哪一类：
A. 存在性检测（有/无特定特征）
   - 示例："有没有脉管瘤栓？"、"切缘干净吗？"、"有坏死吗？"
B. 分类/定性问题（是什么类型、什么程度）
   - 示例："组织学亚型是什么？"、"分化程度如何？"
C. 程度/量化问题（多深、多远）
   - 示例："T分期是多少？"、"浸润多深？"

只输出 A、B 或 C，不要解释。
"""


_TRANSLATE_PROMPT = """\
# 任务
将用户的临床问题，转换为适合 PLIP 模型的图像检索文本（Query）。

# 核心规则
- PLIP 通过图像与文本的语义匹配来检索相似图片。
- 输出必须是对目标病理形态的视觉描述，而不是临床诊断结论。
- 只输出一句英文检索文本，20 词以内，不要有多余解释。

# 翻译对照参考
| 用户问题关键词 | 检索文本（视觉描述） |
|:---|:---|
| 切缘/切干净没 | tumor cells at the inked surgical margin |
| 脉管瘤栓/血管侵犯 | tumor cells inside blood vessel lumen lined by endothelial cells |
| 神经侵犯 | tumor cells invading peripheral nerve bundles |
| 黏液腺癌/黏液湖 | extracellular mucin pools containing floating clusters of carcinoma cells |
| 坏死 | tumor necrosis with cellular debris and ghost cells |
| 肿瘤出芽 | single tumor cells or small clusters at invasive front |
| T分期/浸润多深 | tumor cells at the deepest invasion front |
| 浆膜侵犯/T4 | tumor cells infiltrating into subserosal adipose tissue |

# 保底策略
如果你无法将用户问题翻译为任何特定的视觉描述，使用保底文本：
"{fallback}"

# 输出格式
只输出一句英文检索文本，不要解释，不要输出其他内容。

用户的问题是："{question}"
"""


_DIMENSION_DECOMPOSE_PROMPT = """\
# 任务
用户的临床问题是分类/定性问题。请将该问题拆解为 3-5 个关键诊断维度，
为每个维度生成 PLIP 检索文本，并分配权重。

# 输出格式
严格按照以下 JSON 格式输出，只输出 JSON，不要任何解释性文字：

{{
  "dimensions": [
    {{"name": "维度名称", "text": "英文检索文本", "weight": 0.30}}
  ]
}}

# 权重规则
- 所有权重之和必须等于 1.0
- 权重反映该维度在诊断中的重要性
- 结直肠组织学亚型题中，「黏液特征」是鉴别黏液腺癌/印戒细胞癌的关键，权重应最高

# 临床参考
- 组织学亚型：黏液特征(0.65)、腺体结构(0.15)、细胞形态(0.10)、分化程度(0.10)
- 分化程度：腺体形成比例(0.50)、核异型性(0.30)、核分裂象(0.20)

用户的问题是："{question}"
"""


_OBSERVATION_PROMPT = """\
# Task
For the clinical question below, generate a "layered observation plan" to help a
pathology vision-language model look at the right morphology at each magnification.

# Background
Pathology diagnosis is done by magnification: low power = overall architecture and
growth pattern; medium power = glandular arrangement and stromal reaction; high
power = nuclear and cytoplasmic detail. So the observation task differs by power.

# Output format
Output STRICT JSON only, nothing else:

{{
  "criteria": "discriminative microscopic features to distinguish the possible answers, plus mimics to rule out; 2-4 bullet points",
  "low_task": "one sentence: what to observe at low power (large field)",
  "medium_task": "one sentence: what to observe at medium power",
  "high_task": "one sentence: what to observe at high power (cellular level)"
}}

# Rules
- low_task targets overall architecture / growth pattern / locating abnormal regions;
- medium_task targets glandular arrangement / invasion / stromal reaction;
- high_task targets nuclear atypia / cytoplasmic features / special structures (mucin, signet ring, vessels, etc., depending on the question).
- Each task is ONE English sentence describing WHAT TO OBSERVE, not a diagnosis.
- criteria must be NEUTRAL and cover ALL possible answers: for a classification/typing question, list the distinguishing features of EACH option, NOT evidence for a single presumed diagnosis. English, 2-4 bullet points, one per line.

Clinical question: {question}
"""


_ARBITRATION_PROMPT = """\
# 任务
下面是多位病理专家（Patho-R1）对同一张切片的不同区域、各自**独立**给出的镜下判断
（按观察顺序排列）。请**只依据这些专家结论**回答下面的临床问题。

# 铁律
- 你看不到任何图像，只能看到文字结论。禁止加入你自己的图像观察、记忆或任何推测；
  你写下的每一条判断都必须能在下面的专家结论里找到出处。
- 专家结论可能互相矛盾——这通常是肿瘤异质性，或不同区域取到了不同成分。你必须权衡后
  给出**一个**统一结论，而不是把几种说法罗列一遍让人自己选。
- 若专家结论明显分成两派，以**多数**为准，并指出少数派结论出现在哪些区域。
- 只有当专家结论确实不足以回答该问题时，才回答「专家证据不足」，并说明还缺什么证据。

# 输出要求
用简体中文。先用一句话给出最终结论，再用 2-4 句说明你是依据哪些区域的哪些结论得出的，
以及少数派意见出现在哪里。不要写随访或治疗建议。

# 临床问题
{question}

# 各位专家的结论
{opinions}
"""


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.lstrip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.rstrip("`").strip()
    return stripped


def _parse_json(text: str) -> dict | None:

    try:
        data, _ = json.JSONDecoder().raw_decode(_strip_fences(text))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _parse_intent(raw: str) -> str:

    s = raw.strip().upper()
    if "B" in s:
        return INTENT_CLASSIFICATION
    if "C" in s:
        return INTENT_QUANTIFICATION
    if "A" in s:
        return INTENT_EXISTENCE
    raise ValueError(f"无法识别意图: {raw!r}")


def _normalize_weights(dimensions: list[RetrievalDimension]) -> list[RetrievalDimension]:
    """把权重归一化到和 = 1.0；全 0 时均分。"""
    total = sum(d.weight for d in dimensions)
    if total <= 0:
        n = len(dimensions)
        return [RetrievalDimension(d.name, d.text, 1.0 / n) for d in dimensions]
    return [RetrievalDimension(d.name, d.text, d.weight / total) for d in dimensions]


async def _translate(provider: QwenProvider, question: str, reference: str = "") -> str:

    prompt = _TRANSLATE_PROMPT.format(question=question, fallback=FALLBACK_QUERY)
    if reference:
        prompt = f"{prompt}\n\n# 参考证据（来自病理教科书，用于校准术语）\n{reference}"
    text = (
        await provider.complete_text(
            system=_SYSTEM_PROMPT,
            user=prompt,
            max_tokens=128,
        )
    ).strip()
    return text or FALLBACK_QUERY


async def _decompose(
    provider: QwenProvider, question: str, reference: str = ""
) -> list[RetrievalDimension]:

    prompt = _DIMENSION_DECOMPOSE_PROMPT.format(question=question)
    if reference:
        prompt = f"{prompt}\n\n# 参考证据（来自病理教科书，用于校准维度与权重）\n{reference}"
    raw = await provider.complete_text(
        system=_SYSTEM_PROMPT,
        user=prompt,
        max_tokens=512,
    )
    data = _parse_json(raw)
    items = (data or {}).get("dimensions")
    if not isinstance(items, list) or not items:
        raise ValueError(f"维度拆解输出非法: {raw[:200]!r}")

    dims: list[RetrievalDimension] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        name = str(item.get("name") or "").strip()
        try:
            weight = float(item.get("weight", 0.0))
        except (TypeError, ValueError):
            weight = 0.0
        dims.append(RetrievalDimension(name=name, text=text, weight=weight))
    if not dims:
        raise ValueError("维度拆解结果为空")
    return _normalize_weights(dims)


def _reference_snippet(evidence: object | None, *, max_chars: int = 700) -> str:
    """把检索证据压成一段短参考文本，供规划类提示词使用。"""
    if evidence is None:
        return ""
    passages = getattr(evidence, "passages", None) or []
    if not passages:
        return ""
    lines: list[str] = []
    used = 0
    for index, passage in enumerate(passages[:3]):
        text = " ".join(str(getattr(passage, "text", "")).split())
        if not text:
            continue
        room = max_chars - used
        if room < 80:
            break
        snippet = text[: room - 8].rstrip() + ("…" if len(text) > room - 8 else "")
        lines.append(f"[E{index + 1}] {snippet}")
        used += len(snippet)
    return "\n".join(lines)


async def plan_retrieval(question: str, *, evidence: object | None = None) -> RetrievalPlan:
    """按诊断问题生成用于候选区域预筛的检索计划。

    evidence 非空时，检索到的教科书片段会作为术语参考注入提示词，
    让生成的检索维度更贴近病理学表述。

    千问失败或返回空时，退回**确定性构造**（判据卡英文短语 → 术语表 → 最后兜底），
    而不是所有问题共用一句写死的文本。
    """
    reference = _reference_snippet(evidence)
    try:
        provider = QwenProvider()

        intent_raw = await provider.complete_text(
            system=_SYSTEM_PROMPT,
            user=_INTENT_CLASSIFY_PROMPT.format(question=question),
            max_tokens=16,
        )
        intent = _parse_intent(intent_raw)

        if intent == INTENT_CLASSIFICATION:
            dimensions = await _decompose(provider, question, reference)
        else:
            text = await _translate(provider, question, reference)
            dimensions = [RetrievalDimension(name="", text=text, weight=1.0)]

        # 翻译返回空时 _translate 会给出 FALLBACK_QUERY：这种情况同样改用确定性维度
        if len(dimensions) == 1 and dimensions[0].text.strip() == FALLBACK_QUERY:
            dimensions = _deterministic_dimensions(question, evidence) or dimensions

        return RetrievalPlan(intent=intent, dimensions=dimensions)
    except Exception as e:
        logger.warning("PLIP 检索规划失败，改用确定性维度: %s", e)
        return RetrievalPlan(
            intent=INTENT_EXISTENCE,
            dimensions=_deterministic_dimensions(question, evidence)
            or [RetrievalDimension(name="", text=FALLBACK_QUERY, weight=1.0)],
        )


def _deterministic_dimensions(
    question: str, evidence: object | None = None
) -> list[RetrievalDimension]:
    """用判据卡英文短语 / 术语表构造检索维度（不依赖任何 API）。"""
    try:
        from giant.knowledge.plip_query import build_dimensions

        queries = build_dimensions(question, evidence)  # type: ignore[arg-type]
    except Exception as e:  # 知识层异常不应影响预筛
        logger.warning("确定性检索维度构造失败: %s", e)
        return []
    return [
        RetrievalDimension(name=query.source, text=query.text, weight=query.weight)
        for query in queries
    ]


async def plan_observation(question: str, *, evidence: object | None = None) -> ObservationPlan:
    """根据问题生成低、中、高倍率的观察任务。

    evidence 非空时注入「参考判据」，使判据跟着指南走，而不是由模型凭空生成。
    """
    try:
        provider = QwenProvider()
        prompt = _OBSERVATION_PROMPT.format(question=question)
        reference = _reference_snippet(evidence, max_chars=900)
        if reference:
            prompt = (
                f"{prompt}\n\n# Retrieved reference criteria (background only; "
                f"do not copy verbatim)\n{reference}"
            )
        raw = await provider.complete_text(
            system="You are a pathology diagnostic assistant.",
            user=prompt,
            max_tokens=512,
        )
        data = _parse_json(raw) or {}
        return ObservationPlan(
            criteria=str(data.get("criteria") or "").strip(),
            low_task=str(data.get("low_task") or "").strip(),
            medium_task=str(data.get("medium_task") or "").strip(),
            high_task=str(data.get("high_task") or "").strip(),
        )
    except Exception as e:
        logger.warning("分层观察计划生成失败，回退空计划: %s", e)
        return ObservationPlan(criteria="", low_task="", medium_task="", high_task="")


_STRUCTURED_OUTPUT_SECTION = """\
# 输出格式（必须是严格 JSON，不要输出 JSON 以外的任何文字）
{{
  "answer": "一句话统一结论",
  "weighted_evidence": [
    {{"region": "区域坐标或位置描述", "opinion": "该区域专家结论摘要",
      "weight": 0.0, "why": "为什么给这个权重"}}
  ],
  "conflicts": ["哪些区域的结论互相矛盾、矛盾点是什么"],
  "minority": ["少数派结论及其出现在哪些区域"],
  "citations": ["E1", "E2"]
}}

要求：
- weight 取 0~1，表示该条证据对最终结论的贡献，不要求相加为 1；
- 专家结论互相矛盾时，按「多数 + 证据强度」权衡后给**一个**结论，不要罗列；
- citations 只能引用上面实际提供的证据编号。
"""

_CITATION_ID_RE = re.compile(r"^E\d+$")


def _coerce_weight(value: object) -> float:
    try:
        weight = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return min(max(weight, 0.0), 1.0)


def _parse_arbitration(raw: str) -> ArbitrationResult:
    """把仲裁输出解析成结构化结果；解析失败时退化为「原文即答案」。"""
    text = (raw or "").strip()
    data = _parse_json(text)
    if not isinstance(data, dict):
        return ArbitrationResult(answer=text, structured=False, raw=text)

    answer = str(data.get("answer") or "").strip()
    if not answer:
        return ArbitrationResult(answer=text, structured=False, raw=text)

    weighted: list[WeightedEvidence] = []
    for item in data.get("weighted_evidence") or []:
        if not isinstance(item, dict):
            continue
        weighted.append(
            WeightedEvidence(
                region=str(item.get("region") or "")[:120],
                opinion=str(item.get("opinion") or "")[:300],
                weight=_coerce_weight(item.get("weight")),
                why=str(item.get("why") or "")[:200],
            )
        )

    def _str_list(key: str) -> list[str]:
        value = data.get(key) or []
        if not isinstance(value, list):
            return []
        return [str(item)[:300] for item in value if str(item).strip()]

    citations = [
        str(item).strip().strip("[]")
        for item in (data.get("citations") or [])
        if _CITATION_ID_RE.match(str(item).strip().strip("[]"))
    ]

    return ArbitrationResult(
        answer=answer,
        weighted_evidence=weighted,
        conflicts=_str_list("conflicts"),
        minority=_str_list("minority"),
        citations=citations,
        structured=True,
        raw=text,
    )


async def arbitrate_structured(
    question: str,
    opinions: Sequence[str],
    *,
    evidence: object | None = None,
) -> ArbitrationResult:
    """结构化仲裁：返回答案 + 加权证据 + 冲突 + 少数派 + 引用编号。

    与自由文本仲裁的区别在于「可统计」：加权证据、冲突、引用都是字段，
    既能写进轨迹供审计，也是后续可靠性先验的数据来源。
    """
    if not opinions:
        return ArbitrationResult()

    try:
        from giant.knowledge.cards import evidence_block

        provider = QwenProvider()
        user_prompt = _ARBITRATION_PROMPT.format(
            question=question, opinions="\n".join(opinions)
        )
        block = evidence_block(evidence) if evidence is not None else ""
        if block:
            user_prompt = (
                f"{user_prompt}\n\n# 指南证据（可引用）\n{block}\n\n"
                "# 额外要求\n"
                "在结论与依据中，凡是来自上述证据的判断，请标注对应编号（如 [E1]）；"
                "证据不足以支持的推断不要写。"
            )
        user_prompt = f"{user_prompt}\n\n{_STRUCTURED_OUTPUT_SECTION}"

        text = await provider.complete_text(
            system=(
                "You are a pathology diagnostic arbiter. You reconcile written "
                "expert opinions into a single diagnosis, and you cite the "
                "evidence ids you rely on."
            ),
            user=user_prompt,
            max_tokens=1024,
        )
        return _parse_arbitration(text)
    except Exception as e:
        logger.warning("Patho-R1 结论仲裁失败，保留原答案: %s", e)
        return ArbitrationResult()


async def arbitrate_patho_r1(
    question: str,
    opinions: Sequence[str],
    *,
    evidence: object | None = None,
) -> str:
    """仅汇总专家文字结论，生成最终诊断（兼容旧调用方，返回答案文本）。

    evidence 非空时提供带编号的 `[E1]…` 证据块，要求仲裁者逐条引用出处。
    """
    result = await arbitrate_structured(question, opinions, evidence=evidence)
    return result.answer
