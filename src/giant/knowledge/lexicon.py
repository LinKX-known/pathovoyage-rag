"""中英病理术语对照表：把中文临床问题映射成可检索的英文病理词汇。

检索语料是英文教科书（Robbins / Pathoma），而用户提问几乎总是中文。
BM25 无法跨语言匹配，因此在检索前用本表把中文术语确定性地展开成英文，
既不依赖任何 API，也让「检索到哪一段」这件事可复现、可审计。

这张表同时充当 KG-RAG 的第一层种子（P1 会把「亚型—判据—鉴别」关系补全到 YAML）。
"""

from __future__ import annotations

import re

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# 中文术语 → 英文检索词。键必须是有区分度的病理实体或判据名词。
_LEXICON: dict[str, str] = {
    # —— 部位 ——
    "结直肠": "colorectal colon rectum",
    "直肠": "rectum rectal",
    "结肠": "colon colonic",
    "回盲部": "ileocecal",
    "阑尾": "appendix appendiceal",
    "胃": "gastric stomach",
    "食管": "esophageal esophagus",
    "小肠": "small intestine",
    "十二指肠": "duodenum",
    "肝": "liver hepatic",
    "胰腺": "pancreas pancreatic",
    "胆管": "biliary bile duct",
    "肺": "lung pulmonary",
    "乳腺": "breast mammary",
    "淋巴结": "lymph node",
    # —— 肿瘤类型 ——
    "腺癌": "adenocarcinoma",
    "黏液腺癌": "mucinous adenocarcinoma",
    "印戒细胞癌": "signet ring cell carcinoma",
    "印戒细胞": "signet ring cell",
    "髓样癌": "medullary carcinoma",
    "神经内分泌": "neuroendocrine",
    "癌": "carcinoma",
    "腺瘤": "adenoma",
    "息肉": "polyp",
    "上皮内瘤变": "intraepithelial neoplasia dysplasia",
    "锯齿状": "serrated",
    "淋巴瘤": "lymphoma",
    "转移": "metastasis metastatic",
    "原发": "primary",
    # —— 组织学判据 ——
    "黏液湖": "mucin pool extracellular mucin",
    "细胞外黏液": "extracellular mucin",
    "腺体": "gland glandular",
    "腺管": "gland tubular",
    "组织结构": "architecture architectural",
    "生长模式": "growth pattern",
    "分化程度": "differentiation grade",
    "高分化": "well differentiated low grade",
    "中分化": "moderately differentiated",
    "低分化": "poorly differentiated high grade",
    "未分化": "undifferentiated",
    "核异型": "nuclear atypia pleomorphism",
    "核分裂": "mitotic figure mitosis",
    "染色质": "chromatin",
    "核仁": "nucleolus",
    "胞质": "cytoplasm",
    "间质": "stroma stromal",
    "促结缔组织增生": "desmoplasia desmoplastic",
    "炎症": "inflammation inflammatory",
    "坏死": "necrosis necrotic",
    "溃疡": "ulcer ulceration",
    "钙化": "calcification",
    "黏附": "cohesion",
    # —— 浸润与分期 ——
    "浸润深度": "depth of invasion",
    "浸润": "invasion invasive",
    "浆膜": "serosa serosal",
    "浆膜侵犯": "subserosal invasion serosal penetration",
    "固有肌层": "muscularis propria",
    "黏膜下层": "submucosa submucosal",
    "切缘": "surgical margin resection margin",
    "脉管瘤栓": "lymphovascular invasion vascular invasion",
    "血管侵犯": "vascular invasion",
    "淋巴管": "lymphatic",
    "神经侵犯": "perineural invasion",
    "肿瘤出芽": "tumor budding",
    "分期": "staging stage",
    "T分期": "T stage depth of invasion",
    "N分期": "N stage nodal",
    "淋巴结转移": "lymph node metastasis",
    # —— 结直肠专项（分期/报告规范/分子）——
    "环周切缘": "circumferential resection margin CRM",
    "系膜": "mesorectum mesorectal",
    "全直肠系膜切除": "total mesorectal excision TME",
    "新辅助": "neoadjuvant chemoradiation",
    "术前放化疗": "preoperative chemoradiotherapy",
    "微卫星": "microsatellite instability MSI",
    "错配修复": "mismatch repair MMR",
    "淋巴结检出": "lymph node harvest",
    "检出淋巴结": "lymph nodes examined",
    "腹膜": "peritoneum peritoneal",
    "穿孔": "perforation",
    "梗阻": "obstruction",
    "黏附性": "cohesive",
    "低黏附": "poorly cohesive",
    "筛状": "cribriform",
    "融合腺体": "fused glands",
    "浸润前沿": "invasive front",
    "肿瘤前沿": "tumor front",
    "个体化": "personalized",
    # —— 镜下描述与操作 ——
    "形态学": "morphology",
    "组织学": "histology histologic",
    "亚型": "subtype classification",
    "活检": "biopsy",
    "切片": "section slide",
    "苏木精伊红": "hematoxylin eosin H&E",
    "免疫组化": "immunohistochemistry",
    "上皮": "epithelium epithelial",
    "杯状细胞": "goblet cell",
    "杯状": "goblet",
    "细胞": "cell cellular",
    "诊断": "diagnosis diagnostic",
    "鉴别诊断": "differential diagnosis",
    "鉴别": "differential distinction",
    "预后": "prognosis prognostic",
    "分级": "grading grade",
}

# 展开后仍过于宽泛、单独出现时帮助不大的词（避免稀释 BM25 查询）。
_LOW_SIGNAL: frozenset[str] = frozenset({"cell", "cellular", "primary", "diagnosis", "diagnostic"})


def has_cjk(text: str) -> bool:
    """判断文本是否含中文，用于决定是否需要术语展开。"""
    return bool(_CJK_RE.search(text))


def expand_query(text: str) -> tuple[str, list[str]]:
    """把中文问题展开成「原文 + 英文术语」，返回 (检索串, 命中的术语列表)。

    匹配按术语长度从长到短进行，长术语命中后其子串术语不再重复加入，
    避免「黏液腺癌」同时带出「腺癌」「癌」把查询稀释。
    """
    matched_terms: list[tuple[str, str]] = []
    for zh in sorted(_LEXICON, key=len, reverse=True):
        if zh not in text:
            continue
        if any(zh in longer for longer, _ in matched_terms):
            continue
        matched_terms.append((zh, _LEXICON[zh]))

    expansions: list[str] = []
    for _, english in matched_terms:
        for word in english.split():
            if word not in expansions and word not in _LOW_SIGNAL:
                expansions.append(word)

    query = text if not expansions else f"{text} {' '.join(expansions)}"
    return query.strip(), [zh for zh, _ in matched_terms]
