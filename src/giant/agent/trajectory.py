"""保存每一步的观察、动作与最终诊断，供推理结果追溯。"""

# 单步记录保存观察图、模型动作与裁剪区域；整条轨迹汇总这些记录和最终答案。

from __future__ import annotations

from pydantic import BaseModel, Field

from giant.geometry.primitives import Region
from giant.knowledge.schemas import Passage
from giant.llm.protocol import StepResponse


class Turn(BaseModel):
    """记录一次模型决策所看到的图像、动作和区域信息。"""

    step_index: int = Field(..., ge=0, description="Zero-indexed step number")  # 第几步（从 0 开始）
    image_base64: str = Field(..., description="Base64-encoded image")  # 本步看到的图
    response: StepResponse = Field(..., description="Model's reasoning and action")  # 推理+动作
    region: Region | None = Field(
        default=None,
        description="Region that was cropped (None for thumbnail)",
    )
    conch_scores: list[float] | None = Field(
        default=None,
        description=(
            "CONCH similarity scores aligned to ConchAction.hypotheses "
            "(paper ablation feature)."
        ),
    )
    expert_opinion: str | None = Field(
        default=None,
        description=(
            "Patho-R1 专家对该 crop 区域的一句话诊断结论。"
        ),
    )
    magnification: str | None = Field(
        default=None,
        description="该 crop 会诊时的倍率档位（low/medium/high）。",
    )
    observation_task: str | None = Field(
        default=None,
        description="该 crop 倍率对应的观察任务文本（千问按问题生成，英文）。",
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="该步 Patho-R1 会诊时注入的证据编号（E1、E2…），用于追溯。",
    )


# 是可视化和评测的数据来源。
class Trajectory(BaseModel):
    """汇集一张切片的所有导航步骤和最终答案。"""

    wsi_path: str = Field(..., description="Path to the WSI file")  # 切片路径
    question: str = Field(..., description="Diagnostic question")  # 诊断问题
    turns: list[Turn] = Field(default_factory=list, description="Navigation turns")
    final_answer: str | None = Field(
        default=None,
        description="Final answer (if completed)",  # 最终答案
    )
    slide_width: int | None = Field(
        default=None, description="Width of the WSI level 0"
    )
    slide_height: int | None = Field(
        default=None, description="Height of the WSI level 0"
    )
    thumbnail_base64: str | None = Field(
        default=None, description="Initial WSI thumbnail"  # 缩略图（带坐标刻度）
    )
    plip_plan: dict | None = Field(
        default=None, description="PLIP 检索计划（意图 + 检索维度），千问按问题生成"
    )
    patho_r1_plan: dict | None = Field(
        default=None, description="Patho-R1 分层观察计划（判据 + 低/中/高倍任务）"
    )
    patho_r1_arbitration: dict | None = Field(
        default=None,
        description=(
            "Patho-R1 结论仲裁记录：各区域专家结论（opinions）、仲裁后的答案（answer）"
            "与导航模型的原始答案（original_answer）。"
        ),
    )
    rag_evidence: list[Passage] = Field(
        default_factory=list,
        description="RAG 检索到的知识片段（含出处与分数），供诊断报告引用与审计。",
    )
    rag_corpus_version: str = Field(
        default="", description="知识语料指纹，保证诊断结论可复现"
    )
    rag_gate: list[dict] = Field(
        default_factory=list,
        description="检索闸门判定记录（各阶段的相关性标签、保留/丢弃条数）",
    )
    final_citation: dict | None = Field(
        default=None,
        description="最终结论对证据的引用统计（used/unused/invalid/coverage）",
    )
