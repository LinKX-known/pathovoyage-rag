"""导出组织分割、候选区域预筛和图文评分接口。"""

from __future__ import annotations

from giant.vision.segmentation import TissueSegmentor, segment_tissue

__all__ = [
    "TissueSegmentor",
    "segment_tissue",
]
