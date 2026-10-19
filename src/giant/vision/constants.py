"""保存组织分割和图像预处理使用的默认参数。"""

from __future__ import annotations

DEFAULT_THUMBNAIL_SIZE: tuple[int, int] = (2048, 2048)
MORPH_KERNEL_SIZE: int = 5
MIN_TISSUE_AREA: int = 100
