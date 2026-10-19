"""按颜色饱和度分割染色组织，并清理小连通区域。"""

from __future__ import annotations

from typing import Literal, cast

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image

from giant.vision.constants import MIN_TISSUE_AREA, MORPH_KERNEL_SIZE

# 支持的后端类型。
BackendType = Literal["parity", "clam"]
SUPPORTED_BACKENDS: frozenset[str] = frozenset({"parity", "clam"})


class TissueSegmentor:
    """根据染色饱和度提取组织区域并去除小噪声。"""

    __slots__ = ("_backend", "_kernel")

    def __init__(self, backend: BackendType = "parity") -> None:

        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{backend}'. "
                f"Supported: {sorted(SUPPORTED_BACKENDS)}"
            )
        self._backend = backend
        # 预先创建闭运算的形态学核。
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
        )

    @property
    def backend(self) -> str:

        return self._backend

    def segment(self, image: Image.Image) -> npt.NDArray[np.bool_]:
        """对输入图像生成组织掩码，并清理小面积噪声。"""

        if image.mode != "RGB":
            # 拒绝灰度图像，但尝试转换其他图像模式。
            if image.mode == "L":
                raise ValueError(
                    "Grayscale images not supported. Please provide RGB image."
                )
            image = image.convert("RGB")

        arr = np.array(image)


        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

        # 提取饱和度通道（索引 1）。
        saturation = hsv[:, :, 1]


        # 染色组织的饱和度通常高于白色背景。
        _, mask = cv2.threshold(
            saturation,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        # 使用形态学闭运算填补小孔洞。
        mask_uint8 = cast(npt.NDArray[np.uint8], mask.astype(np.uint8, copy=False))
        mask_uint8 = cast(
            npt.NDArray[np.uint8],
            cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, self._kernel),
        )

        mask_uint8 = self._remove_small_components(mask_uint8)

        return mask_uint8 > 0

    @staticmethod
    def _remove_small_components(mask: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:

        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
        if num_labels <= 1:
            return mask

        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < MIN_TISSUE_AREA:
                mask[labels == label] = 0

        return mask


def segment_tissue(
    image: Image.Image,
    backend: BackendType = "parity",
) -> npt.NDArray[np.bool_]:

    segmentor = TissueSegmentor(backend=backend)
    return segmentor.segment(image)
