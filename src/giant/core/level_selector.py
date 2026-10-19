"""根据目标图像尺寸从切片金字塔中选择合适的读取层级。"""

from __future__ import annotations

from typing import NamedTuple, Protocol

from giant.geometry import Region
from giant.wsi.types import WSIMetadata


class SelectedLevel(NamedTuple):
    """保存选中的金字塔层级及降采样系数。"""

    level: int
    downsample: float


class LevelSelectorProtocol(Protocol):
    """约定裁剪引擎所需的层级选择接口。"""

    def select_level(
        self,
        region: Region,
        metadata: WSIMetadata,
        target_size: int = 1000,
        bias: float = 0.85,
    ) -> SelectedLevel:

        ...


class PyramidLevelSelector:
    """优先选接近目标尺寸且像素足够的金字塔层级。"""

    def select_level(
        self,
        region: Region,
        metadata: WSIMetadata,
        target_size: int = 1000,
        bias: float = 0.85,
    ) -> SelectedLevel:
        """选取最接近目标像素尺寸的层级，并避免读取尺寸过小。"""
        self._validate_parameters(target_size, bias)


        region_long_side = max(region.width, region.height)
        target_native = target_size / bias


        best_k = self._find_closest_level(region_long_side, target_native, metadata)

        # 第三步：修正分辨率不足的情况。
        best_k = self._apply_undershoot_correction(
            best_k, region_long_side, target_size, metadata
        )

        return SelectedLevel(
            level=best_k,
            downsample=metadata.level_downsamples[best_k],
        )

    def _validate_parameters(self, target_size: int, bias: float) -> None:

        if target_size <= 0:
            raise ValueError(f"target_size must be positive, got {target_size}")
        if bias <= 0:
            raise ValueError(f"bias must be positive, got {bias}")

    def _find_closest_level(
        self,
        region_long_side: int,
        target_native: float,
        metadata: WSIMetadata,
    ) -> int:

        best_k = 0
        best_diff = float("inf")

        # 按层级升序遍历（从最精细到最粗糙）。
        # 使用严格小于号可保留第一个达到最小差值的层级，
        # 因而距离相同时自然优先选择更精细的层级。
        for k, ds in enumerate(metadata.level_downsamples):

            size_at_level = region_long_side / ds
            diff = abs(size_at_level - target_native)

            if diff < best_diff:
                best_diff = diff
                best_k = k

        return best_k

    def _apply_undershoot_correction(
        self,
        level: int,
        region_long_side: int,
        target_size: int,
        metadata: WSIMetadata,
    ) -> int:

        while level > 0:

            size_at_level = region_long_side / metadata.level_downsamples[level]
            if size_at_level >= target_size:
                break
            level -= 1

        return level
