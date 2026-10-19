"""定义切片元数据和读取接口，并提供层级坐标换算工具。"""

from dataclasses import dataclass
from typing import Protocol

from PIL import Image


@dataclass(frozen=True)
class WSIMetadata:
    """记录切片尺寸、层级降采样和像素物理尺寸。"""

    path: str
    width: int
    height: int
    level_count: int
    level_dimensions: tuple[tuple[int, int], ...]
    level_downsamples: tuple[float, ...]
    vendor: str
    mpp_x: float | None
    mpp_y: float | None

    @property
    def dimensions(self) -> tuple[int, int]:

        return (self.width, self.height)

    @property
    def aspect_ratio(self) -> float:

        return self.width / self.height

    def get_level_dimensions(self, level: int) -> tuple[int, int]:

        if level < 0 or level >= self.level_count:
            raise IndexError(f"Level {level} out of range [0, {self.level_count - 1}]")
        return self.level_dimensions[level]

    def get_downsample(self, level: int) -> float:

        if level < 0 or level >= self.level_count:
            raise IndexError(f"Level {level} out of range [0, {self.level_count - 1}]")
        return self.level_downsamples[level]


class WSIReaderProtocol(Protocol):
    """约定裁剪流程使用的切片读取接口。"""

    def get_metadata(self) -> WSIMetadata:

        ...

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> Image.Image:

        ...

    def get_thumbnail(self, max_size: tuple[int, int]) -> Image.Image:

        ...

    def close(self) -> None:

        ...


# 坐标变换工具函数。


def _validate_downsample(downsample: float) -> None:

    if downsample <= 0:
        raise ValueError(f"Downsample factor must be positive, got {downsample}")


def level0_to_level(
    coord: tuple[int, int],
    downsample: float,
) -> tuple[int, int]:
    """将原始切片坐标换算到指定金字塔层级。"""
    _validate_downsample(downsample)
    x, y = coord
    return (int(x / downsample), int(y / downsample))


def level_to_level0(
    coord: tuple[int, int],
    downsample: float,
) -> tuple[int, int]:
    """将金字塔层级坐标换算回原始切片坐标。"""
    _validate_downsample(downsample)
    x, y = coord
    return (int(x * downsample), int(y * downsample))


def size_at_level(
    size: tuple[int, int],
    downsample: float,
) -> tuple[int, int]:

    _validate_downsample(downsample)
    w, h = size
    return (max(1, int(w / downsample)), max(1, int(h / downsample)))


def size_to_level0(
    size: tuple[int, int],
    downsample: float,
) -> tuple[int, int]:

    _validate_downsample(downsample)
    w, h = size
    return (int(w * downsample), int(h * downsample))
