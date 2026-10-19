"""在原始切片坐标与金字塔层级坐标之间转换点和区域。"""

from __future__ import annotations

from giant.geometry.primitives import Point, Region, Size
from giant.wsi.types import (
    level0_to_level,
    level_to_level0,
    size_at_level,
    size_to_level0,
)


__all__ = [
    "level0_to_level",
    "level_to_level0",
    "point_level0_to_level",
    "point_level_to_level0",
    "region_level0_to_level",
    "region_level_to_level0",
    "size_at_level",
    "size_level0_to_level",
    "size_level_to_level0",
    "size_to_level0",
]


def point_level0_to_level(point: Point, downsample: float) -> Point:

    transformed = level0_to_level(point.to_tuple(), downsample)
    return Point.from_tuple(transformed)


def point_level_to_level0(point: Point, downsample: float) -> Point:

    transformed = level_to_level0(point.to_tuple(), downsample)
    return Point.from_tuple(transformed)


def size_level0_to_level(size: Size, downsample: float) -> Size:

    transformed = size_at_level(size.to_tuple(), downsample)
    return Size.from_tuple(transformed)


def size_level_to_level0(size: Size, downsample: float) -> Size:

    transformed = size_to_level0(size.to_tuple(), downsample)
    return Size.from_tuple(transformed)


def region_level0_to_level(region: Region, downsample: float) -> Region:

    origin = level0_to_level((region.x, region.y), downsample)
    dims = size_at_level((region.width, region.height), downsample)
    return Region(
        x=origin[0],
        y=origin[1],
        width=dims[0],
        height=dims[1],
    )


def region_level_to_level0(region: Region, downsample: float) -> Region:

    origin = level_to_level0((region.x, region.y), downsample)
    dims = size_to_level0((region.width, region.height), downsample)
    return Region(
        x=origin[0],
        y=origin[1],
        width=dims[0],
        height=dims[1],
    )
