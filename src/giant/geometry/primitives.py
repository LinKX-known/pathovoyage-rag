"""定义点、尺寸和矩形区域及其几何关系。"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, Field


class Point(BaseModel, frozen=True):
    """表示二维坐标点。"""

    x: int = Field(..., ge=0, description="X coordinate (pixels from left)")
    y: int = Field(..., ge=0, description="Y coordinate (pixels from top)")

    def to_tuple(self) -> tuple[int, int]:

        return (self.x, self.y)

    @classmethod
    def from_tuple(cls, coord: tuple[int, int]) -> Self:

        return cls(x=coord[0], y=coord[1])


class Size(BaseModel, frozen=True):
    """表示二维尺寸并计算面积。"""

    width: int = Field(..., gt=0, description="Width in pixels")
    height: int = Field(..., gt=0, description="Height in pixels")

    @property
    def area(self) -> int:

        return self.width * self.height

    def to_tuple(self) -> tuple[int, int]:

        return (self.width, self.height)

    @classmethod
    def from_tuple(cls, size: tuple[int, int]) -> Self:

        return cls(width=size[0], height=size[1])


class Region(BaseModel, frozen=True):
    """表示矩形区域并计算包含、相交等几何关系。"""

    x: int = Field(..., ge=0, description="Left edge X coordinate")
    y: int = Field(..., ge=0, description="Top edge Y coordinate")
    width: int = Field(..., gt=0, description="Width in pixels")
    height: int = Field(..., gt=0, description="Height in pixels")

    @property
    def area(self) -> int:

        return self.width * self.height

    @property
    def right(self) -> int:

        return self.x + self.width

    @property
    def bottom(self) -> int:

        return self.y + self.height

    @property
    def top_left(self) -> Point:

        return Point(x=self.x, y=self.y)

    @property
    def size(self) -> Size:

        return Size(width=self.width, height=self.height)

    @property
    def center(self) -> tuple[int, int]:

        return (self.x + self.width // 2, self.y + self.height // 2)

    def to_tuple(self) -> tuple[int, int, int, int]:

        return (self.x, self.y, self.width, self.height)

    @classmethod
    def from_tuple(cls, bbox: tuple[int, int, int, int]) -> Self:

        return cls(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3])

    @classmethod
    def from_corners(
        cls,
        top_left: tuple[int, int],
        bottom_right: tuple[int, int],
    ) -> Self:

        x1, y1 = top_left
        x2, y2 = bottom_right
        width = x2 - x1
        height = y2 - y1
        return cls(x=x1, y=y1, width=width, height=height)

    def contains_point(self, point: Point) -> bool:

        return self.x <= point.x < self.right and self.y <= point.y < self.bottom

    def intersects(self, other: Region) -> bool:

        return not (
            other.x >= self.right
            or other.right <= self.x
            or other.y >= self.bottom
            or other.bottom <= self.y
        )

    def intersection(self, other: Region) -> Region | None:

        if not self.intersects(other):
            return None

        new_x = max(self.x, other.x)
        new_y = max(self.y, other.y)
        new_right = min(self.right, other.right)
        new_bottom = min(self.bottom, other.bottom)

        return Region(
            x=new_x,
            y=new_y,
            width=new_right - new_x,
            height=new_bottom - new_y,
        )
