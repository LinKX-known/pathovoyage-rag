"""校验裁剪区域并将越界范围限制在切片边界内。"""

from __future__ import annotations

from giant.geometry.primitives import Region, Size


class ValidationError(Exception):
    """几何坐标或尺寸无效时抛出的错误。"""

    def __init__(
        self,
        message: str,
        *,
        region: Region,
        bounds: Size,
    ) -> None:
        self.region = region
        self.bounds = bounds
        region_str = region.to_tuple()
        bounds_str = bounds.to_tuple()
        super().__init__(f"{message} (region={region_str}, bounds={bounds_str})")


class GeometryValidator:
    """校验区域尺寸和切片边界，必要时裁剪越界部分。"""

    def validate(
        self,
        region: Region,
        bounds: Size,
        *,
        strict: bool = True,
    ) -> bool:

        is_valid = region.right <= bounds.width and region.bottom <= bounds.height

        if not is_valid and strict:
            violations: list[str] = []
            if region.right > bounds.width:
                violations.append(
                    f"right edge ({region.right}) exceeds width ({bounds.width})"
                )
            if region.bottom > bounds.height:
                violations.append(
                    f"bottom edge ({region.bottom}) exceeds height ({bounds.height})"
                )
            raise ValidationError(
                f"Region out of bounds: {'; '.join(violations)}",
                region=region,
                bounds=bounds,
            )

        return is_valid

    def clamp_region(self, region: Region, bounds: Size) -> Region:


        # 减 1 确保至少能容纳 1 像素。
        clamped_x = max(0, min(region.x, bounds.width - 1))
        clamped_y = max(0, min(region.y, bounds.height - 1))

        # 计算剩余的最大可用空间。
        max_remaining_width = bounds.width - clamped_x
        max_remaining_height = bounds.height - clamped_y

        # 将尺寸限制在剩余空间内，最小为 1 像素。
        clamped_width = max(1, min(region.width, max_remaining_width))
        clamped_height = max(1, min(region.height, max_remaining_height))

        return Region(
            x=clamped_x,
            y=clamped_y,
            width=clamped_width,
            height=clamped_height,
        )

    def is_within_bounds(self, region: Region, bounds: Size) -> bool:

        return self.validate(region, bounds, strict=False)
