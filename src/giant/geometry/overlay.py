"""在缩略图上绘制坐标刻度和候选区域，辅助模型定位。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

from giant.utils.logging import get_logger

if TYPE_CHECKING:
    from giant.geometry.primitives import Region
    from giant.wsi.types import WSIMetadata

logger = get_logger(__name__)


@dataclass(frozen=True)
class OverlayStyle:
    """定义坐标参考线的颜色、粗细和字体样式。"""

    line_color: tuple[int, int, int, int] = (255, 0, 0, 180)  # 半透明红色。
    line_width: int = 2
    label_color: tuple[int, int, int, int] = (255, 255, 255, 255)  # 白色。
    font_size: int = 12
    label_padding: int = 5
    num_guides: int = 4
    strict_font_check: bool = False

    def __post_init__(self) -> None:

        if self.num_guides < 1:
            raise ValueError(f"num_guides must be at least 1, got {self.num_guides}")


class AxisGuideGenerator:
    """按切片原始坐标为缩略图生成参考线与刻度。"""

    def __init__(self, style: OverlayStyle | None = None) -> None:

        self.style = style or OverlayStyle()

    def generate(
        self,
        thumbnail_size: tuple[int, int],
        slide_dimensions: tuple[int, int],
    ) -> Image.Image:

        # 校验输入，避免除零。
        if thumbnail_size[0] <= 0 or thumbnail_size[1] <= 0:
            raise ValueError(f"thumbnail_size must be positive, got {thumbnail_size}")
        if slide_dimensions[0] <= 0 or slide_dimensions[1] <= 0:
            raise ValueError(
                f"slide_dimensions must be positive, got {slide_dimensions}"
            )

        thumb_w, thumb_h = thumbnail_size
        slide_w, slide_h = slide_dimensions

        # 创建透明叠加层。
        overlay = Image.new("RGBA", thumbnail_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # 计算坐标转换的缩放系数。

        scale_x = slide_w / thumb_w
        scale_y = slide_h / thumb_h

        # 加载字体；不可用时退回默认字体。
        font = self._get_font()


        step_x = thumb_w / (self.style.num_guides + 1)
        step_y = thumb_h / (self.style.num_guides + 1)

        # 绘制竖线和标签。
        for i in range(1, self.style.num_guides + 1):
            x = int(step_x * i)
            l0_x = int(x * scale_x)

            # 绘制竖线。
            draw.line(
                [(x, 0), (x, thumb_h)],
                fill=self.style.line_color,
                width=self.style.line_width,
            )

            # 在竖线顶部绘制标签。
            label = self._format_coordinate(l0_x)
            self._draw_label(
                draw,
                label,
                (x, self.style.label_padding),
                font,
                anchor="mt",  # 顶部中央。
            )

        # 绘制横线和标签。
        for i in range(1, self.style.num_guides + 1):
            y = int(step_y * i)
            l0_y = int(y * scale_y)

            # 绘制横线。
            draw.line(
                [(0, y), (thumb_w, y)],
                fill=self.style.line_color,
                width=self.style.line_width,
            )

            # 在横线左侧绘制标签。
            label = self._format_coordinate(l0_y)
            self._draw_label(
                draw,
                label,
                (self.style.label_padding, y),
                font,
                anchor="lm",  # 左侧中央。
            )

        return overlay

    def _get_font(self) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:

        try:
            # 尝试加载常见的无衬线字体。
            return ImageFont.truetype("DejaVuSans.ttf", self.style.font_size)
        except OSError:
            try:

                return ImageFont.truetype("Arial.ttf", self.style.font_size)
            except OSError:
                if self.style.strict_font_check:
                    raise RuntimeError(
                        "No TrueType fonts available (DejaVuSans.ttf, Arial.ttf). "
                        "Strict font check is enabled. Install system fonts."
                    ) from None

                logger.warning(
                    "No TrueType fonts available (DejaVuSans.ttf, Arial.ttf). "
                    "Using low-resolution default font. Install fonts for better "
                    "quality."
                )
                return ImageFont.load_default()

    def _format_coordinate(self, coord: int) -> str:

        return str(coord)

    def _draw_label(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        position: tuple[int, int],
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        anchor: str,
    ) -> None:

        # 绘制带轻微阴影的文字，提高可读性。
        x, y = position
        shadow_color = (0, 0, 0, 200)

        # 阴影偏移量。
        for dx, dy in [(1, 1), (-1, -1), (1, -1), (-1, 1)]:
            draw.text(
                (x + dx, y + dy),
                text,
                fill=shadow_color,
                font=font,
                anchor=anchor,
            )

        # 正文文字。
        draw.text(
            position,
            text,
            fill=self.style.label_color,
            font=font,
            anchor=anchor,
        )


class OverlayService:
    """将坐标参考线叠加到缩略图上。"""

    def __init__(
        self,
        generator: AxisGuideGenerator | None = None,
    ) -> None:

        self.generator = generator or AxisGuideGenerator()

    def create_navigable_thumbnail(
        self,
        thumbnail: Image.Image,
        metadata: WSIMetadata,
    ) -> Image.Image:
        """把原始坐标刻度绘制在缩略图上，供导航模型定位。"""
        # 生成叠加层。
        overlay = self.generator.generate(
            thumbnail_size=thumbnail.size,
            slide_dimensions=(metadata.width, metadata.height),
        )


        if thumbnail.mode != "RGBA":
            thumb_rgba = thumbnail.convert("RGBA")
        else:
            thumb_rgba = thumbnail

        # 将叠加层合成到缩略图上。
        composited = Image.alpha_composite(thumb_rgba, overlay)


        return composited.convert("RGB")

    def create_overlay_only(
        self,
        thumbnail_size: tuple[int, int],
        metadata: WSIMetadata,
    ) -> Image.Image:

        return self.generator.generate(
            thumbnail_size=thumbnail_size,
            slide_dimensions=(metadata.width, metadata.height),
        )


def highlight_regions(
    thumbnail: Image.Image,
    regions: list[Region],
    metadata: WSIMetadata,
    *,
    marker_color: tuple[int, int, int] = (0, 255, 0),
    dot_radius: int = 5,
    centers: list[tuple[int, int]] | None = None,
) -> Image.Image:

    img = thumbnail.convert("RGB")
    draw = ImageDraw.Draw(img)
    thumb_w, thumb_h = img.size
    scale_x = thumb_w / metadata.width
    scale_y = thumb_h / metadata.height

    for i, region in enumerate(regions):

        if centers and i < len(centers):
            cx_raw, cy_raw = centers[i]
        else:
            cx_raw = region.x + region.width / 2
            cy_raw = region.y + region.height / 2
        cx = int(cx_raw * scale_x)
        cy = int(cy_raw * scale_y)

        draw.ellipse(
            [(cx - dot_radius, cy - dot_radius), (cx + dot_radius, cy + dot_radius)],
            fill=marker_color,
        )

    return img
