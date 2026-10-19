"""按原始切片坐标裁剪图像，选择读取层级并编码模型输入。"""

# 裁剪区域始终使用原始切片坐标；读取层级由目标尺寸决定。
# 裁剪后保持宽高比，并编码为模型消息可直接使用的图像。

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO

from PIL import Image

from giant.config import settings
from giant.core.level_selector import LevelSelectorProtocol, PyramidLevelSelector
from giant.geometry import Region
from giant.wsi.types import WSIReaderProtocol, size_at_level


_JPEG_QUALITY_MIN = 1
_JPEG_QUALITY_MAX = 100


_DEFAULT_MAX_READ_DIMENSION = 10000

# 安全上限：单次读区域的最大像素面积（防止单层切片上出现超大分配）


_DEFAULT_MAX_READ_PIXELS = 40_000_000


# 存放处理好的图片 + 抠图过程的元数据，便于调试和缓存。
@dataclass(frozen=True)
class CroppedImage:
    """保存裁剪图像、编码结果和实际使用的读取层级。"""

    image: Image.Image
    base64_content: str
    original_region: Region
    read_level: int  # 实际从哪个金字塔层级读的
    scale_factor: float  # 最终尺寸 / 读取尺寸 的比例


class CropEngine:
    """将原始坐标区域读取、缩放并编码为模型可用的图像。"""

    __slots__ = ("_level_selector", "_reader")

    def __init__(
        self,
        reader: WSIReaderProtocol,
        level_selector: LevelSelectorProtocol | None = None,
    ) -> None:

        self._reader = reader  # 切片读取器

        self._level_selector = level_selector or PyramidLevelSelector()

    def crop(
        self,
        region: Region,
        target_size: int = 1000,
        bias: float = 0.85,
        jpeg_quality: int | None = None,
        max_read_dimension: int | None = None,
    ) -> CroppedImage:
        """校验区域、选读取层级，再缩放并编码裁剪图像。"""
        if jpeg_quality is None:
            jpeg_quality = settings.JPEG_QUALITY  # 没传就用配置里的默认值（通常 85）

        if not _JPEG_QUALITY_MIN <= jpeg_quality <= _JPEG_QUALITY_MAX:
            raise ValueError(
                f"jpeg_quality must be {_JPEG_QUALITY_MIN}-{_JPEG_QUALITY_MAX}, "
                f"got {jpeg_quality}"
            )

        # 第①步：读元数据 + 选最优层级
        metadata = self._reader.get_metadata()
        selected = self._level_selector.select_level(
            region, metadata, target_size=target_size, bias=bias
        )

        # 第②步：算出该层级下区域的尺寸，再读图

        region_size_at_level = size_at_level(
            (region.width, region.height), selected.downsample
        )

        # 内存安全：拒绝会分配过多内存的区域。

        effective_max = (
            _DEFAULT_MAX_READ_DIMENSION
            if max_read_dimension is None
            else max_read_dimension
        )
        # 按边长上限等比缩放像素面积上限
        effective_max_pixels = (
            0
            if effective_max <= 0
            else (effective_max * effective_max * _DEFAULT_MAX_READ_PIXELS)
            // (_DEFAULT_MAX_READ_DIMENSION * _DEFAULT_MAX_READ_DIMENSION)
        )
        max_dim = max(region_size_at_level)
        # 检查 ①：边长是否超限
        if effective_max > 0 and max_dim > effective_max:
            w, h = region_size_at_level
            raise ValueError(
                f"Region too large: {w}x{h} pixels at level {selected.level} "
                f"exceeds maximum dimension {effective_max}px. "
                f"Use a smaller region or get_thumbnail() for full-slide overview."
            )
        # 检查 ②：像素面积是否超限
        if effective_max_pixels > 0:
            w, h = region_size_at_level
            pixels = w * h
            if pixels > effective_max_pixels:
                raise ValueError(
                    f"Region too large: {w}x{h} pixels at level {selected.level} "
                    f"({pixels} pixels) exceeds maximum pixels {effective_max_pixels}. "
                    "Use a smaller region or get_thumbnail() for full-slide overview."
                )


        raw_image = self._reader.read_region(
            location=(region.x, region.y),
            level=selected.level,
            size=region_size_at_level,
        )

        # 第③步：缩放到目标尺寸（绝不放大）
        resized_image, scale_factor = self._resize_to_target(raw_image, target_size)


        base64_content = self._encode_base64_jpeg(resized_image, jpeg_quality)

        return CroppedImage(
            image=resized_image,
            base64_content=base64_content,
            original_region=region,
            read_level=selected.level,
            scale_factor=scale_factor,
        )

    def _resize_to_target(
        self,
        image: Image.Image,
        target_size: int,
    ) -> tuple[Image.Image, float]:

        original_width, original_height = image.size
        original_long_side = max(original_width, original_height)

        # 绝不放大：如果图已经比目标尺寸小，直接原样返回
        if original_long_side <= target_size:
            return image, 1.0

        # 计算缩放比例
        scale_factor = target_size / original_long_side


        # （这样能避免浮点截断误差）
        if original_width >= original_height:
            # 宽是长边
            new_width = target_size
            new_height = max(1, round(original_height * scale_factor))
        else:
            # 高是长边
            new_height = target_size
            new_width = max(1, round(original_width * scale_factor))


        resized = image.resize(
            (new_width, new_height),
            resample=Image.Resampling.LANCZOS,
        )

        return resized, scale_factor

    def _encode_base64_jpeg(
        self,
        image: Image.Image,
        quality: int,
    ) -> str:

        buffer = BytesIO()  # 内存缓冲区，避免写临时文件
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)  # 把读写指针拨回开头

        return base64.b64encode(buffer.getvalue()).decode("ascii")
