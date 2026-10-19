"""封装全切片文件的打开、元数据读取、区域裁剪与缩略图生成。"""

# 区域坐标以切片原始分辨率为准，读取尺寸以所选金字塔层级为准。
# 文件句柄由上下文管理器关闭，元数据读取后缓存以避免重复查询。

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import openslide
from PIL import Image

from giant.wsi.exceptions import WSIOpenError, WSIReadError
from giant.wsi.types import WSIMetadata

if TYPE_CHECKING:
    from types import TracebackType


SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
    ".svs",
    ".ndpi",
        ".tiff",
        ".tif",
    ".mrxs",
    ".vms",
    ".vmu",
    ".scn",
    ".bif",
        ".svslide",
        ".dcm",
    }
)


class WSIReader:
    """管理全切片文件句柄并按指定层级读取图像。"""

    __slots__ = ("_metadata", "_path", "_slide")

    def __init__(self, path: str | Path) -> None:

        self._path = Path(path).resolve()
        self._metadata: WSIMetadata | None = None  # 元数据缓存（首次调用才读）
        self._slide: openslide.OpenSlide | None = None

        # ① 校验文件是否存在
        if not self._path.exists():
            raise WSIOpenError(
                "File not found",
                path=self._path,
            )

        # ② 校验扩展名是否支持
        suffix = self._path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise WSIOpenError(
                f"Unsupported file extension '{suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
                path=self._path,
            )


        try:
            self._slide = openslide.OpenSlide(str(self._path))
        except openslide.OpenSlideError as e:
            raise WSIOpenError(
                f"Failed to open WSI: {e}",
                path=self._path,
            ) from e

    @property
    def path(self) -> Path:

        return self._path


    def _ensure_open(self) -> openslide.OpenSlide:
        slide = self._slide
        if slide is None:
            raise WSIReadError("WSI is closed", path=self._path)
        return slide


    # 首次调用后缓存，之后直接返回（避免重复读）。
    def get_metadata(self) -> WSIMetadata:
        """读取并缓存层级尺寸、降采样系数和像素物理尺寸。"""
        if self._metadata is not None:
            return self._metadata  # 已有缓存，直接返回

        slide = self._ensure_open()


        props = slide.properties
        mpp_x = self._extract_mpp(props, "openslide.mpp-x")
        mpp_y = self._extract_mpp(props, "openslide.mpp-y")


        vendor = props.get("openslide.vendor", "unknown")

        self._metadata = WSIMetadata(
            path=str(self._path),
            width=slide.dimensions[0],
            height=slide.dimensions[1],
            level_count=slide.level_count,
            level_dimensions=tuple(slide.level_dimensions),
            level_downsamples=tuple(slide.level_downsamples),
            vendor=vendor,
            mpp_x=mpp_x,
            mpp_y=mpp_y,
        )
        return self._metadata


    @staticmethod
    def _extract_mpp(
        props: Mapping[str, str],
        key: str,
    ) -> float | None:

        value = props.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None


    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> Image.Image:
        """校验坐标和层级后，从切片读取指定区域并转为彩色图像。"""
        slide = self._ensure_open()


        metadata = self.get_metadata()
        if level < 0 or level >= metadata.level_count:
            raise WSIReadError(
                f"Invalid level {level}. "
                f"Must be in range [0, {metadata.level_count - 1}]",
                path=self._path,
                level=level,
                location=location,
                size=size,
            )

        # ② 校验尺寸为正
        if size[0] <= 0 or size[1] <= 0:
            raise WSIReadError(
                f"Invalid size {size}. Width and height must be positive.",
                path=self._path,
                level=level,
                location=location,
                size=size,
            )

        # ③ 校验坐标非负
        if location[0] < 0 or location[1] < 0:
            raise WSIReadError(
                f"Invalid location {location}. Coordinates must be non-negative.",
                path=self._path,
                level=level,
                location=location,
                size=size,
            )

        try:

            rgba_image = slide.read_region(location, level, size)
            return rgba_image.convert("RGB")
        except Exception as e:


            raise WSIReadError(
                f"Failed to read region: {e}",
                path=self._path,
                level=level,
                location=location,
                size=size,
            ) from e


    def get_thumbnail(self, max_size: tuple[int, int]) -> Image.Image:

        slide = self._ensure_open()

        if max_size[0] <= 0 or max_size[1] <= 0:
            raise WSIReadError(
                f"Invalid max_size {max_size}. Dimensions must be positive.",
                path=self._path,
            )

        try:

            thumbnail = slide.get_thumbnail(max_size)
            return thumbnail.convert("RGB")
        except Exception as e:
            raise WSIReadError(
                f"Failed to generate thumbnail: {e}",
                path=self._path,
            ) from e


    def get_best_level_for_downsample(self, downsample: float) -> int:

        slide = self._ensure_open()
        try:
            return slide.get_best_level_for_downsample(downsample)
        except Exception as e:
            raise WSIReadError(
                f"Failed to get best level for downsample: {e}",
                path=self._path,
            ) from e


    def close(self) -> None:

        if self._slide is None:
            return
        self._slide.close()
        self._slide = None


    def __enter__(self) -> WSIReader:

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:

        self.close()

    def __repr__(self) -> str:

        return f"WSIReader(path={self._path!r})"
