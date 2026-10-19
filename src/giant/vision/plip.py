"""用病理图文模型对缩略图滑窗评分，筛出不重叠的候选区域。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from giant.geometry.primitives import Region
from giant.vision.segmentation import TissueSegmentor
from giant.wsi.reader import WSIReader
from giant.wsi.types import WSIMetadata

# =============================================================================

# 这里准备两组文本：一组描述肿瘤/恶性，一组描述正常/良性。
# 最终分数 = 肿瘤相似度 - 正常相似度，越高越可疑。
# =============================================================================
_TUMOR_PROMPTS = (
    "a histopathology image of malignant tumor",
    "atypical cancer cells",
)
_NORMAL_PROMPTS = (
    "a histopathology image of normal tissue",
    "benign tissue",
)


@dataclass(frozen=True)
class CandidateRegion:
    """保存候选区域的位置和相关性分数。"""

    region: Region
    score: float
    center: tuple[int, int]


class PlipCandidateScorer:
    """按检索文本对缩略图区域评分并去除重叠候选。"""

    def __init__(
        self,
        *,
        model_name: str = "vinid/plip",
        window_size: int = 224,
        stride: int = 112,
        min_tissue_frac: float = 0.8,
        device: str | None = None,
    ) -> None:

        self._model_name = model_name
        self._window_size = window_size
        self._stride = stride
        self._min_tissue_frac = min_tissue_frac
        self._device = device

        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None

    def _ensure_loaded(self) -> None:

        if self._model is not None:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(
                "PLIP pre-screening requires the optional 'plip' dependencies "
                "(torch + transformers). Install with: uv sync --extra plip"
            ) from exc
        self._torch = torch
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = (
            CLIPModel.from_pretrained(
                self._model_name, local_files_only=True
            ).to(self._device).eval()
        )
        self._processor = CLIPProcessor.from_pretrained(
            self._model_name, local_files_only=True
        )

    def score_thumbnail(
        self,
        thumbnail: Image.Image,
        metadata: WSIMetadata,
        k: int = 5,
        dimensions: Sequence[tuple[str, float]] | None = None,
    ) -> list[CandidateRegion]:
        """对组织区域滑窗提取图文相似度并返回候选。"""
        self._ensure_loaded()


        mask = TissueSegmentor().segment(thumbnail)
        thumb_w, thumb_h = thumbnail.size


        patches: list[Image.Image] = []
        anchors: list[tuple[int, int]] = []
        for y in range(0, thumb_h - self._window_size + 1, self._stride):
            for x in range(0, thumb_w - self._window_size + 1, self._stride):
                window = mask[y : y + self._window_size, x : x + self._window_size]
                if window.mean() < self._min_tissue_frac:
                    continue
                patches.append(
                    thumbnail.crop(
                        (x, y, x + self._window_size, y + self._window_size)
                    )
                )
                anchors.append((x, y))

        if not patches:
            return []


        if dimensions:
            texts = [t for t, _ in dimensions]
            weights = [w for _, w in dimensions]
            matrix = self._similarity_matrix(patches, texts)  # （图像块数量，特征维度数）
            mat = matrix.cpu().tolist()
            scores = [
                sum(weights[i] * mat[j][i] for i in range(len(weights)))
                for j in range(len(patches))
            ]
            nms_candidates = self._nms_topk(anchors, scores, k)
        else:
            scores = self._score_patches(patches)
            nms_candidates = self._nms_topk(anchors, scores, k)


        l0_scale_x = metadata.width / thumb_w
        l0_scale_y = metadata.height / thumb_h

        candidates: list[CandidateRegion] = []
        for idx, score in nms_candidates:
            x, y = anchors[idx]
            cx = x + self._window_size // 2
            cy = y + self._window_size // 2


            center_x = int(cx * l0_scale_x)
            center_y = int(cy * l0_scale_y)

            candidates.append(
                CandidateRegion(
                    region=self._anchor_to_level0((x, y), metadata, thumb_w, thumb_h),
                    score=score,
                    center=(center_x, center_y),
                )
            )
            if len(candidates) >= k:
                break

        return candidates

    def _similarity_matrix(self, patches: list[Image.Image], texts: list[str]) -> Any:

        inputs = self._processor(
            text=list(texts),
            images=patches,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.no_grad():
            return self._model(**inputs).logits_per_image  # （图像块数量，文本数量）

    def _score_patches(self, patches: list[Image.Image]) -> list[float]:

        prompts = list(_TUMOR_PROMPTS) + list(_NORMAL_PROMPTS)
        n_tumor = len(_TUMOR_PROMPTS)
        logits = self._similarity_matrix(patches, prompts)
        tumor = logits[:, :n_tumor].mean(dim=1)
        normal = logits[:, n_tumor:].mean(dim=1)
        return (tumor - normal).cpu().tolist()

    def _score_patches_by_query(
        self,
        patches: list[Image.Image],
        query_texts: list[str],
    ) -> list[float]:

        logits = self._similarity_matrix(patches, list(query_texts))
        return logits.max(dim=1).values.cpu().tolist()

    def _nms_topk(
        self,
        anchors: list[tuple[int, int]],
        scores: list[float],
        k: int,
    ) -> list[tuple[int, float]]:

        nms_radius = int(self._window_size * 1.5)  # 336 像素。
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        kept_centers: list[tuple[int, int]] = []
        nms_candidates: list[tuple[int, float]] = []
        for idx in order:
            x, y = anchors[idx]
            cx = x + self._window_size // 2
            cy = y + self._window_size // 2
            if any(
                (cx - px) ** 2 + (cy - py) ** 2 < nms_radius**2
                for px, py in kept_centers
            ):
                continue
            kept_centers.append((cx, cy))
            nms_candidates.append((idx, scores[idx]))
            if len(nms_candidates) >= k:
                break
        return nms_candidates

    def _anchor_to_level0(
        self,
        anchor: tuple[int, int],
        metadata: WSIMetadata,
        thumb_w: int,
        thumb_h: int,
    ) -> Region:

        x, y = anchor

        scale_x = metadata.width / thumb_w
        scale_y = metadata.height / thumb_h


        center_x = int((x + self._window_size / 2) * scale_x)
        center_y = int((y + self._window_size / 2) * scale_y)


        CROP_SIZE_L0 = 2048
        l0_w = CROP_SIZE_L0
        l0_h = CROP_SIZE_L0
        l0_x = center_x - CROP_SIZE_L0 // 2
        l0_y = center_y - CROP_SIZE_L0 // 2

        # 边界保护：防止中心靠近边缘时出界。
        l0_x = max(0, min(l0_x, metadata.width - l0_w))
        l0_y = max(0, min(l0_y, metadata.height - l0_h))
        return Region(x=l0_x, y=l0_y, width=l0_w, height=l0_h)


def prescreen_candidates(
    wsi_path: str | Path,
    *,
    k: int = 5,
    thumbnail_size: int = 2048,
    dimensions: Sequence[tuple[str, float]] | None = None,
    **scorer_kwargs: Any,
) -> tuple[list[Region], list[tuple[int, int]]]:
    """按问题检索计划筛出供导航模型参考的候选区域。"""
    with WSIReader(wsi_path) as reader:
        thumbnail = reader.get_thumbnail((thumbnail_size, thumbnail_size))
        metadata = reader.get_metadata()
    scorer = PlipCandidateScorer(**scorer_kwargs)
    candidates = scorer.score_thumbnail(
        thumbnail, metadata, k=k, dimensions=dimensions
    )
    return (
        [c.region for c in candidates],
        [c.center for c in candidates],
    )
