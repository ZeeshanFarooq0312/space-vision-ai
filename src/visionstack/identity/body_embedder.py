"""Phase 3 — full-body ReID embedding, used as a fallback when face data is unavailable, and as
the appearance signal for TrackTrackLocalTracker's association cost (see tracking/local_tracker.py
and tracking/tracktrack/params.py's appearance_weight docstring for why that was 0 until this
existed).
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from visionstack.common.types import BBox

EMBEDDING_DIM = 512


class BodyEmbedder(Protocol):
    def embed(self, person_crop: np.ndarray) -> np.ndarray:
        """Return a 512-d, L2-normalized full-body appearance embedding for one crop."""
        ...

    def embed_batch(self, frame_image: np.ndarray, bboxes: list[BBox]) -> list[np.ndarray]:
        """Batched form of embed() -- one model call for every box in a frame instead of one per
        detection. Real embedders should override this; callers (video_upload.py,
        pipeline/orchestrator.py) always go through this method now, even for the trivial NoOp
        case, so swapping in a real embedder doesn't require touching the call sites.
        """
        ...


class NoOpBodyEmbedder:
    """Stub -- returns an all-zero vector for every crop. Kept around as the default for the
    orchestrator/run_pipeline.py demo path and as a fallback if OSNetBodyEmbedder fails to load.
    """

    def embed(self, person_crop: np.ndarray) -> np.ndarray:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    def embed_batch(self, frame_image: np.ndarray, bboxes: list[BBox]) -> list[np.ndarray]:
        return [self.embed(frame_image) for _ in bboxes]


class OSNetBodyEmbedder:
    """Real full-body ReID embedding via OSNet (osnet_x0_25, MSMT17-trained -- boxmot's default
    checkpoint: small/fast, trained on a large multi-scene person-ReID dataset so it generalizes
    reasonably to an arbitrary office CCTV camera without any fine-tuning of our own), through
    boxmot's `ReidAutoBackend`.

    Pinned to boxmot==10.0.83, not latest -- checked the latest release (22.0.0) by hand first: its
    top-level `ReIDModel` import chain is broken (`ModuleNotFoundError: No module named
    'boxmot.data'`), a bug in that specific release, not a version-skew issue on our side. 10.0.83
    is the last release with the classic, well-documented `ReidAutoBackend(weights, device,
    half).model.get_features(xyxys, img)` API used here and by boxmot's own BoT-SORT/StrongSORT
    trackers internally (see their source for the exact same call shape this class wraps).
    Installing it forces torch 2.13->2.2.2 and torchvision 0.28->0.17.2 (boxmot's own pin) --
    confirmed by hand afterward that YOLO inference and CUDA both still work fine on that pair
    before wiring this in for real.

    `get_features` returns embeddings that are NOT unit-normalized (measured norms ~0.70, not
    1.0) despite the BodyEmbedder Protocol's contract -- normalized here so
    tracktrack.association.cos_distance's `1 - dot(t_feat, d_feat.T)` stays a valid, bounded
    cosine distance.
    """

    DEFAULT_WEIGHTS = "models/osnet_ibn_x1_0_msmt17.pt"

    def __init__(self, weights_path: str = DEFAULT_WEIGHTS, device: str = "auto") -> None:
        from boxmot.appearance.reid_auto_backend import ReidAutoBackend

        weights = Path(weights_path)
        weights.parent.mkdir(parents=True, exist_ok=True)
        backend = ReidAutoBackend(weights=weights, device=self._resolve_device(device), half=False)
        self._model = backend.model

    @staticmethod
    def _resolve_device(device: str) -> str:
        # boxmot's own device convention (a GPU index string like "0", not torch's "cuda") --
        # see boxmot.utils.torch_utils.select_device.
        if device != "auto":
            return device
        import torch

        return "0" if torch.cuda.is_available() else "cpu"

    def embed(self, person_crop: np.ndarray) -> np.ndarray:
        h, w = person_crop.shape[:2]
        return self.embed_batch(person_crop, [BBox(x1=0, y1=0, x2=w, y2=h)])[0]

    def embed_batch(self, frame_image: np.ndarray, bboxes: list[BBox]) -> list[np.ndarray]:
        if not bboxes:
            return []
        xyxys = np.array([[b.x1, b.y1, b.x2, b.y2] for b in bboxes], dtype=np.float64)
        feats = self._model.get_features(xyxys, frame_image)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return list((feats / norms).astype(np.float32))
