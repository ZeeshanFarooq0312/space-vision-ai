"""Phase 3 — face embedding. TODO: implement ArcFace with a GhostNet backbone."""
from __future__ import annotations

from typing import Protocol

import numpy as np

EMBEDDING_DIM = 512


class FaceEmbedder(Protocol):
    def embed(self, aligned_face: np.ndarray) -> np.ndarray:
        """Return a 512-d, L2-normalized embedding for an aligned face crop."""
        ...


class NoOpFaceEmbedder:
    """TODO: implement ArcFace (GhostNet backbone) embedding extraction."""

    def embed(self, aligned_face: np.ndarray) -> np.ndarray:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
