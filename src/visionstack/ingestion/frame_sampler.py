"""Drops frames from a VideoSource iterator to approximate a configured target FPS."""
from __future__ import annotations

from typing import Iterable, Iterator

from visionstack.common.types import Frame


class FrameSampler:
    def __init__(self, frames: Iterable[Frame], source_fps: float, sample_fps: float) -> None:
        self._frames = frames
        # If the source FPS is unknown/zero (common for some RTSP streams), don't drop anything.
        self._stride = max(1, round(source_fps / sample_fps)) if source_fps and sample_fps else 1

    def __iter__(self) -> Iterator[Frame]:
        for i, frame in enumerate(self._frames):
            if i % self._stride == 0:
                yield frame
