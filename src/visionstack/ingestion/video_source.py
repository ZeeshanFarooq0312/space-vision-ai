"""Phase 1 — video/RTSP ingestion. Wraps cv2.VideoCapture for both file and RTSP sources."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Iterator

import cv2

from visionstack.common.errors import VideoSourceError
from visionstack.common.types import Frame


class VideoSource(ABC):
    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self._cap: cv2.VideoCapture | None = None
        self._frame_id = 0

    @abstractmethod
    def _open_capture(self) -> cv2.VideoCapture: ...

    def open(self) -> None:
        self._cap = self._open_capture()
        if not self._cap.isOpened():
            raise VideoSourceError(f"Failed to open video source for camera '{self.camera_id}'")

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float:
        if self._cap is None:
            raise VideoSourceError("Video source is not open")
        return self._cap.get(cv2.CAP_PROP_FPS) or 0.0

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[Frame]:
        if self._cap is None:
            raise VideoSourceError("Video source is not open — call open() or use as a context manager")
        while True:
            ok, image = self._cap.read()
            if not ok:
                return
            frame = Frame(
                camera_id=self.camera_id,
                frame_id=self._frame_id,
                timestamp=datetime.now(timezone.utc),
                image=image,
            )
            self._frame_id += 1
            yield frame


class FileVideoSource(VideoSource):
    def __init__(self, path: str, camera_id: str) -> None:
        super().__init__(camera_id)
        self._path = path

    def _open_capture(self) -> cv2.VideoCapture:
        return cv2.VideoCapture(self._path)


class RTSPVideoSource(VideoSource):
    def __init__(self, url: str, camera_id: str) -> None:
        super().__init__(camera_id)
        self._url = url

    def _open_capture(self) -> cv2.VideoCapture:
        return cv2.VideoCapture(self._url)


def video_source_from_config(camera_id: str, source_type: str, uri: str) -> VideoSource:
    if source_type == "file":
        return FileVideoSource(path=uri, camera_id=camera_id)
    if source_type == "rtsp":
        return RTSPVideoSource(url=uri, camera_id=camera_id)
    raise VideoSourceError(f"Unknown video source type '{source_type}'")
