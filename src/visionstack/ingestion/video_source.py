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
            # Release before raising: if open() is called via `with source:` (the normal path,
            # see Pipeline.run()), raising here happens inside __enter__, and Python's context
            # manager protocol never calls __exit__/close() when __enter__ itself raises. Left
            # unreleased, a single failed open leaks the OS-level device claim -- observed in
            # practice as a real webcam permanently EBUSY for every later attempt until the
            # process holding the stale handle was restarted.
            self._cap.release()
            self._cap = None
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


class WebcamVideoSource(VideoSource):
    """Local webcam via OpenCV device index (e.g. 0 for the default camera).

    Requests 1920x1080 by default rather than whatever the driver's own default is (measured at
    640x480 on this hardware) -- the external face-verify API quality-gates on how much of the
    frame the face fills ("Move closer."), and at 480p a person standing at a normal camera
    distance is nowhere near large/sharp enough to pass; 1080p was the minimum tested that
    reliably cleared it (720p still failed). cv2 falls back to the nearest supported mode if a
    device can't do the requested resolution, so this degrades gracefully on lower-end cameras.
    """

    def __init__(self, device_index: int, camera_id: str, width: int = 1920, height: int = 1080) -> None:
        super().__init__(camera_id)
        self._device_index = device_index
        self._width = width
        self._height = height

    def _open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._device_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        return cap

    @property
    def fps(self) -> float:
        # Some webcams/drivers report 0 for CAP_PROP_FPS. FrameSampler treats a falsy
        # source_fps as "unknown" and disables downsampling, so fall back to a sane
        # default rather than silently ignoring --sample-fps.
        return super().fps or 30.0


def video_source_from_config(camera_id: str, source_type: str, uri: str) -> VideoSource:
    if source_type == "file":
        return FileVideoSource(path=uri, camera_id=camera_id)
    if source_type == "rtsp":
        return RTSPVideoSource(url=uri, camera_id=camera_id)
    if source_type == "webcam":
        try:
            device_index = int(uri)
        except ValueError as e:
            raise VideoSourceError(
                f"webcam source expects an integer device index, got '{uri}'"
            ) from e
        return WebcamVideoSource(device_index=device_index, camera_id=camera_id)
    raise VideoSourceError(f"Unknown video source type '{source_type}'")
