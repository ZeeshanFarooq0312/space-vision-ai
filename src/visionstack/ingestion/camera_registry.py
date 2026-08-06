"""Loads configs/cameras.yaml into typed CameraConfig objects."""
from __future__ import annotations

from pathlib import Path

from visionstack.common.config import CameraConfig, load_cameras


class CameraRegistry:
    def __init__(self, path: Path | None = None) -> None:
        self._cameras: dict[str, CameraConfig] = {
            c.camera_id: c for c in load_cameras(path).cameras
        }

    def get(self, camera_id: str) -> CameraConfig:
        if camera_id not in self._cameras:
            raise KeyError(f"Unknown camera_id '{camera_id}'")
        return self._cameras[camera_id]

    def all(self) -> list[CameraConfig]:
        return list(self._cameras.values())

    def enabled(self) -> list[CameraConfig]:
        return [c for c in self._cameras.values() if c.enabled]
