"""Config loading: secrets from .env via pydantic-settings, structural config from YAML."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from visionstack.common.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = REPO_ROOT / "configs"


class Settings(BaseSettings):
    """Secrets and environment-specific values, loaded from .env / real env vars."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://visionstack:visionstack@localhost:5432/visionstack"
    log_level: str = "INFO"
    # External face-embedding service (onboard/verify/compare/embeddings) — see
    # visionstack.identity.face_api_client. No local RetinaFace/ArcFace model yet (see README).
    face_api_base_url: str = "http://182.180.87.19:3009"
    # YOLO weights used by the live-capture and video-upload pipelines (api/live_stream.py,
    # api/video_upload.py). Defaults to the stock COCO model; point this at a fine-tuned
    # checkpoint (e.g. runs/detect/person_train/weights/best.pt, copied into ./models so it's
    # visible inside the container) to use it instead.
    detection_weights_path: str = "models/yolov8n.pt"


class VideoSourceConfig(BaseModel):
    type: Literal["rtsp", "file"]
    uri: str


class CameraConfig(BaseModel):
    camera_id: str
    role: Literal["entry", "zone", "exit"]
    source: VideoSourceConfig
    sample_fps: float = 5.0
    resolution: tuple[int, int] | None = None
    enabled: bool = True


class CamerasConfig(BaseModel):
    cameras: list[CameraConfig] = Field(default_factory=list)


class ZoneConfig(BaseModel):
    zone_id: str
    camera_id: str
    name: str
    zone_type: Literal["allowed", "restricted", "exit"]
    polygon: list[tuple[float, float]]
    allowed_roles: list[str] = Field(default_factory=list)


class ZonesConfig(BaseModel):
    zones: list[ZoneConfig] = Field(default_factory=list)


class DetectionConfig(BaseModel):
    weights: str = "models/yolov8n.pt"
    device: str = "auto"
    conf_threshold: float = 0.5
    iou_threshold: float = 0.45


class TrackingConfig(BaseModel):
    # "passthrough" (fresh id per detection, no association) or "tracktrack" (vendored
    # Kalman+ReID tracker, see tracking/tracktrack/ and tracking/local_tracker.py's module
    # docstring for why it's not the default on a non-fixed camera).
    tracker: Literal["passthrough", "tracktrack"] = "passthrough"
    det_thr: float = 0.6
    init_thr: float = 0.6
    match_thr: float = 0.7
    max_time_lost_seconds: float = 3.0


class IdentityConfig(BaseModel):
    face_match_threshold: float = 0.55
    body_match_threshold: float = 0.45


class AttendanceConfig(BaseModel):
    presence_confirmation_seconds: int = 8
    end_of_day_auto_logout_time: str = "23:59"


class PipelineConfig(BaseModel):
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    attendance: AttendanceConfig = Field(default_factory=AttendanceConfig)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_cameras(path: Path | None = None) -> CamerasConfig:
    return CamerasConfig.model_validate(_load_yaml(path or CONFIGS_DIR / "cameras.yaml"))


def load_zones(path: Path | None = None) -> ZonesConfig:
    return ZonesConfig.model_validate(_load_yaml(path or CONFIGS_DIR / "zones.yaml"))


def load_pipeline_config(path: Path | None = None) -> PipelineConfig:
    return PipelineConfig.model_validate(_load_yaml(path or CONFIGS_DIR / "pipeline.yaml"))


def get_settings() -> Settings:
    return Settings()
