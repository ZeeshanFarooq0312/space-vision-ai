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


class IdentityConfig(BaseModel):
    face_match_threshold: float = 0.55
    body_match_threshold: float = 0.45


class AttendanceConfig(BaseModel):
    presence_confirmation_seconds: int = 8
    end_of_day_auto_logout_time: str = "23:59"


class PipelineConfig(BaseModel):
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
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
