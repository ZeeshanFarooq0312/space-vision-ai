from visionstack.common.config import load_cameras, load_pipeline_config, load_zones


def test_load_cameras_from_repo_config():
    cfg = load_cameras()
    assert len(cfg.cameras) >= 1
    ids = {c.camera_id for c in cfg.cameras}
    assert "entry-cam-1" in ids


def test_load_zones_from_repo_config():
    cfg = load_zones()
    assert len(cfg.zones) >= 1
    server_room = next(z for z in cfg.zones if z.zone_id == "server-room")
    assert server_room.zone_type == "restricted"
    assert len(server_room.polygon) >= 3


def test_load_pipeline_config_defaults_present():
    cfg = load_pipeline_config()
    assert cfg.detection.weights == "models/yolov8n.pt"
    assert 0 < cfg.identity.face_match_threshold <= 1
    assert cfg.attendance.presence_confirmation_seconds > 0
