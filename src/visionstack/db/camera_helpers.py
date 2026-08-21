"""Small shared helper for FK-constrained tables (zones, attendance_events, tracks) that
reference `cameras.camera_id`.

Live-session camera_ids (e.g. "webcam-0", chosen client-side in frontend/src/pages/Live.tsx) are
never registered via configs/cameras.yaml or a cameras API -- see api/live_stream.py's module
docstring and api/routers/cameras.py, which only reads the table, never writes it. Saving a zone
or an attendance event against one of these ids would otherwise fail the foreign key.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from visionstack.db.models import Camera


def ensure_camera(db: Session, camera_id: str, role: str = "zone") -> None:
    """Insert a minimal Camera row for `camera_id` if one doesn't already exist. Does not commit
    -- caller is expected to be inside its own transaction."""
    if db.get(Camera, camera_id) is not None:
        return
    db.add(Camera(camera_id=camera_id, name=camera_id, role=role, active=True))
