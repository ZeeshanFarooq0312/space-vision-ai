import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from visionstack.api.deps import get_db
from visionstack.api.schemas import ZoneCreate, ZoneRead
from visionstack.db.camera_helpers import ensure_camera
from visionstack.db.models import Zone, ZoneRoleAccess

router = APIRouter(prefix="/zones", tags=["zones"])


@router.get("", response_model=list[ZoneRead])
def list_zones(db: Session = Depends(get_db)) -> list[Zone]:
    return db.query(Zone).all()


@router.post("", response_model=ZoneRead, status_code=201)
def create_zone(payload: ZoneCreate, db: Session = Depends(get_db)) -> Zone:
    """Saves a zone drawn on the live preview (see frontend/src/components/ZoneDrawer.tsx) or
    created any other way. `camera_id` doesn't need to pre-exist in the `cameras` table -- a
    minimal row is upserted for it (see db/camera_helpers.ensure_camera), since live-session
    camera_ids are ad-hoc client-chosen strings, not pre-registered cameras.
    """
    zone_id = payload.zone_id or uuid.uuid4().hex
    ensure_camera(db, payload.camera_id)

    zone = Zone(
        zone_id=zone_id,
        camera_id=payload.camera_id,
        name=payload.name,
        zone_type=payload.zone_type,
        polygon=[list(point) for point in payload.polygon],
        triggers_login=payload.triggers_login,
    )
    db.add(zone)
    db.flush()
    for role in payload.allowed_roles:
        db.add(ZoneRoleAccess(zone_id=zone_id, role=role))

    db.commit()
    db.refresh(zone)
    return zone


@router.delete("/{zone_id}", status_code=204)
def delete_zone(zone_id: str, db: Session = Depends(get_db)) -> None:
    zone = db.get(Zone, zone_id)
    if zone is None:
        raise HTTPException(status_code=404, detail="Zone not found")

    # No ondelete=CASCADE on this FK, so clear role-access rows explicitly first -- same pattern
    # as employees.delete_employee. zone_events isn't written to yet (see zones/monitor.py), so
    # that FK is never actually populated today; the IntegrityError catch below is just a safety
    # net if that changes rather than something expected to fire in practice.
    db.query(ZoneRoleAccess).filter(ZoneRoleAccess.zone_id == zone_id).delete()
    db.delete(zone)
    try:
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            status_code=409, detail="Cannot delete: zone still has related records (zone events)."
        ) from e
