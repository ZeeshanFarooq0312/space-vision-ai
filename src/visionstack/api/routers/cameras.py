from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from visionstack.api.deps import get_db
from visionstack.api.schemas import CameraRead
from visionstack.db.models import Camera

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.get("", response_model=list[CameraRead])
def list_cameras(db: Session = Depends(get_db)) -> list[Camera]:
    return db.query(Camera).all()
