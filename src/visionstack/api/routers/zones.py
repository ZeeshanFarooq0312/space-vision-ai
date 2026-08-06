from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from visionstack.api.deps import get_db
from visionstack.api.schemas import ZoneRead
from visionstack.db.models import Zone

router = APIRouter(prefix="/zones", tags=["zones"])


@router.get("", response_model=list[ZoneRead])
def list_zones(db: Session = Depends(get_db)) -> list[Zone]:
    return db.query(Zone).all()
