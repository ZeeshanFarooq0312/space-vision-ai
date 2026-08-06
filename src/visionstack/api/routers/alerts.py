from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from visionstack.api.deps import get_db
from visionstack.api.schemas import AlertRead
from visionstack.db.models import Alert

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("", response_model=list[AlertRead])
def list_alerts(status: str | None = None, db: Session = Depends(get_db)) -> list[Alert]:
    query = db.query(Alert)
    if status is not None:
        query = query.filter(Alert.status == status)
    return query.order_by(Alert.created_at.desc()).all()
