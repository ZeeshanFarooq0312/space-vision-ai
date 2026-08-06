import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from visionstack.api.deps import get_db
from visionstack.api.schemas import AttendanceEventRead
from visionstack.db.models import AttendanceEvent

router = APIRouter(prefix="/attendance", tags=["attendance"])


@router.get("", response_model=list[AttendanceEventRead])
def list_attendance_events(
    employee_id: uuid.UUID | None = None, db: Session = Depends(get_db)
) -> list[AttendanceEvent]:
    query = db.query(AttendanceEvent)
    if employee_id is not None:
        query = query.filter(AttendanceEvent.employee_id == employee_id)
    return query.order_by(AttendanceEvent.occurred_at.desc()).all()
