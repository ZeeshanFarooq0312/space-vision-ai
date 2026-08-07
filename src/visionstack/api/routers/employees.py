import shutil
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from visionstack.api.deps import get_db
from visionstack.api.schemas import EmployeeCreate, EmployeeRead
from visionstack.common.config import REPO_ROOT
from visionstack.db.models import Employee

router = APIRouter(prefix="/employees", tags=["employees"])

# Raw enrollment photos, captured during onboarding for later use by the (not yet implemented)
# face/body embedding pipeline — see identity/ TODOs. No detection or embedding runs on these
# yet; they're just stored so onboarding doesn't have to be redone once that phase lands.
ENROLLMENT_PHOTOS_DIR = REPO_ROOT / "data" / "employee_photos"
EnrollmentPose = Literal["straight", "left", "right"]


@router.get("", response_model=list[EmployeeRead])
def list_employees(db: Session = Depends(get_db)) -> list[Employee]:
    return db.query(Employee).all()


@router.post("", response_model=EmployeeRead, status_code=201)
def create_employee(payload: EmployeeCreate, db: Session = Depends(get_db)) -> Employee:
    employee = Employee(**payload.model_dump())
    db.add(employee)
    try:
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            status_code=409, detail=f"employee_code '{payload.employee_code}' already exists"
        ) from e
    db.refresh(employee)
    return employee


@router.get("/{employee_id}", response_model=EmployeeRead)
def get_employee(employee_id: uuid.UUID, db: Session = Depends(get_db)) -> Employee:
    employee = db.get(Employee, employee_id)
    if employee is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    return employee


@router.put("/{employee_id}/photos/{pose}", status_code=204)
def upload_enrollment_photo(
    employee_id: uuid.UUID,
    pose: EnrollmentPose,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> None:
    if db.get(Employee, employee_id) is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail=f"expected an image, got '{file.content_type}'")

    employee_dir = ENROLLMENT_PHOTOS_DIR / str(employee_id)
    employee_dir.mkdir(parents=True, exist_ok=True)
    with (employee_dir / f"{pose}.jpg").open("wb") as dest:
        shutil.copyfileobj(file.file, dest)


@router.get("/{employee_id}/photos", response_model=list[EnrollmentPose])
def list_enrollment_photos(employee_id: uuid.UUID, db: Session = Depends(get_db)) -> list[str]:
    if db.get(Employee, employee_id) is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    employee_dir = ENROLLMENT_PHOTOS_DIR / str(employee_id)
    if not employee_dir.exists():
        return []
    return sorted(p.stem for p in employee_dir.glob("*.jpg"))
