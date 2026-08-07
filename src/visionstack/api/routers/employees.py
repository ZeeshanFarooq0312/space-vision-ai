import shutil
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from visionstack.api.deps import get_db
from visionstack.api.schemas import (
    EmployeeCreate,
    EmployeeRead,
    FaceEnrollmentResult,
    FaceEnrollmentStatus,
)
from visionstack.common.config import REPO_ROOT
from visionstack.common.errors import FaceApiError
from visionstack.db.models import Employee, EmployeeBodyEmbedding, EmployeeFaceEmbedding
from visionstack.db.vector import FACE_EMBEDDING_DIM
from visionstack.identity import face_api_client

router = APIRouter(prefix="/employees", tags=["employees"])

# Raw enrollment photos, kept alongside the external face API's response for audit/debugging.
# Not used to compute embeddings ourselves -- see face_api_client.py.
ENROLLMENT_PHOTOS_DIR = REPO_ROOT / "data" / "employee_photos"
EnrollmentPose = Literal["front", "left", "right"]


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


@router.delete("/{employee_id}", status_code=204)
def delete_employee(employee_id: uuid.UUID, db: Session = Depends(get_db)) -> None:
    employee = db.get(Employee, employee_id)
    if employee is None:
        raise HTTPException(status_code=404, detail="Employee not found")

    # Embeddings have no ondelete=CASCADE on their FK, so clear them explicitly first. Other
    # FK'd tables (attendance_events, tracks, ...) are still stub-phase and expected to be empty;
    # if one does reference this employee, the final delete below surfaces that as a 409 instead
    # of a raw 500.
    db.query(EmployeeFaceEmbedding).filter(EmployeeFaceEmbedding.employee_id == employee_id).delete()
    db.query(EmployeeBodyEmbedding).filter(EmployeeBodyEmbedding.employee_id == employee_id).delete()
    db.delete(employee)
    try:
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="Cannot delete: employee still has related records (attendance/zone events, tracks).",
        ) from e

    employee_dir = ENROLLMENT_PHOTOS_DIR / str(employee_id)
    shutil.rmtree(employee_dir, ignore_errors=True)


def _read_image(file: UploadFile) -> bytes:
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail=f"expected an image, got '{file.content_type}'")
    return file.file.read()


@router.post("/{employee_id}/enroll-face", response_model=FaceEnrollmentResult)
def enroll_face(
    employee_id: uuid.UUID,
    front: UploadFile = File(..., description="Front-facing face photo"),
    right: UploadFile = File(..., description="Right-turn face photo"),
    left: UploadFile = File(..., description="Left-turn face photo"),
    db: Session = Depends(get_db),
) -> dict:
    """Submits the 3 enrollment photos to the external face API, stores the embedding it returns
    against this employee, and keeps a local copy of the raw photos for audit/debugging."""
    if db.get(Employee, employee_id) is None:
        raise HTTPException(status_code=404, detail="Employee not found")

    photos = {"front": _read_image(front), "right": _read_image(right), "left": _read_image(left)}

    employee_dir = ENROLLMENT_PHOTOS_DIR / str(employee_id)
    employee_dir.mkdir(parents=True, exist_ok=True)
    for pose, data in photos.items():
        (employee_dir / f"{pose}.jpg").write_bytes(data)

    try:
        face_id = face_api_client.onboard_face(**photos)
        embedding = face_api_client.get_embedding(face_id)
    except FaceApiError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    if len(embedding) != FACE_EMBEDDING_DIM:
        raise HTTPException(
            status_code=502,
            detail=f"face API returned a {len(embedding)}-d embedding, expected {FACE_EMBEDDING_DIM}-d",
        )

    db.query(EmployeeFaceEmbedding).filter(
        EmployeeFaceEmbedding.employee_id == employee_id, EmployeeFaceEmbedding.is_active
    ).update({"is_active": False})
    db.add(
        EmployeeFaceEmbedding(employee_id=employee_id, embedding=embedding, source_ref=face_id, is_active=True)
    )
    db.commit()

    return {"employee_id": employee_id, "face_id": face_id, "embedding_dim": len(embedding)}


@router.get("/{employee_id}/face-enrollment", response_model=FaceEnrollmentStatus)
def get_face_enrollment(employee_id: uuid.UUID, db: Session = Depends(get_db)) -> dict:
    if db.get(Employee, employee_id) is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    record = (
        db.query(EmployeeFaceEmbedding)
        .filter(EmployeeFaceEmbedding.employee_id == employee_id, EmployeeFaceEmbedding.is_active)
        .order_by(EmployeeFaceEmbedding.enrolled_at.desc())
        .first()
    )
    if record is None:
        return {"enrolled": False, "face_id": None, "enrolled_at": None}
    return {"enrolled": True, "face_id": record.source_ref, "enrolled_at": record.enrolled_at}


@router.get("/{employee_id}/photos/{pose}")
def get_enrollment_photo(employee_id: uuid.UUID, pose: EnrollmentPose) -> FileResponse:
    path = ENROLLMENT_PHOTOS_DIR / str(employee_id) / f"{pose}.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no '{pose}' photo stored for this employee")
    return FileResponse(path, media_type="image/jpeg")
