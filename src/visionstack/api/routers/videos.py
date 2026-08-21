import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from visionstack.api.live_stream import get_processed_video_path, list_processed_videos
from visionstack.api.schemas import ProcessedVideo, UploadJobStatus, VideoUploadResponse
from visionstack.api.video_upload import UPLOADS_DIR, video_upload_processor

router = APIRouter(prefix="/videos", tags=["videos"])


@router.get("", response_model=list[ProcessedVideo])
def get_videos() -> list[dict]:
    return list_processed_videos()


@router.get("/{video_id}/file")
def get_video_file(video_id: str) -> FileResponse:
    path = get_processed_video_path(video_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"no processed video '{video_id}'")
    return FileResponse(path, media_type="video/mp4")


@router.post("/upload", response_model=VideoUploadResponse, status_code=202)
def upload_video(file: UploadFile = File(...), sample_fps: float = Form(4.0)) -> dict:
    """Accepts a pre-recorded video (e.g. a CCTV export) and processes it through the same
    detection + recognition pipeline as a live session, in the background -- poll
    /videos/upload/{video_id}/status, then find the result in GET /videos once status is 'done'.
    """
    if not (file.content_type or "").startswith("video/"):
        raise HTTPException(status_code=400, detail=f"expected a video, got '{file.content_type}'")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "upload.mp4").suffix or ".mp4"
    raw_path = UPLOADS_DIR / f"{uuid.uuid4().hex}{ext}"
    with raw_path.open("wb") as dest:
        # UploadFile streams to a spooled temp file already; copy in chunks rather than
        # file.read() so a large video doesn't get pulled fully into memory first.
        while chunk := file.file.read(1024 * 1024):
            dest.write(chunk)

    video_id = video_upload_processor.start(raw_path, file.filename or "upload.mp4", sample_fps)
    return {"video_id": video_id, "status": "processing"}


@router.get("/upload/{video_id}/status", response_model=UploadJobStatus)
def get_upload_status(video_id: str) -> dict:
    status = video_upload_processor.status(video_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"no upload job '{video_id}'")
    return status


@router.get("/upload/{video_id}/preview-frame")
def get_upload_preview_frame(video_id: str) -> Response:
    """First frame of the uploaded video, for the zone-drawing UI (frontend's ZoneDrawer, reused
    against a static image here instead of a live MJPEG stream) -- see video_upload.py's
    UPLOAD_CAMERA_ID docstring for why a zone drawn against this applies to every future upload.
    """
    frame = video_upload_processor.preview_frame(video_id)
    if frame is None:
        raise HTTPException(status_code=404, detail=f"no preview frame for '{video_id}'")
    return Response(content=frame, media_type="image/jpeg")


@router.get("/upload/{video_id}/stream")
def stream_upload_progress(video_id: str) -> StreamingResponse:
    """Annotated-frame MJPEG stream of an in-progress upload job -- same idea as
    /live/{camera_id}/stream, so the frontend can show "results in preview" while a video is
    being processed, not just once it's finished. Stops once the job leaves 'processing'."""
    return StreamingResponse(
        video_upload_processor.mjpeg_frames(video_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/{video_id}/zone-crop/{filename}")
def get_zone_crop(video_id: str, filename: str) -> FileResponse:
    path = video_upload_processor.zone_crop_path(video_id, filename)
    if path is None:
        raise HTTPException(status_code=404, detail=f"no zone crop '{filename}' for '{video_id}'")
    return FileResponse(path, media_type="image/jpeg")
