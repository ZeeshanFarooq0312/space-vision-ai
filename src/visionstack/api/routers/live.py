from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from visionstack.api.live_stream import list_webcam_devices, live_stream_manager
from visionstack.api.schemas import LiveSessionStart, LiveSessionStatus, WebcamDevice

router = APIRouter(prefix="/live", tags=["live"])


@router.get("/devices", response_model=list[WebcamDevice])
def get_devices() -> list[dict]:
    return list_webcam_devices()


@router.post("/{camera_id}/start", response_model=LiveSessionStatus)
def start_live(camera_id: str, payload: LiveSessionStart) -> dict:
    try:
        live_stream_manager.start(camera_id, payload.device_index, payload.sample_fps)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return live_stream_manager.status(camera_id)


@router.post("/{camera_id}/stop", response_model=LiveSessionStatus)
def stop_live(camera_id: str) -> dict:
    try:
        live_stream_manager.stop(camera_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"no session for '{camera_id}'") from e
    return live_stream_manager.status(camera_id)


@router.get("/{camera_id}/status", response_model=LiveSessionStatus)
def get_status(camera_id: str) -> dict:
    return live_stream_manager.status(camera_id)


@router.get("/{camera_id}/stream")
def stream_live(camera_id: str) -> StreamingResponse:
    return StreamingResponse(
        live_stream_manager.mjpeg_frames(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
