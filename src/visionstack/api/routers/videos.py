from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from visionstack.api.live_stream import get_processed_video_path, list_processed_videos
from visionstack.api.schemas import ProcessedVideo

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
