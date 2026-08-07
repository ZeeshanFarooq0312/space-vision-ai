"""Background processing of an uploaded (pre-recorded) video file through Phase 1-3
(detection + recognition), writing the same annotated-recording + metadata format
live_stream.py writes for webcam sessions -- uploaded videos show up in the same /videos
list and Recordings UI, no separate view needed.

Exists for diagnosing recognition behavior against footage the live pipeline already
captured (or any other higher-quality/pre-recorded CCTV clip) without needing to
reproduce a live camera session: a fixed file can be re-processed, watched frame by frame,
and its log lines correlated with the annotated output on demand.
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import cv2

from visionstack.api.live_stream import (
    PROCESSED_VIDEOS_DIR,
    _draw_detections,
    _transcode_to_h264,
    build_face_crops,
    verify_crops_batch,
)
from visionstack.common.config import REPO_ROOT
from visionstack.detection.person_detector import PersonDetector
from visionstack.identity.face_detector import HaarFaceDetector
from visionstack.ingestion.frame_sampler import FrameSampler
from visionstack.ingestion.video_source import FileVideoSource

logger = logging.getLogger("visionstack.api.video_upload")

UPLOADS_DIR = REPO_ROOT / "data" / "video_uploads"
JobStatus = Literal["processing", "done", "error"]
# Verification is keyed to processed-frame count, not wall-clock time: a file is processed as
# fast as the CPU/GPU allow (not paced to real time like a live session), so a time-based
# throttle would fire far more or less often than intended depending on machine speed.
VERIFY_EVERY_N_FRAMES = 8


@dataclass
class _Job:
    video_id: str
    original_filename: str
    status: JobStatus = "processing"
    frame_count: int = 0
    detection_count: int = 0
    max_people_in_frame: int = 0
    error: str | None = None


class VideoUploadProcessor:
    """Process-wide registry of upload-processing jobs, keyed by video_id."""

    def __init__(self) -> None:
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()

    def start(self, raw_path: Path, original_filename: str, sample_fps: float) -> str:
        video_id = uuid.uuid4().hex
        job = _Job(video_id=video_id, original_filename=original_filename)
        with self._lock:
            self._jobs[video_id] = job
        threading.Thread(
            target=self._process, args=(job, raw_path, sample_fps), daemon=True,
            name=f"upload-{video_id}",
        ).start()
        return video_id

    def status(self, video_id: str) -> dict | None:
        job = self._jobs.get(video_id)
        if job is None:
            return None
        return {
            "video_id": job.video_id,
            "status": job.status,
            "frame_count": job.frame_count,
            "detection_count": job.detection_count,
            "error": job.error,
        }

    def _process(self, job: _Job, raw_path: Path, sample_fps: float) -> None:
        PROCESSED_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        working_mp4 = PROCESSED_VIDEOS_DIR / f"{job.video_id}_raw.mp4"
        output_mp4 = PROCESSED_VIDEOS_DIR / f"{job.video_id}.mp4"
        started_at = datetime.now(timezone.utc)
        all_recognized_names: set[str] = set()
        writer: cv2.VideoWriter | None = None
        camera_id = f"upload:{job.original_filename}"

        try:
            detector = PersonDetector(device="auto")
            face_detector = HaarFaceDetector()
            video_source = FileVideoSource(path=str(raw_path), camera_id=camera_id)

            with video_source as source:
                sampler = FrameSampler(source, source_fps=source.fps, sample_fps=sample_fps)
                recognized_labels: list[str | None] = []
                for processed_index, frame in enumerate(sampler):
                    detections = detector.detect(frame)

                    if detections and processed_index % VERIFY_EVERY_N_FRAMES == 0:
                        face_crops = build_face_crops(detections, frame.image, face_detector)
                        recognized_labels = verify_crops_batch(face_crops, context=camera_id)
                        all_recognized_names.update(n for n in recognized_labels if n)
                    elif not detections:
                        recognized_labels = []

                    annotated = _draw_detections(frame.image.copy(), detections, recognized_labels)

                    if writer is None:
                        h, w = annotated.shape[:2]
                        writer = cv2.VideoWriter(
                            str(working_mp4), cv2.VideoWriter_fourcc(*"mp4v"), sample_fps, (w, h)
                        )
                    writer.write(annotated)

                    job.frame_count += 1
                    job.detection_count += len(detections)
                    job.max_people_in_frame = max(job.max_people_in_frame, len(detections))
        except Exception as e:  # noqa: BLE001 -- surface via status() instead of a silent thread death
            logger.exception("upload processing failed for '%s'", job.original_filename)
            job.status = "error"
            job.error = str(e)
            if writer is not None:
                writer.release()
            working_mp4.unlink(missing_ok=True)
            raw_path.unlink(missing_ok=True)
            return

        if writer is not None:
            writer.release()
        raw_path.unlink(missing_ok=True)

        if job.frame_count == 0:
            working_mp4.unlink(missing_ok=True)
            job.status = "error"
            job.error = "No frames could be read from the uploaded video."
            return

        playable = _transcode_to_h264(working_mp4, output_mp4)
        if not playable:
            working_mp4.replace(output_mp4)

        ended_at = datetime.now(timezone.utc)
        metadata = {
            "video_id": job.video_id,
            "camera_id": camera_id,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_seconds": (ended_at - started_at).total_seconds(),
            "frame_count": job.frame_count,
            "detection_count": job.detection_count,
            "max_people_in_frame": job.max_people_in_frame,
            "filename": output_mp4.name,
            "browser_playable": playable,
            "recognized_names": sorted(all_recognized_names),
        }
        output_mp4.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
        job.status = "done"


video_upload_processor = VideoUploadProcessor()
