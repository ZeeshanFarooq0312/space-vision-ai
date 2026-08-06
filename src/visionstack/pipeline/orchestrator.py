"""Composes VideoSource -> FrameSampler -> PersonDetector -> LocalTracker -> identity/attendance/
zone phases into a single runnable pipeline. Phases 3-7 are stubs today (see each module's NoOp
implementation) but are wired in now so the module-boundary design proves out end-to-end even
though only ingestion (Phase 1) and detection (Phase 2) do real work.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from visionstack.attendance.engine import AttendanceEngine, NoOpAttendanceEngine
from visionstack.attendance.presence_window import (
    NoOpPresenceConfirmationWindow,
    PresenceConfirmationWindow,
)
from visionstack.common.geometry import bbox_foot_point
from visionstack.common.types import Detection, Frame, Track
from visionstack.detection.person_detector import PersonDetector
from visionstack.identity.body_embedder import BodyEmbedder, NoOpBodyEmbedder
from visionstack.identity.face_detector import FaceDetector, NoOpFaceDetector
from visionstack.identity.face_embedder import FaceEmbedder, NoOpFaceEmbedder
from visionstack.identity.matcher import IdentityMatcher, NoOpIdentityMatcher
from visionstack.ingestion.frame_sampler import FrameSampler
from visionstack.ingestion.video_source import VideoSource
from visionstack.tracking.local_tracker import LocalTracker, PassthroughLocalTracker
from visionstack.zones.alerts import AlertDispatcher, LoggingAlertDispatcher
from visionstack.zones.monitor import NoOpZoneMonitor, ZoneMonitor

logger = logging.getLogger("visionstack.pipeline")


@dataclass
class FrameResult:
    frame: Frame
    detections: list[Detection]
    tracks: list[Track]


@dataclass
class Pipeline:
    video_source: VideoSource
    detector: PersonDetector
    sample_fps: float = 5.0
    local_tracker: LocalTracker = field(default_factory=PassthroughLocalTracker)
    face_detector: FaceDetector = field(default_factory=NoOpFaceDetector)
    face_embedder: FaceEmbedder = field(default_factory=NoOpFaceEmbedder)
    body_embedder: BodyEmbedder = field(default_factory=NoOpBodyEmbedder)
    identity_matcher: IdentityMatcher = field(default_factory=NoOpIdentityMatcher)
    zone_monitor: ZoneMonitor = field(default_factory=NoOpZoneMonitor)
    attendance_engine: AttendanceEngine = field(default_factory=NoOpAttendanceEngine)
    presence_window: PresenceConfirmationWindow = field(
        default_factory=NoOpPresenceConfirmationWindow
    )
    alert_dispatcher: AlertDispatcher = field(default_factory=LoggingAlertDispatcher)

    def run(self) -> None:
        with self.video_source as source:
            sampler = FrameSampler(source, source_fps=source.fps, sample_fps=self.sample_fps)
            for frame in sampler:
                yield self._process_frame(frame)

    def _process_frame(self, frame: Frame) -> FrameResult:
        detections = self.detector.detect(frame)
        tracks = self.local_tracker.update(detections)

        for track in tracks:
            latest = track.detections[-1]
            person_crop = frame.image[
                int(latest.bbox.y1) : int(latest.bbox.y2), int(latest.bbox.x1) : int(latest.bbox.x2)
            ]

            faces = self.face_detector.detect(person_crop)
            face_embedding = self.face_embedder.embed(person_crop) if faces else None
            body_embedding = self.body_embedder.embed(person_crop)

            self.identity_matcher.match(face_embedding, body_embedding)

            foot_point = bbox_foot_point(latest.bbox)
            self.zone_monitor.check(
                track_id=track.track_id,
                camera_id=frame.camera_id,
                point=foot_point,
                employee_role=None,
                ts=frame.timestamp,
            )

        logger.info(
            "camera=%s frame=%d ts=%s detections=%d",
            frame.camera_id,
            frame.frame_id,
            frame.timestamp.isoformat(),
            len(detections),
        )
        return FrameResult(frame=frame, detections=detections, tracks=tracks)
