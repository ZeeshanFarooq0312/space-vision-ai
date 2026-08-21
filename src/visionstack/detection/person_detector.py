"""Phase 2 — YOLO person detection. Model architecture/version comes from whatever checkpoint
`weights_path` points at (see DETECTION_WEIGHTS_PATH in docker-compose.yml -- currently stock
YOLOv12m); this module itself is version-agnostic and just filters to PERSON_CLASS_ID. NMS +
confidence filtering handled internally by ultralytics."""
from __future__ import annotations

import torch
from ultralytics import YOLO

from visionstack.common.types import BBox, Detection, Frame

PERSON_CLASS_ID = 0  # COCO class index for "person"


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class PersonDetector:
    def __init__(
        self,
        weights_path: str = "models/yolov8n.pt",
        device: str = "auto",
        # Lowered from 0.3 after switching DETECTION_WEIGHTS_PATH to stock (non-fine-tuned)
        # yolo12m.pt -- real footage showed people fully missing boxes (not just low-confidence
        # ones) on this camera's wide/fisheye angle and unusual poses (bent over, side-on, partly
        # behind a monitor), which stock COCO weights handle worse than the office-trained
        # fine-tune did. This only recovers borderline-confidence misses, not detections that
        # never fired at all -- if people are still going unboxed, that's a model-capability gap,
        # not a threshold one, and the fine-tune (models/person_yolo12m_custom.pt) is the fix.
        conf_threshold: float = 0.15,
        iou_threshold: float = 0.2,
    ) -> None:
        self.device = resolve_device(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._model = YOLO(weights_path)

    def detect(self, frame: Frame) -> list[Detection]:
        results = self._model.predict(
            frame.image,
            classes=[PERSON_CLASS_ID],
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        boxes = results[0].boxes
        detections: list[Detection] = []
        for xyxy, conf in zip(boxes.xyxy.tolist(), boxes.conf.tolist()):
            x1, y1, x2, y2 = xyxy
            detections.append(
                Detection(
                    camera_id=frame.camera_id,
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=conf,
                    class_name="person",
                )
            )
        return detections

    def track(self, frame: Frame) -> list[Detection]:
        """Like detect(), but assigns a persistent track_id per person via ultralytics' built-in
        BoT-SORT, so the same person keeps the same id across frames instead of every detection
        being independent. `persist=True` keeps the tracker's internal state alive across calls on
        this same PersonDetector instance -- callers must reuse one instance for a whole
        session/video (as live_stream.py and video_upload.py already do), not construct a fresh
        one per frame, or every person would get a new id every call.

        BoT-SORT, not ByteTrack: confirmed directly against a real live session recording where a
        handheld/laptop webcam got physically moved mid-session (room angle changed completely,
        motion blur) -- ByteTrack's Kalman-filter motion model assumes a roughly static camera, so
        every pan shifted all boxes at once and looked like an entirely new set of people to it,
        producing runaway track ids (already in the high 20s after 20 real seconds) instead of a
        handful of stable ones. BoT-SORT's gmc_method (global motion compensation via sparse optical
        flow, on by default, no extra ReID model/dependency needed) is specifically built for camera
        ego-motion like this. For a truly fixed CCTV mount this distinction matters less, but a
        handheld/laptop camera is exactly what was being tested, and BoT-SORT costs nothing extra
        here since ultralytics ships it already.
        """
        results = self._model.track(
            frame.image,
            classes=[PERSON_CLASS_ID],
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            persist=True,
            tracker="botsort.yaml",
            verbose=False,
        )
        boxes = results[0].boxes
        track_ids = boxes.id.tolist() if boxes.id is not None else [None] * len(boxes)
        detections: list[Detection] = []
        for xyxy, conf, track_id in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), track_ids):
            x1, y1, x2, y2 = xyxy
            detections.append(
                Detection(
                    camera_id=frame.camera_id,
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=conf,
                    class_name="person",
                    track_id=str(int(track_id)) if track_id is not None else None,
                )
            )
        return detections
