"""Phase 2 — YOLOv8 person detection. NMS + confidence filtering handled internally by ultralytics."""
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
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
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
