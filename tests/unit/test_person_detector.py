from datetime import datetime, timezone
from pathlib import Path

import cv2

from visionstack.common.types import Frame
from visionstack.detection.person_detector import PersonDetector


def test_person_detector_finds_people_in_bus_jpg(person_image_path: Path):
    image = cv2.imread(str(person_image_path))
    assert image is not None, f"could not read fixture image at {person_image_path}"

    frame = Frame(camera_id="test-cam", frame_id=0, timestamp=datetime.now(timezone.utc), image=image)
    detector = PersonDetector(device="cpu")
    detections = detector.detect(frame)

    assert len(detections) >= 1
    for d in detections:
        assert 0.0 <= d.confidence <= 1.0
        assert d.class_name == "person"
        assert 0 <= d.bbox.x1 < d.bbox.x2 <= image.shape[1]
        assert 0 <= d.bbox.y1 < d.bbox.y2 <= image.shape[0]
