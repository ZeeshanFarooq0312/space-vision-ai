from pathlib import Path

import pytest

from visionstack.detection.person_detector import PersonDetector
from visionstack.ingestion.video_source import FileVideoSource
from visionstack.pipeline.orchestrator import Pipeline


def test_pipeline_runs_end_to_end_on_sample_video(sample_video_path: Path):
    if not sample_video_path.exists():
        pytest.skip(
            f"{sample_video_path} not present — see samples/README.md to add a clip for this test"
        )

    video_source = FileVideoSource(path=str(sample_video_path), camera_id="entry-cam-1")
    detector = PersonDetector(device="cpu")
    pipeline = Pipeline(video_source=video_source, detector=detector, sample_fps=5.0)

    frame_count = 0
    for result in pipeline.run():
        frame_count += 1
        assert result.frame.camera_id == "entry-cam-1"

    assert frame_count > 0
