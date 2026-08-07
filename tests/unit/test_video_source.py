from pathlib import Path

import cv2
import numpy as np
import pytest

from visionstack.common.errors import VideoSourceError
from visionstack.ingestion.frame_sampler import FrameSampler
from visionstack.ingestion.video_source import (
    FileVideoSource,
    WebcamVideoSource,
    video_source_from_config,
)


def _write_synthetic_video(path: Path, num_frames: int, fps: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 32))
    for i in range(num_frames):
        frame = np.full((32, 32, 3), fill_value=i % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    video_path = tmp_path / "synthetic.mp4"
    _write_synthetic_video(video_path, num_frames=20, fps=10)
    return video_path


def test_file_video_source_reads_all_frames(synthetic_video: Path):
    source = FileVideoSource(path=str(synthetic_video), camera_id="test-cam")
    with source as opened:
        frames = list(opened)
    assert len(frames) == 20
    assert all(f.camera_id == "test-cam" for f in frames)
    assert [f.frame_id for f in frames] == list(range(20))


def test_video_source_error_on_missing_file(tmp_path: Path):
    source = FileVideoSource(path=str(tmp_path / "does_not_exist.mp4"), camera_id="test-cam")
    with pytest.raises(VideoSourceError):
        source.open()


def test_frame_sampler_downsamples_to_target_fps(synthetic_video: Path):
    source = FileVideoSource(path=str(synthetic_video), camera_id="test-cam")
    with source as opened:
        sampler = FrameSampler(opened, source_fps=10, sample_fps=5)
        frames = list(sampler)
    # source fps 10 -> target 5 means stride 2, so ~half the frames
    assert len(frames) == 10


def test_video_source_from_config_selects_file_backend(synthetic_video: Path):
    source = video_source_from_config("cam", "file", str(synthetic_video))
    assert isinstance(source, FileVideoSource)


def test_video_source_from_config_selects_webcam_backend():
    source = video_source_from_config("cam", "webcam", "0")
    assert isinstance(source, WebcamVideoSource)
    assert source._device_index == 0


def test_video_source_from_config_rejects_non_integer_webcam_index():
    with pytest.raises(VideoSourceError):
        video_source_from_config("cam", "webcam", "not-a-number")
