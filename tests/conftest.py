from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def person_image_path() -> Path:
    """A real photo containing people, for deterministic PersonDetector tests.

    Sourced from ultralytics' own bundled sample assets (ships with the `ultralytics` pip
    package) rather than a custom committed binary — avoids bloating the repo and needs no
    network access beyond what installing `ultralytics` already requires.
    """
    from ultralytics.utils import ASSETS

    return Path(ASSETS) / "bus.jpg"


@pytest.fixture
def sample_video_path() -> Path:
    """User-supplied sample clip. See samples/README.md. Tests that need this should skip
    (not fail) when it's absent."""
    return REPO_ROOT / "samples" / "sample.mp4"
