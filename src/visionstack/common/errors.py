class VisionStackError(Exception):
    """Base class for all visionstack-raised exceptions."""


class ConfigError(VisionStackError):
    """Raised when a config file (cameras.yaml, zones.yaml, pipeline.yaml) is invalid."""


class VideoSourceError(VisionStackError):
    """Raised when a video/RTSP source cannot be opened or read."""


class FaceApiError(VisionStackError):
    """Raised when the external face-embedding API rejects a request or returns an unexpected shape."""
