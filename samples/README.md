Drop a short (5-15s) video clip of a person walking here as `sample.mp4` (gitignored — not
committed). It's used by:

- `python -m visionstack.pipeline.run_pipeline --source samples/sample.mp4 --camera-id entry-cam-1`
- `tests/integration/test_run_pipeline_on_sample_video.py` (auto-skips if this file is absent)

Any MP4 with at least one visible person works — a phone recording of someone walking through
a doorway is enough to sanity-check the detection pipeline end-to-end.
