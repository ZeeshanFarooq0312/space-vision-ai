"""Runnable Phase 1-2 demo:

    python -m visionstack.pipeline.run_pipeline --source samples/sample.mp4 --camera-id entry-cam-1
"""
from __future__ import annotations

import logging

import cv2
import typer

from visionstack.common.types import Detection
from visionstack.detection.person_detector import PersonDetector
from visionstack.ingestion.video_source import video_source_from_config
from visionstack.pipeline.orchestrator import FrameResult, Pipeline
from visionstack.tracking.local_tracker import (
    LocalTracker,
    PassthroughLocalTracker,
    TrackTrackLocalTracker,
)

app = typer.Typer(add_completion=False)


def _draw_detections(image, detections: list[Detection]):
    for d in detections:
        p1 = (int(d.bbox.x1), int(d.bbox.y1))
        p2 = (int(d.bbox.x2), int(d.bbox.y2))
        cv2.rectangle(image, p1, p2, (0, 200, 0), 2)
        cv2.putText(
            image,
            f"person {d.confidence:.2f}",
            (p1[0], max(0, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            1,
            cv2.LINE_AA,
        )
    return image


@app.command()
def main(
    source: str = typer.Option(..., help="Video file path, RTSP URL, or webcam device index (e.g. '0')"),
    camera_id: str = typer.Option("camera-1", help="Camera identifier to tag frames/detections with"),
    source_type: str = typer.Option("file", help="'file', 'rtsp', or 'webcam'"),
    sample_fps: float = typer.Option(5.0, help="Target frames-per-second to process"),
    weights: str = typer.Option("models/yolov8n.pt", help="YOLOv8 weights path"),
    device: str = typer.Option("auto", help="'auto', 'cpu', or 'cuda'"),
    conf_threshold: float = typer.Option(0.5),
    iou_threshold: float = typer.Option(0.45),
    tracker: str = typer.Option(
        "passthrough",
        help="'passthrough' (fresh id per detection) or 'tracktrack' (real Kalman+ReID tracking "
        "-- fixed cameras only, see tracking/local_tracker.py)",
    ),
    show: bool = typer.Option(False, help="Write an annotated output video (out.mp4) for visual sanity-checking"),
    display: bool = typer.Option(
        False, help="Show a live annotated preview window while processing (press 'q' to stop)"
    ),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    if tracker not in ("passthrough", "tracktrack"):
        raise typer.BadParameter("--tracker must be 'passthrough' or 'tracktrack'")

    video_source = video_source_from_config(camera_id=camera_id, source_type=source_type, uri=source)
    detector = PersonDetector(
        weights_path=weights, device=device, conf_threshold=conf_threshold, iou_threshold=iou_threshold
    )
    local_tracker: LocalTracker = (
        TrackTrackLocalTracker(sample_fps=sample_fps)
        if tracker == "tracktrack"
        else PassthroughLocalTracker()
    )
    pipeline = Pipeline(
        video_source=video_source,
        detector=detector,
        sample_fps=sample_fps,
        local_tracker=local_tracker,
    )

    writer: cv2.VideoWriter | None = None
    frame_count = 0
    detection_count = 0
    window_name = f"visionstack — {camera_id}"

    try:
        result: FrameResult
        for result in pipeline.run():
            frame_count += 1
            detection_count += len(result.detections)
            for d in result.detections:
                typer.echo(
                    f"  frame={d.frame_id} bbox=({d.bbox.x1:.0f},{d.bbox.y1:.0f},{d.bbox.x2:.0f},{d.bbox.y2:.0f}) conf={d.confidence:.2f}"
                )

            if show or display:
                annotated = _draw_detections(result.frame.image.copy(), result.detections)

                if show:
                    if writer is None:
                        h, w = annotated.shape[:2]
                        writer = cv2.VideoWriter(
                            "out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), sample_fps, (w, h)
                        )
                    writer.write(annotated)

                if display:
                    cv2.imshow(window_name, annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        typer.echo("Stopped by user ('q' pressed).")
                        break
    finally:
        if writer is not None:
            writer.release()
            typer.echo("Wrote annotated output to out.mp4")
        if display:
            cv2.destroyAllWindows()

    typer.echo(f"Processed {frame_count} frames, {detection_count} total detections.")


if __name__ == "__main__":
    app()
