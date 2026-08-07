import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from visionstack.api.routers import alerts, attendance, cameras, employees, health, live, videos, zones
from visionstack.common.config import get_settings


def create_app() -> FastAPI:
    # Without this, our loggers (visionstack.*) have no handler under uvicorn and INFO-level
    # calls are silently dropped -- run_pipeline.py's CLI entrypoint sets this up, but nothing
    # did for the API entrypoint, so every logger.info/exception call in the live-recognition
    # pipeline (verify_face results, batch errors, etc.) was invisible in `docker logs` the whole
    # time this ran in the container, regardless of whether recognition itself was working.
    logging.basicConfig(level=get_settings().log_level, format="%(asctime)s %(name)s %(message)s")

    app = FastAPI(title="Vision-Stack AI", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(employees.router)
    app.include_router(cameras.router)
    app.include_router(zones.router)
    app.include_router(attendance.router)
    app.include_router(alerts.router)
    app.include_router(live.router)
    app.include_router(videos.router)

    return app


app = create_app()
