from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from visionstack.api.routers import alerts, attendance, cameras, employees, health, zones


def create_app() -> FastAPI:
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

    return app


app = create_app()
