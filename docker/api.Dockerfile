FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CUDA-enabled torch/torchvision (via ultralytics' unpinned torch>=1.8.0): this container now
# gets GPU passthrough (see docker-compose.yml's `api` service `deploy.resources.reservations`),
# so pull the real CUDA build instead of the CPU-only one. Those wheels run multiple GB total --
# --default-timeout/--retries guard against the same slow-segment timeout hit previously on the
# single ~366MB nvidia-cudnn wheel.
RUN pip install --no-cache-dir --default-timeout=180 --retries 10 -r requirements.txt

COPY src/ src/
COPY configs/ configs/

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "visionstack.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
