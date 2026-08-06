# GPU-enabled inference worker. Opt-in via `docker compose --profile gpu up inference`.
# Requires the NVIDIA Container Toolkit on the host — not verified to work on Windows/Docker
# Desktop without a validated WSL2 + CUDA driver setup. See README for the CPU-fallback dev path.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY samples/ samples/

ENV PYTHONPATH=/app/src

ENTRYPOINT ["python3", "-m", "visionstack.pipeline.run_pipeline"]
