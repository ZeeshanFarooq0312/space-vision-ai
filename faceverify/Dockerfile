# tensorflow/tensorflow:2.16.1-gpu ships Python 3.11 + CUDA 12.3 + cuDNN 8.9 pre-wired.
# Using python:*-slim was the bug: it has NO CUDA libs, so tf.config.list_physical_devices('GPU') → [].
FROM tensorflow/tensorflow:2.16.1-gpu

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    ABSL_LOGGING_LEVEL=0 \
    TF_FORCE_GPU_ALLOW_GROWTH=true

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --ignore-installed blinker && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create the temp directory for image processing and the public src directory
RUN mkdir -p /app/temp_mobile /app/src

# Expose the API port
EXPOSE 8024

# Run the application (Change 'main:app' if your file is named differently)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8024"]