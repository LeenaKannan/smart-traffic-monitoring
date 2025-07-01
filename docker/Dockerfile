# docker/Dockerfile.rpi
FROM python:3.9-slim

# Install system dependencies for Raspberry Pi
RUN apt-get update && apt-get install -y \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgomp1 \
  libgstreamer1.0-0 \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  python3-opencv \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements for Pi
COPY requirements_rpi.txt .
RUN pip install --no-cache-dir -r requirements_rpi.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p data/models logs

# Download YOLOv8 nano model
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Set environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=10s --retries=3 \
  CMD python -c "import cv2; print('OK')" || exit 1

EXPOSE 8000

CMD ["python", "demo/rpi_demo.py"]
