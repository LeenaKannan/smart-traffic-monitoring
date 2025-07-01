# docker/Dockerfile
FROM python:3.9-slim

# Install system dependencies
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
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly \
  tesseract-ocr \
  tesseract-ocr-eng \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/models logs

# Download YOLOv8 model
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start application
CMD ["python", "-m", "app.main"]
