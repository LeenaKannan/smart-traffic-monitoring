# app/config/settings.py
import os
from pydantic import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    
    IS_RASPBERRY_PI: bool = platform.machine().startswith('arm')
    PROCESSING_MODE: str = "rpi" if IS_RASPBERRY_PI else "standard"
    MAX_RESOLUTION: str = "640x480" if IS_RASPBERRY_PI else "1920x1080"

    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False
    
    # AI Model Paths
    YOLO_MODEL_PATH: str = "data/models/yolov8n.pt"
    LICENSE_PLATE_MODEL_PATH: str = "data/models/license_plate_model.pt"
    
    # Video Processing
    VIDEO_SOURCES: List[str] = ["rtsp://camera1", "rtsp://camera2"]
    FRAME_SKIP: int = 2
    CONFIDENCE_THRESHOLD: float = 0.5
    
    # Traffic Control
    MIN_GREEN_TIME: int = 15
    MAX_GREEN_TIME: int = 120
    YELLOW_TIME: int = 3
    
    # Database
    DATABASE_URL: str = "postgresql://user:pass@localhost/traffic_db"
    REDIS_URL: str = "redis://localhost:6379"
    
    # Kafka Configuration
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TOPIC_VIOLATIONS: str = "traffic_violations"
    KAFKA_TOPIC_DENSITY: str = "traffic_density"
    
    # Indian Traffic Specific
    INDIAN_VEHICLE_CLASSES: List[str] = [
        "car", "motorcycle", "bus", "truck", "auto_rickshaw", 
        "bicycle", "pedestrian", "cow", "dog"
    ]
    
    class Config:
        env_file = ".env"

settings = Settings()
