# app/database/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid

Base = declarative_base()

class TrafficEvent(Base):
    __tablename__ = "traffic_events"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    camera_id = Column(String, nullable=False, index=True)
    event_type = Column(String, nullable=False)  # detection, violation, density
    timestamp = Column(DateTime, server_default=func.now())
    data = Column(JSON)
    processed = Column(Boolean, default=False)

class VehicleDetection(Base):
    __tablename__ = "vehicle_detections"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    camera_id = Column(String, nullable=False, index=True)
    vehicle_class = Column(String, nullable=False)
    confidence = Column(Float)
    bbox = Column(JSON)  # [x1, y1, x2, y2]
    timestamp = Column(DateTime, server_default=func.now())
    speed = Column(Float, nullable=True)
    license_plate = Column(String, nullable=True)

class TrafficViolation(Base):
    __tablename__ = "traffic_violations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    camera_id = Column(String, nullable=False, index=True)
    violation_type = Column(String, nullable=False)
    vehicle_class = Column(String)
    license_plate = Column(String, nullable=True)
    timestamp = Column(DateTime, server_default=func.now())
    evidence_path = Column(String)  # Path to saved image/video
    fine_amount = Column(Float, nullable=True)
    status = Column(String, default="pending")  # pending, processed, dismissed

class IntersectionStatus(Base):
    __tablename__ = "intersection_status"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    current_signal_group = Column(String)
    signal_state = Column(String)  # green, yellow, red
    last_updated = Column(DateTime, server_default=func.now())
    config = Column(JSON)

class TrafficDensity(Base):
    __tablename__ = "traffic_density"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    camera_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, server_default=func.now())
    total_vehicles = Column(Integer)
    vehicle_counts = Column(JSON)  # {"car": 5, "motorcycle": 10, ...}
    density_level = Column(String)  # low, moderate, high, severe
    congestion_score = Column(Float)
