# app/models/vehicle_detector.py
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class VehicleDetector:
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize vehicle detector with YOLOv8 model optimized for Indian traffic
        """
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(device)
        
        # Indian vehicle class mapping
        self.indian_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 
            5: 'bus', 7: 'truck', 9: 'auto_rickshaw'
        }
        
        logger.info(f"Vehicle detector initialized with model: {model_path}")
    
    def detect_vehicles(self, frame: np.ndarray, 
                       confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Detect vehicles in frame with Indian traffic considerations
        """
        try:
            results = self.model(frame, conf=confidence_threshold)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        if class_id in self.indian_classes:
                            detection = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(confidence),
                                'class_id': class_id,
                                'class_name': self.indian_classes[class_id],
                                'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                            }
                            detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in vehicle detection: {e}")
            return []
    
    def estimate_speed(self, prev_center: Tuple[float, float], 
                      curr_center: Tuple[float, float], 
                      time_diff: float, 
                      pixels_per_meter: float = 10.0) -> float:
        """
        Estimate vehicle speed based on center point movement
        """
        if time_diff <= 0:
            return 0.0
        
        distance_pixels = np.sqrt(
            (curr_center[0] - prev_center[0])**2 + 
            (curr_center[1] - prev_center[1])**2
        )
        
        distance_meters = distance_pixels / pixels_per_meter
        speed_mps = distance_meters / time_diff
        speed_kmph = speed_mps * 3.6
        
        return speed_kmph