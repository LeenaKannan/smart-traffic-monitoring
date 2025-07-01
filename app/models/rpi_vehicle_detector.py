# app/models/rpi_vehicle_detector.py
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple
import logging
import time

logger = logging.getLogger(__name__)

class RPiVehicleDetector:
    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu"):
        """
        Raspberry Pi optimized vehicle detector
        Uses YOLOv8 nano for best performance on Pi 4
        """
        self.device = device
        
        # Load optimized model for Raspberry Pi
        try:
            self.model = YOLO(model_path)
            # Set to CPU and optimize for inference
            self.model.to('cpu')
            
            # Optimize for Raspberry Pi
            torch.set_num_threads(4)  # Pi 4 has 4 cores
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Indian vehicle classes optimized for detection
        self.indian_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
            5: 'bus', 7: 'truck', 9: 'auto_rickshaw'
        }
        
        # Performance tracking
        self.inference_times = []
        
        logger.info(f"RPi Vehicle detector initialized with model: {model_path}")
    
    def detect_vehicles(self, frame: np.ndarray, 
                       confidence_threshold: float = 0.4,
                       img_size: int = 416) -> List[Dict]:
        """
        Detect vehicles optimized for Raspberry Pi performance
        Lower image size and confidence for better FPS
        """
        start_time = time.time()
        
        try:
            # Resize frame for faster processing on Pi
            original_shape = frame.shape[:2]
            resized_frame = cv2.resize(frame, (img_size, img_size))
            
            # Run inference
            results = self.model(resized_frame, conf=confidence_threshold, verbose=False)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Scale coordinates back to original frame
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Scale back to original dimensions
                        scale_x = original_shape[1] / img_size
                        scale_y = original_shape[0] / img_size
                        
                        x1 = int(x1 * scale_x)
                        y1 = int(y1 * scale_y)
                        x2 = int(x2 * scale_x)
                        y2 = int(y2 * scale_y)
                        
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        if class_id in self.indian_classes:
                            detection = {
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': self.indian_classes[class_id],
                                'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                            }
                            detections.append(detection)
            
            # Track performance
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Keep only last 100 measurements
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-100:]
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in vehicle detection: {e}")
            return []
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.inference_times:
            return {"avg_fps": 0, "avg_inference_time": 0}
        
        avg_inference_time = np.mean(self.inference_times)
        avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        return {
            "avg_fps": round(avg_fps, 2),
            "avg_inference_time": round(avg_inference_time * 1000, 2),  # ms
            "total_inferences": len(self.inference_times)
        }
