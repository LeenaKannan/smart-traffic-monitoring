# demo/rpi_demo.py
#!/usr/bin/env python3
"""
Raspberry Pi 4 Demo for Smart Traffic Monitoring System
Optimized for real-time performance on Pi 4
"""

import cv2
import asyncio
import argparse
import logging
import time
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.rpi_stream_processor import RPiStreamProcessor
from app.models.rpi_vehicle_detector import RPiVehicleDetector
from app.models.vehicle_tracker import VehicleTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RPiTrafficDemo:
    def __init__(self, source=0, model_path="yolov8n.pt"):
        """Initialize Raspberry Pi traffic monitoring demo"""
        self.source = source
        self.detector = RPiVehicleDetector(model_path)
        self.tracker = VehicleTracker()
        self.running = False
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def draw_detections(self, frame: np.ndarray, detections: list, 
                       tracked_vehicles: dict) -> np.ndarray:
        """Draw detection boxes and tracking info on frame"""
        # Draw detections
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = bbox
            
            # Color coding for different vehicle types
            colors = {
                'car': (0, 255, 0),
                'motorcycle': (255, 0, 0),
                'bus': (0, 0, 255),
                'truck': (255, 255, 0),
                'auto_rickshaw': (255, 0, 255),
                'bicycle': (0, 255, 255),
                'person': (128, 128, 128)
            }
            
            color = colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw tracking info
        for obj_id, vehicle_data in tracked_vehicles.items():
            centroid = vehicle_data['centroid']
            speed = vehicle_data.get('speed', 0)
            direction = vehicle_data.get('direction', 'unknown')
            
            # Draw centroid
            cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 5, (0, 0, 255), -1)
            
            # Draw object ID and speed
            text = f"ID:{obj_id} {speed:.1f}km/h {direction}"
            cv2.putText(frame, text, (int(centroid[0]) - 50, int(centroid[1]) - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Draw trajectory
            trajectory = vehicle_data.get('trajectory', [])
            if len(trajectory) > 1:
                points = np.array(trajectory, dtype=np.int32)
                cv2.polylines(frame, [points], False, (0, 255, 255), 2)
        
        return frame
    
    def draw_stats(self, frame: np.ndarray, detections: list, 
                   tracked_vehicles: dict, processing_time: float) -> np.ndarray:
        """Draw statistics on frame"""
        height, width = frame.shape[:2]
        
        # Count vehicles by type
        vehicle_counts = {}
        for detection in detections:
            class_name = detection['class_name']
            vehicle_counts[class_name] = vehicle_counts.get(class_name, 0) + 1
        
        # Draw statistics panel
        panel_height = 150
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        
        y_offset = 20
        
        # FPS and processing time
        cv2.putText(panel, f"FPS: {self.current_fps:.1f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(panel, f"Processing: {processing_time*1000:.1f}ms", (150, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y_offset += 25
        
        # Vehicle counts
        cv2.putText(panel, f"Total Vehicles: {len(detections)}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(panel, f"Tracked: {len(tracked_vehicles)}", (200, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 20
        
        # Individual vehicle counts
        for i, (vehicle_type, count) in enumerate(vehicle_counts.items()):
            x_pos = 10 + (i % 3) * 120
            y_pos = y_offset + (i // 3) * 20
            cv2.putText(panel, f"{vehicle_type}: {count}", (x_pos, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Combine frame and panel
        combined = np.vstack([frame, panel])
        return combined
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def run(self):
        """Run the demo"""
        logger.info("Starting Raspberry Pi Traffic Monitoring Demo")
        logger.info(f"Using source: {self.source}")
        
        # Initialize video capture
        cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            logger.error(f"Cannot open video source: {self.source}")
            return
        
        # Optimize capture for Raspberry Pi
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.running = True
        frame_count = 0
        
        logger.info("Demo started. Press 'q' to quit, 's' to save screenshot")
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    break
                
                frame_count += 1
                
                # Process every 2nd frame for better performance
                if frame_count % 2 == 0:
                    start_time = time.time()
                    
                    # Detect vehicles
                    detections = self.detector.detect_vehicles(frame, confidence_threshold=0.4)
                    
                    # Track vehicles
                    tracked_vehicles = self.tracker.update(detections)
                    
                    processing_time = time.time() - start_time
                    
                    # Draw results
                    frame = self.draw_detections(frame, detections, tracked_vehicles)
                    frame = self.draw_stats(frame, detections, tracked_vehicles, processing_time)
                
                # Update FPS
                self.update_fps()
                
                # Display frame
                cv2.imshow('Raspberry Pi Traffic Monitoring', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = int(time.time())
                    filename = f"traffic_screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"Screenshot saved: {filename}")
        
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        
        finally:
            self.running = False
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Demo stopped")

def main():
    parser = argparse.ArgumentParser(description='Raspberry Pi Traffic Monitoring Demo')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (0 for webcam, path for video file)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Path to YOLO model file')
    
    args = parser.parse_args()
    
    # Convert source to int if it's a digit
    source = int(args.source) if args.source.isdigit() else args.source
    
    # Create and run demo
    demo = RPiTrafficDemo(source=source, model_path=args.model)
    demo.run()

if __name__ == "__main__":
    main()
