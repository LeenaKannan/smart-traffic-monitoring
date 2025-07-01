# app/services/rpi_stream_processor.py
import cv2
import asyncio
import numpy as np
from typing import Dict, List, Optional
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
from app.models.rpi_vehicle_detector import RPiVehicleDetector
from app.models.vehicle_tracker import VehicleTracker
from app.models.violation_detector import ViolationDetector
from app.models.traffic_density_estimator import TrafficDensityEstimator
from app.config.settings import settings

logger = logging.getLogger(__name__)

class RPiStreamProcessor:
    def __init__(self):
        """Raspberry Pi optimized stream processor"""
        self.vehicle_detector = RPiVehicleDetector(
            model_path=settings.YOLO_MODEL_PATH,
            device="cpu"
        )
        self.vehicle_tracker = VehicleTracker()
        self.violation_detector = ViolationDetector()
        self.density_estimator = TrafficDensityEstimator()
        
        self.active_streams = {}
        self.frame_queues = {}
        self.result_queues = {}
        self.processing_threads = {}
        
        # Optimized for Raspberry Pi
        self.max_workers = 2  # Limited for Pi 4
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Performance monitoring
        self.performance_stats = {}
    
    async def start_stream(self, camera_id: str, stream_url: str) -> Dict:
        """Start processing video stream optimized for Raspberry Pi"""
        try:
            # Configure camera for Raspberry Pi
            cap = cv2.VideoCapture(stream_url)
            
            if not cap.isOpened():
                raise Exception(f"Cannot open stream: {stream_url}")
            
            # Optimize capture settings for Pi 4
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for Pi 4
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer
            
            self.active_streams[camera_id] = {
                'capture': cap,
                'url': stream_url,
                'status': 'active',
                'start_time': time.time()
            }
            
            # Initialize queues with smaller sizes for Pi
            self.frame_queues[camera_id] = queue.Queue(maxsize=5)
            self.result_queues[camera_id] = queue.Queue(maxsize=50)
            
            # Start processing thread
            processing_thread = threading.Thread(
                target=self._process_stream_rpi,
                args=(camera_id,),
                daemon=True
            )
            processing_thread.start()
            self.processing_threads[camera_id] = processing_thread
            
            logger.info(f"Started RPi stream processing for camera {camera_id}")
            
            return {
                'camera_id': camera_id,
                'status': 'started',
                'stream_url': stream_url,
                'optimized_for': 'raspberry_pi_4'
            }
            
        except Exception as e:
            logger.error(f"Error starting RPi stream {camera_id}: {e}")
            return {
                'camera_id': camera_id,
                'status': 'error',
                'error': str(e)
            }
    
    def _process_stream_rpi(self, camera_id: str):
        """Raspberry Pi optimized stream processing loop"""
        cap = self.active_streams[camera_id]['capture']
        frame_count = 0
        last_process_time = time.time()
        
        while camera_id in self.active_streams:
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame from camera {camera_id}")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            current_time = time.time()
            
            # Process every 3rd frame for Pi 4 (instead of every frame)
            if frame_count % 3 != 0:
                continue
            
            # Skip if processing is too slow
            if current_time - last_process_time < 0.2:  # Max 5 FPS processing
                continue
            
            try:
                # Process frame
                result = self._process_frame_rpi(frame, camera_id)
                
                # Add result to queue
                if not self.result_queues[camera_id].full():
                    self.result_queues[camera_id].put(result)
                else:
                    # Remove oldest result if queue is full
                    try:
                        self.result_queues[camera_id].get_nowait()
                        self.result_queues[camera_id].put(result)
                    except queue.Empty:
                        pass
                
                last_process_time = current_time
                
            except Exception as e:
                logger.error(f"Error processing frame for camera {camera_id}: {e}")
        
        cap.release()
        logger.info(f"RPi stream processing stopped for camera {camera_id}")
    
    def _process_frame_rpi(self, frame: np.ndarray, camera_id: str) -> Dict:
        """Process single frame optimized for Raspberry Pi"""
        start_time = time.time()
        
        # Vehicle detection with lower confidence for Pi
        detections = self.vehicle_detector.detect_vehicles(
            frame, 
            confidence_threshold=0.4,  # Lower for Pi
            img_size=416  # Smaller image size
        )
        
        # Vehicle tracking
        tracked_vehicles = self.vehicle_tracker.update(detections)
        
        # Traffic density estimation
        density_data = self.density_estimator.estimate_density(
            detections, camera_id, frame.shape
        )
        
        # Basic violation detection (simplified for Pi)
        violations = []
        signal_state = self._get_signal_state(camera_id)
        
        if signal_state == "red":
            signal_violations = self.violation_detector.detect_signal_violation(
                detections, signal_state, camera_id
            )
            violations.extend(signal_violations)
        
        processing_time = time.time() - start_time
        
        # Update performance stats
        if camera_id not in self.performance_stats:
            self.performance_stats[camera_id] = []
        
        self.performance_stats[camera_id].append(processing_time)
        if len(self.performance_stats[camera_id]) > 50:
            self.performance_stats[camera_id] = self.performance_stats[camera_id][-50:]
        
        result = {
            'camera_id': camera_id,
            'timestamp': time.time(),
            'processing_time': processing_time,
            'detections': detections,
            'tracked_vehicles': dict(tracked_vehicles),
            'density_data': density_data,
            'violations': violations,
            'frame_shape': frame.shape,
            'detector_stats': self.vehicle_detector.get_performance_stats()
        }
        
        return result
    
    def _get_signal_state(self, camera_id: str) -> str:
        """Get current signal state - placeholder"""
        return "green"
    
    def get_performance_stats(self, camera_id: str) -> Dict:
        """Get processing performance statistics"""
        if camera_id not in self.performance_stats:
            return {"error": "No stats available"}
        
        times = self.performance_stats[camera_id]
        avg_time = np.mean(times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            "avg_processing_time_ms": round(avg_time * 1000, 2),
            "avg_fps": round(avg_fps, 2),
            "detector_stats": self.vehicle_detector.get_performance_stats()
        }
