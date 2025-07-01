# app/services/stream_processor.py
import cv2
import asyncio
import numpy as np
from typing import Dict, List, Optional
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

from app.models.vehicle_detector import VehicleDetector
from app.models.violation_detector import ViolationDetector
from app.models.traffic_density_estimator import TrafficDensityEstimator
from app.models.license_plate_recognizer import IndianLicensePlateRecognizer
from app.config.settings import settings

logger = logging.getLogger(__name__)

class StreamProcessor:
    def __init__(self):
        self.vehicle_detector = VehicleDetector(settings.YOLO_MODEL_PATH)
        self.violation_detector = ViolationDetector()
        self.density_estimator = TrafficDensityEstimator()
        self.plate_recognizer = IndianLicensePlateRecognizer()
        
        self.active_streams = {}
        self.frame_queues = {}
        self.result_queues = {}
        self.processing_threads = {}
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def start_stream(self, camera_id: str, stream_url: str) -> Dict:
        """
        Start processing video stream from camera
        """
        try:
            cap = cv2.VideoCapture(stream_url)
            
            if not cap.isOpened():
                raise Exception(f"Cannot open stream: {stream_url}")
            
            self.active_streams[camera_id] = {
                'capture': cap,
                'url': stream_url,
                'status': 'active',
                'start_time': time.time()
            }
            
            # Initialize queues
            self.frame_queues[camera_id] = queue.Queue(maxsize=10)
            self.result_queues[camera_id] = queue.Queue(maxsize=100)
            
            # Start processing thread
            processing_thread = threading.Thread(
                target=self._process_stream,
                args=(camera_id,),
                daemon=True
            )
            processing_thread.start()
            self.processing_threads[camera_id] = processing_thread
            
            logger.info(f"Started stream processing for camera {camera_id}")
            
            return {
                'camera_id': camera_id,
                'status': 'started',
                'stream_url': stream_url
            }
            
        except Exception as e:
            logger.error(f"Error starting stream {camera_id}: {e}")
            return {
                'camera_id': camera_id,
                'status': 'error',
                'error': str(e)
            }
    
    def _process_stream(self, camera_id: str):
        """
        Main stream processing loop
        """
        cap = self.active_streams[camera_id]['capture']
        frame_count = 0
        
        while camera_id in self.active_streams:
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Failed to read frame from camera {camera_id}")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Skip frames based on FRAME_SKIP setting
            if frame_count % (settings.FRAME_SKIP + 1) != 0:
                continue
            
            try:
                # Add frame to processing queue
                if not self.frame_queues[camera_id].full():
                    self.frame_queues[camera_id].put((frame.copy(), time.time()))
                
                # Process frame
                result = self._process_frame(frame, camera_id)
                
                # Add result to queue
                if not self.result_queues[camera_id].full():
                    self.result_queues[camera_id].put(result)
                
            except Exception as e:
                logger.error(f"Error processing frame for camera {camera_id}: {e}")
        
        cap.release()
        logger.info(f"Stream processing stopped for camera {camera_id}")
    
    def _process_frame(self, frame: np.ndarray, camera_id: str) -> Dict:
        """
        Process single frame through AI pipeline
        """
        start_time = time.time()
        
        # Vehicle detection
        detections = self.vehicle_detector.detect_vehicles(
            frame, settings.CONFIDENCE_THRESHOLD
        )
        
        # Traffic density estimation
        density_data = self.density_estimator.estimate_density(
            detections, camera_id, frame.shape
        )
        
        # Violation detection
        violations = []
        
        # Signal violations (assuming we have signal state)
        signal_state = self._get_signal_state(camera_id)  # Implement based on your setup
        signal_violations = self.violation_detector.detect_signal_violation(
            detections, signal_state, camera_id
        )
        violations.extend(signal_violations)
        
        # Helmet violations
        helmet_violations = self.violation_detector.detect_helmet_violation(
            frame, detections
        )
        violations.extend(helmet_violations)
        
        # License plate recognition (for violations)
        plate_results = []
        if violations:
            plate_results = self.plate_recognizer.detect_and_recognize_plates(
                frame, detections
            )
        
        processing_time = time.time() - start_time
        
        result = {
            'camera_id': camera_id,
            'timestamp': time.time(),
            'processing_time': processing_time,
            'detections': detections,
            'density_data': density_data,
            'violations': violations,
            'license_plates': plate_results,
            'frame_shape': frame.shape
        }
        
        return result
    
    def _get_signal_state(self, camera_id: str) -> str:
        """
        Get current signal state for camera
        This should be integrated with your traffic controller
        """
        # Placeholder - integrate with TrafficController
        return "green"  # or "red", "yellow"
    
    async def stop_stream(self, camera_id: str) -> Dict:
        """
        Stop processing stream
        """
        if camera_id not in self.active_streams:
            return {'error': 'Stream not found'}
        
        # Remove from active streams
        stream_info = self.active_streams.pop(camera_id)
        stream_info['capture'].release()
        
        # Clean up queues
        if camera_id in self.frame_queues:
            del self.frame_queues[camera_id]
        if camera_id in self.result_queues:
            del self.result_queues[camera_id]
        
        logger.info(f"Stopped stream processing for camera {camera_id}")
        
        return {
            'camera_id': camera_id,
            'status': 'stopped'
        }
    
    def get_latest_results(self, camera_id: str, count: int = 10) -> List[Dict]:
        """
        Get latest processing results for camera
        """
        if camera_id not in self.result_queues:
            return []
        
        results = []
        result_queue = self.result_queues[camera_id]
        
        # Get up to 'count' latest results
        for _ in range(min(count, result_queue.qsize())):
            try:
                result = result_queue.get_nowait()
                results.append(result)
            except queue.Empty:
                break
        
        return results
    
    def get_stream_status(self, camera_id: str) -> Dict:
        """
        Get stream status information
        """
        if camera_id not in self.active_streams:
            return {'error': 'Stream not found'}
        
        stream_info = self.active_streams[camera_id]
        uptime = time.time() - stream_info['start_time']
        
        return {
            'camera_id': camera_id,
            'status': stream_info['status'],
            'uptime': uptime,
            'queue_size': self.result_queues[camera_id].qsize() if camera_id in self.result_queues else 0
        }
