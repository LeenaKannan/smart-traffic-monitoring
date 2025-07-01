# app/models/vehicle_tracker.py
import numpy as np
from collections import OrderedDict
import cv2
from typing import List, Dict, Tuple
import time

class VehicleTracker:
    def __init__(self, max_disappeared=30, max_distance=50):
        """
        Initialize vehicle tracker with DeepSORT-like functionality
        Optimized for Raspberry Pi 4
        """
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def register(self, centroid: Tuple[int, int], detection: Dict) -> int:
        """Register a new vehicle"""
        self.objects[self.next_object_id] = {
            'centroid': centroid,
            'detection': detection,
            'last_seen': time.time(),
            'trajectory': [centroid],
            'speed': 0.0,
            'direction': None
        }
        self.disappeared[self.next_object_id] = 0
        object_id = self.next_object_id
        self.next_object_id += 1
        return object_id
    
    def deregister(self, object_id: int):
        """Remove a vehicle from tracking"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        """
        Update tracked vehicles with new detections
        Returns: {object_id: vehicle_data}
        """
        if len(detections) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        # Initialize centroids array
        input_centroids = np.array([det['center'] for det in detections])
        
        if len(self.objects) == 0:
            # Register all detections as new objects
            for i, detection in enumerate(detections):
                self.register(input_centroids[i], detection)
        else:
            # Get existing centroids
            object_centroids = np.array([obj['centroid'] for obj in self.objects.values()])
            object_ids = list(self.objects.keys())
            
            # Compute distance matrix
            D = self._compute_distance_matrix(object_centroids, input_centroids)
            
            # Assign detections to existing objects
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                
                if D[row, col] > self.max_distance:
                    continue
                
                # Update existing object
                object_id = object_ids[row]
                old_centroid = self.objects[object_id]['centroid']
                new_centroid = input_centroids[col]
                
                # Calculate speed and direction
                speed = self._calculate_speed(old_centroid, new_centroid)
                direction = self._calculate_direction(old_centroid, new_centroid)
                
                self.objects[object_id]['centroid'] = tuple(new_centroid)
                self.objects[object_id]['detection'] = detections[col]
                self.objects[object_id]['last_seen'] = time.time()
                self.objects[object_id]['trajectory'].append(tuple(new_centroid))
                self.objects[object_id]['speed'] = speed
                self.objects[object_id]['direction'] = direction
                
                # Keep trajectory length manageable
                if len(self.objects[object_id]['trajectory']) > 10:
                    self.objects[object_id]['trajectory'] = self.objects[object_id]['trajectory'][-10:]
                
                self.disappeared[object_id] = 0
                
                used_row_indices.add(row)
                used_col_indices.add(col)
            
            # Handle unmatched detections and objects
            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)
            
            if D.shape[0] >= D.shape[1]:
                # More objects than detections
                for row in unused_row_indices:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                # More detections than objects
                for col in unused_col_indices:
                    self.register(input_centroids[col], detections[col])
        
        return self.objects
    
    def _compute_distance_matrix(self, object_centroids: np.ndarray, 
                                input_centroids: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance matrix"""
        return np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids, axis=2)
    
    def _calculate_speed(self, old_centroid: Tuple[int, int], 
                        new_centroid: Tuple[int, int], 
                        time_diff: float = 0.1, 
                        pixels_per_meter: float = 10.0) -> float:
        """Calculate vehicle speed in km/h"""
        distance_pixels = np.sqrt(
            (new_centroid[0] - old_centroid[0])**2 + 
            (new_centroid[1] - old_centroid[1])**2
        )
        distance_meters = distance_pixels / pixels_per_meter
        speed_mps = distance_meters / time_diff
        return speed_mps * 3.6  # Convert to km/h
    
    def _calculate_direction(self, old_centroid: Tuple[int, int], 
                           new_centroid: Tuple[int, int]) -> str:
        """Calculate movement direction"""
        dx = new_centroid[0] - old_centroid[0]
        dy = new_centroid[1] - old_centroid[1]
        
        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "down" if dy > 0 else "up"
