# app/models/traffic_density_estimator.py
import numpy as np
from typing import List, Dict, Tuple
import cv2

class TrafficDensityEstimator:
    def __init__(self):
        self.lane_configs = {}  # Lane configuration for each camera
        self.density_history = {}  # Historical density data
        
    def estimate_density(self, detections: List[Dict], 
                        camera_id: str, 
                        frame_shape: Tuple[int, int]) -> Dict:
        """
        Estimate traffic density considering Indian traffic patterns
        """
        height, width = frame_shape[:2]
        
        # Create density grid
        grid_size = 50
        grid_rows = height // grid_size
        grid_cols = width // grid_size
        density_grid = np.zeros((grid_rows, grid_cols))
        
        # Vehicle weights for Indian traffic
        vehicle_weights = {
            'car': 1.0,
            'bus': 2.5,
            'truck': 2.0,
            'motorcycle': 0.5,
            'auto_rickshaw': 0.8,
            'bicycle': 0.3,
            'person': 0.2
        }
        
        total_density = 0
        vehicle_counts = {}
        
        for detection in detections:
            center_x, center_y = detection['center']
            vehicle_class = detection['class_name']
            
            # Calculate grid position
            grid_x = min(int(center_x // grid_size), grid_cols - 1)
            grid_y = min(int(center_y // grid_size), grid_rows - 1)
            
            # Add weighted density
            weight = vehicle_weights.get(vehicle_class, 1.0)
            density_grid[grid_y, grid_x] += weight
            total_density += weight
            
            # Count vehicles by type
            vehicle_counts[vehicle_class] = vehicle_counts.get(vehicle_class, 0) + 1
        
        # Calculate congestion level
        area_km2 = (width * height) / (1000000)  # Approximate area
        density_per_km2 = total_density / area_km2 if area_km2 > 0 else 0
        
        congestion_level = self._calculate_congestion_level(density_per_km2)
        
        density_data = {
            'camera_id': camera_id,
            'total_vehicles': len(detections),
            'vehicle_counts': vehicle_counts,
            'density_grid': density_grid.tolist(),
            'total_density': total_density,
            'density_per_km2': density_per_km2,
            'congestion_level': congestion_level,
            'timestamp': time.time()
        }
        
        # Store in history
        if camera_id not in self.density_history:
            self.density_history[camera_id] = []
        
        self.density_history[camera_id].append(density_data)
        
        # Keep only last 100 records
        if len(self.density_history[camera_id]) > 100:
            self.density_history[camera_id] = self.density_history[camera_id][-100:]
        
        return density_data
    
    def _calculate_congestion_level(self, density_per_km2: float) -> str:
        """
        Calculate congestion level based on Indian traffic standards
        """
        if density_per_km2 < 50:
            return "low"
        elif density_per_km2 < 150:
            return "moderate"
        elif density_per_km2 < 300:
            return "high"
        else:
            return "severe"
    
    def predict_congestion(self, camera_id: str, 
                          prediction_minutes: int = 15) -> Dict:
        """
        Predict future congestion based on historical patterns
        """
        if camera_id not in self.density_history:
            return {"prediction": "insufficient_data"}
        
        history = self.density_history[camera_id]
        if len(history) < 10:
            return {"prediction": "insufficient_data"}
        
        # Simple trend analysis
        recent_densities = [h['density_per_km2'] for h in history[-10:]]
        trend = np.polyfit(range(len(recent_densities)), recent_densities, 1)[0]
        
        current_density = recent_densities[-1]
        predicted_density = current_density + (trend * prediction_minutes)
        predicted_level = self._calculate_congestion_level(predicted_density)
        
        return {
            'current_density': current_density,
            'predicted_density': max(0, predicted_density),
            'predicted_level': predicted_level,
            'trend': 'increasing' if trend > 0 else 'decreasing',
            'confidence': min(len(history) / 50.0, 1.0)
        }