# app/models/violation_detector.py
class ViolationDetector:
    def __init__(self):
        self.signal_zones = {}  # Define signal zones for each camera
        self.lane_boundaries = {}  # Define lane boundaries
        
    def detect_signal_violation(self, detections: List[Dict], 
                               signal_state: str, 
                               camera_id: str) -> List[Dict]:
        """
        Detect red light violations specific to Indian traffic patterns
        """
        violations = []
        
        if signal_state != "red":
            return violations
        
        signal_zone = self.signal_zones.get(camera_id, {})
        if not signal_zone:
            return violations
        
        for detection in detections:
            bbox = detection['bbox']
            center = detection['center']
            
            # Check if vehicle is in signal zone during red light
            if self._is_in_zone(center, signal_zone):
                violation = {
                    'type': 'red_light_violation',
                    'vehicle_class': detection['class_name'],
                    'bbox': bbox,
                    'confidence': detection['confidence'],
                    'timestamp': time.time(),
                    'camera_id': camera_id
                }
                violations.append(violation)
        
        return violations
    
    def detect_helmet_violation(self, frame: np.ndarray, 
                               detections: List[Dict]) -> List[Dict]:
        """
        Detect helmet violations for motorcycles - critical for Indian traffic
        """
        violations = []
        
        for detection in detections:
            if detection['class_name'] == 'motorcycle':
                bbox = detection['bbox']
                # Extract rider region
                rider_region = self._extract_rider_region(frame, bbox)
                
                # Simple helmet detection (can be enhanced with dedicated model)
                has_helmet = self._detect_helmet_simple(rider_region)
                
                if not has_helmet:
                    violation = {
                        'type': 'helmet_violation',
                        'vehicle_class': 'motorcycle',
                        'bbox': bbox,
                        'timestamp': time.time()
                    }
                    violations.append(violation)
        
        return violations
    
    def _is_in_zone(self, point: Tuple[float, float], zone: Dict) -> bool:
        """Check if point is within defined zone"""
        x, y = point
        return (zone['x1'] <= x <= zone['x2'] and 
                zone['y1'] <= y <= zone['y2'])
    
    def _extract_rider_region(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Extract rider region from motorcycle detection"""
        x1, y1, x2, y2 = bbox
        # Focus on upper portion where helmet would be
        rider_y1 = y1
        rider_y2 = y1 + int((y2 - y1) * 0.4)
        return frame[rider_y1:rider_y2, x1:x2]
    
    def _detect_helmet_simple(self, rider_region: np.ndarray) -> bool:
        """Simple helmet detection using color and shape analysis"""
        if rider_region.size == 0:
            return False
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(rider_region, cv2.COLOR_BGR2HSV)
        
        # Define helmet color ranges (common helmet colors in India)
        helmet_ranges = [
            ([0, 0, 0], [180, 255, 50]),      # Black
            ([100, 50, 50], [130, 255, 255]), # Blue
            ([0, 50, 50], [10, 255, 255]),    # Red
            ([25, 50, 50], [35, 255, 255])    # Yellow
        ]
        
        total_helmet_pixels = 0
        for lower, upper in helmet_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            total_helmet_pixels += cv2.countNonZero(mask)
        
        # If significant helmet-colored pixels found
        helmet_ratio = total_helmet_pixels / (rider_region.shape[0] * rider_region.shape[1])
        return helmet_ratio > 0.15  # Threshold can be tuned