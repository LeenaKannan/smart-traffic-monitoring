# app/models/license_plate_recognizer.py
import cv2
import numpy as np
import pytesseract
import re
from typing import List, Dict, Optional

class IndianLicensePlateRecognizer:
    def __init__(self):
        # Indian license plate patterns
        self.plate_patterns = [
            r'^[A-Z]{2}\s?[0-9]{2}\s?[A-Z]{1,2}\s?[0-9]{4}$',  # Standard format
            r'^[A-Z]{2}\s?[0-9]{1,2}\s?[A-Z]{1,2}\s?[0-9]{4}$',  # Older format
            r'^[0-9]{2}\s?BH\s?[0-9]{4}\s?[A-Z]{2}$'  # Bharat series
        ]
        
        # Configure Tesseract for Indian plates
        self.tesseract_config = '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    def detect_and_recognize_plates(self, frame: np.ndarray, 
                                   vehicle_detections: List[Dict]) -> List[Dict]:
        """
        Detect and recognize license plates from vehicle detections
        """
        plate_results = []
        
        for detection in vehicle_detections:
            if detection['class_name'] in ['car', 'bus', 'truck', 'auto_rickshaw']:
                bbox = detection['bbox']
                vehicle_region = self._extract_vehicle_region(frame, bbox)
                
                # Detect license plate region
                plate_regions = self._detect_plate_regions(vehicle_region)
                
                for plate_region in plate_regions:
                    # Preprocess plate region
                    processed_plate = self._preprocess_plate(plate_region)
                    
                    # Recognize text
                    plate_text = self._recognize_text(processed_plate)
                    
                    if self._validate_indian_plate(plate_text):
                        plate_result = {
                            'vehicle_bbox': bbox,
                            'vehicle_class': detection['class_name'],
                            'plate_text': plate_text,
                            'confidence': detection['confidence'],
                            'timestamp': time.time()
                        }
                        plate_results.append(plate_result)
        
        return plate_results
    
    def _extract_vehicle_region(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Extract vehicle region from frame"""
        x1, y1, x2, y2 = bbox
        return frame[y1:y2, x1:x2]
    
    def _detect_plate_regions(self, vehicle_region: np.ndarray) -> List[np.ndarray]:
        """
        Detect potential license plate regions using morphological operations
        """
        if vehicle_region.size == 0:
            return []
        
        gray = cv2.cvtColor(vehicle_region, cv2.COLOR_BGR2GRAY)
        
        # Apply morphological operations to find rectangular regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # Gradient
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        
        # Combine gradients
        gradient = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
        gradient = np.uint8(gradient)
        
        # Threshold and morphological operations
        _, thresh = cv2.threshold(gradient, 50, 255, cv2.THRESH_BINARY)
        
        # Morphological closing
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        plate_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter based on aspect ratio and size (typical for Indian plates)
            aspect_ratio = w / h
            if 2.0 <= aspect_ratio <= 5.0 and w > 50 and h > 15:
                plate_region = vehicle_region[y:y+h, x:x+w]
                plate_regions.append(plate_region)
        
        return plate_regions
    
    def _preprocess_plate(self, plate_region: np.ndarray) -> np.ndarray:
        """
        Preprocess license plate for better OCR recognition
        """
        if plate_region.size == 0:
            return plate_region
        
        # Convert to grayscale
        gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        
        # Resize for better OCR
        height, width = gray.shape
        if height < 40:
            scale_factor = 40 / height
            new_width = int(width * scale_factor)
            gray = cv2.resize(gray, (new_width, 40))
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _recognize_text(self, processed_plate: np.ndarray) -> str:
        """
        Recognize text from preprocessed license plate
        """
        try:
            text = pytesseract.image_to_string(processed_plate, config=self.tesseract_config)
            # Clean up the text
            text = re.sub(r'[^A-Z0-9\s]', '', text.upper())
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        except Exception:
            return ""
    
    def _validate_indian_plate(self, plate_text: str) -> bool:
        """
        Validate if the recognized text matches Indian license plate patterns
        """
        if not plate_text or len(plate_text) < 8:
            return False
        
        # Remove spaces for pattern matching
        clean_text = plate_text.replace(' ', '')
        
        for pattern in self.plate_patterns:
            if re.match(pattern, clean_text):
                return True
        
        return False