# training/indian_dataset_collector.py
import cv2
import requests
import time
from pathlib import Path
import logging

class IndianTrafficDataCollector:
    def __init__(self):
        # Indian traffic camera URLs (public streams)
        self.traffic_sources = [
            "https://www.youtube.com/watch?v=INDIAN_TRAFFIC_LIVE_1",
            "https://www.youtube.com/watch?v=INDIAN_TRAFFIC_LIVE_2",
            # Add more Indian traffic live streams
        ]
        
    def collect_frames(self, output_dir="data/raw/indian_traffic", 
                      frames_per_source=1000):
        """Collect frames from Indian traffic sources"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, source in enumerate(self.traffic_sources):
            try:
                # Use youtube-dl or similar to get actual stream URL
                cap = cv2.VideoCapture(source)
                frame_count = 0
                
                while cap.isOpened() and frame_count < frames_per_source:
                    ret, frame = cap.read()
                    if ret:
                        # Save frame with timestamp
                        timestamp = int(time.time())
                        filename = f"indian_traffic_{i}_{frame_count:06d}_{timestamp}.jpg"
                        cv2.imwrite(str(output_path / filename), frame)
                        frame_count += 1
                        
                        # Skip frames for diversity
                        for _ in range(30):  # Skip 30 frames (~1 second)
                            cap.read()
                
                cap.release()
                print(f"✅ Collected {frame_count} frames from source {i}")
                
            except Exception as e:
                print(f"❌ Error with source {i}: {e}")
