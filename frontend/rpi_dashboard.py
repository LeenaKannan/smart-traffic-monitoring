# frontend/rpi_dashboard.py
import streamlit as st
import cv2
import numpy as np
import time
import threading
import queue
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.models.rpi_vehicle_detector import RPiVehicleDetector
from app.models.vehicle_tracker import VehicleTracker

st.set_page_config(
    page_title="🚦 RPi Traffic Monitor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

class RPiDashboard:
    def __init__(self):
        self.detector = None
        self.tracker = None
        self.camera = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=5)
        
    def initialize_components(self):
        """Initialize AI components"""
        if self.detector is None:
            with st.spinner("Loading AI models..."):
                self.detector = RPiVehicleDetector()
                self.tracker = VehicleTracker()
        
    def start_camera(self, source=0):
        """Start camera capture"""
        if self.camera is None:
            self.camera = cv2.VideoCapture(source)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 15)
            
        return self.camera.isOpened()
    
    def capture_frames(self):
        """Capture frames in background thread"""
        while self.running and self.camera:
            ret, frame = self.camera.read()
            if ret:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
            time.sleep(0.1)
    
    def process_frame(self, frame):
        """Process frame with AI models"""
        # Detect vehicles
        detections = self.detector.detect_vehicles(frame, confidence_threshold=0.4)
        
        # Track vehicles
        tracked_vehicles = self.tracker.update(detections)
        
        # Draw results
        processed_frame = self.draw_results(frame, detections, tracked_vehicles)
        
        return processed_frame, detections, tracked_vehicles
    
    def draw_results(self, frame, detections, tracked_vehicles):
        """Draw detection and tracking results"""
        # Draw detections
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = bbox
            
            # Color coding
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
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw tracking info
        for obj_id, vehicle_data in tracked_vehicles.items():
            centroid = vehicle_data['centroid']
            
            # Draw centroid
            cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 5, (0, 0, 255), -1)
            
            # Draw ID
            cv2.putText(frame, f"ID:{obj_id}", 
                       (int(centroid[0]) - 20, int(centroid[1]) - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        return frame

def main():
    st.title("🚦 Raspberry Pi Traffic Monitoring Dashboard")
    st.markdown("Real-time AI-powered traffic monitoring optimized for Raspberry Pi 4")
    
    dashboard = RPiDashboard()
    
    # Sidebar controls
    st.sidebar.header("🔧 Controls")
    
    # Camera source selection
    camera_source = st.sidebar.selectbox(
        "📷 Camera Source",
        options=[0, 1, "USB Camera", "Pi Camera"],
        index=0
    )
    
    # Convert camera source
    if camera_source == "USB Camera":
        camera_source = 0
    elif camera_source == "Pi Camera":
        camera_source = 0  # Usually the same on Pi
    
    # Control buttons
    col1, col2 = st.sidebar.columns(2)
    
    start_button = col1.button("▶️ Start", type="primary")
    stop_button = col2.button("⏹️ Stop")
    
    # Settings
    st.sidebar.header("⚙️ Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.4, 
        step=0.1
    )
    
    show_stats = st.sidebar.checkbox("📊 Show Statistics", value=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Feed")
        video_placeholder = st.empty()
    
    with col2:
        st.subheader("📊 Statistics")
        stats_placeholder = st.empty()
        
        if show_stats:
            performance_placeholder = st.empty()
    
    # Initialize session state
    if 'monitoring' not in st.session_state:
        st.session_state.monitoring = False
    
    # Handle start/stop
    if start_button and not st.session_state.monitoring:
        st.session_state.monitoring = True
        dashboard.initialize_components()
        
        if dashboard.start_camera(camera_source):
            st.success("✅ Camera started successfully!")
            dashboard.running = True
            
            # Start capture thread
            capture_thread = threading.Thread(target=dashboard.capture_frames, daemon=True)
            capture_thread.start()
        else:
            st.error("❌ Failed to start camera")
            st.session_state.monitoring = False
    
    if stop_button and st.session_state.monitoring:
        st.session_state.monitoring = False
        dashboard.running = False
        if dashboard.camera:
            dashboard.camera.release()
            dashboard.camera = None
        st.info("⏹️ Monitoring stopped")
    
    # Main processing loop
    if st.session_state.monitoring and dashboard.running:
        try:
            if not dashboard.frame_queue.empty():
                frame = dashboard.frame_queue.get()
                
                # Process frame
                processed_frame, detections, tracked_vehicles = dashboard.process_frame(frame)
                
                # Convert BGR to RGB for Streamlit
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Display video
                video_placeholder.image(processed_frame_rgb, channels="RGB", use_column_width=True)
                
                # Display statistics
                with stats_placeholder.container():
                    st.metric("🚗 Total Vehicles", len(detections))
                    st.metric("🎯 Tracked Objects", len(tracked_vehicles))
                    
                    # Vehicle type breakdown
                    vehicle_counts = {}
                    for detection in detections:
                        vehicle_type = detection['class_name']
                        vehicle_counts[vehicle_type] = vehicle_counts.get(vehicle_type, 0) + 1
                    
                    if vehicle_counts:
                        st.write("**Vehicle Types:**")
                        for vehicle_type, count in vehicle_counts.items():
                            st.write(f"• {vehicle_type}: {count}")
                
                # Performance stats
                if show_stats and 'performance_placeholder' in locals():
                    with performance_placeholder.container():
                        detector_stats = dashboard.detector.get_performance_stats()
                        st.write("**Performance:**")
                        st.write(f"• FPS: {detector_stats.get('avg_fps', 0):.1f}")
                        st.write(f"• Inference: {detector_stats.get('avg_inference_time', 0):.1f}ms")
        
        except Exception as e:
            st.error(f"Error: {e}")
    
    # Auto-refresh
    if st.session_state.monitoring:
        time.sleep(0.1)
        st.rerun()

if __name__ == "__main__":
    main()
