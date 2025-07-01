#!/bin/bash
# scripts/rpi_setup.sh

echo "🚦 Setting up Smart Traffic Monitoring System for Raspberry Pi 4"

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo; then
    echo "⚠️  This script is optimized for Raspberry Pi"
fi

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo "📥 Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-opencv \
    libopencv-dev \
    libatlas-base-dev \
    libjasper-dev \
    libqtgui4 \
    libqt4-test \
    libhdf5-dev \
    libhdf5-serial-dev \
    libatlas-base-dev \
    libjasper-dev \
    libqtgui4 \
    libqt4-test \
    git \
    cmake \
    build-essential \
    pkg-config \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libfontconfig1-dev \
    libcairo2-dev \
    libgdk-pixbuf2.0-dev \
    libpango1.0-dev \
    libgtk2.0-dev \
    libgtk-3-dev

# Enable camera
echo "📷 Enabling camera interface..."
sudo raspi-config nonint do_camera 0

# Increase GPU memory split
echo "🔧 Optimizing GPU memory..."
sudo raspi-config nonint do_memory_split 128

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch for Raspberry Pi (CPU only)
echo "🔥 Installing PyTorch for Raspberry Pi..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
echo "📦 Installing Python dependencies..."
pip install -r requirements_rpi.txt

# Create requirements_rpi.txt
cat > requirements_rpi.txt << EOL
# Raspberry Pi optimized requirements
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Computer Vision (Pi optimized)
opencv-python==4.8.1.78
ultralytics==8.0.206
numpy==1.24.3
Pillow==10.1.0

# Lightweight alternatives
streamlit==1.28.1
matplotlib==3.8.2
pandas==2.1.4

# Utilities
python-dotenv==1.0.0
click==8.1.7
tqdm==4.66.1

# Testing
pytest==7.4.3
EOL

pip install -r requirements_rpi.txt

# Download YOLOv8 nano model (smallest for Pi)
echo "🤖 Downloading YOLOv8 nano model..."
python3 -c "
from ultralytics import YOLO
import os
os.makedirs('data/models', exist_ok=True)
model = YOLO('yolov8n.pt')
print('✅ YOLOv8 nano model downloaded')
"

# Create directories
echo "📁 Creating project directories..."
mkdir -p data/{raw,processed,models}
mkdir -p logs
mkdir -p demo/videos

# Set up camera test
echo "📷 Setting up camera test..."
cat > test_camera.py << EOL
#!/usr/bin/env python3
import cv2
import time

def test_camera():
    print("Testing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Camera not found")
        return False
    
    # Test capture
    ret, frame = cap.read()
    if ret:
        print("✅ Camera working")
        print(f"Frame shape: {frame.shape}")
        
        # Save test image
        cv2.imwrite('camera_test.jpg', frame)
        print("📸 Test image saved as camera_test.jpg")
    else:
        print("❌ Failed to capture frame")
        return False
    
    cap.release()
    return True

if __name__ == "__main__":
    test_camera()
EOL

chmod +x test_camera.py

# Create environment file for Pi
echo "⚙️ Creating Raspberry Pi environment configuration..."
cat > .env << EOL
# Raspberry Pi Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# AI Models (optimized for Pi)
YOLO_MODEL_PATH=yolov8n.pt

# Video Processing (Pi optimized)
CONFIDENCE_THRESHOLD=0.4
FRAME_SKIP=2
MAX_RESOLUTION=640x480

# Performance settings
MAX_WORKERS=2
PROCESSING_FPS=5

# Indian Vehicle Classes
INDIAN_VEHICLE_CLASSES=["car", "motorcycle", "bus", "truck", "auto_rickshaw", "bicycle", "person"]
EOL

# Create systemd service for auto-start
echo "🔧 Creating systemd service..."
sudo tee /etc/systemd/system/traffic-monitor.service > /dev/null << EOL
[Unit]
Description=Smart Traffic Monitoring System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/smart-traffic-monitoring
Environment=PATH=/home/pi/smart-traffic-monitoring/venv/bin
ExecStart=/home/pi/smart-traffic-monitoring/venv/bin/python demo/rpi_demo.py
Restart=always

[Install]
WantedBy=multi-user.target
EOL

# Enable service (but don't start yet)
sudo systemctl enable traffic-monitor.service

echo "✅ Raspberry Pi setup completed successfully!"
echo ""
echo "🚀 To test the system:"
echo "1. Test camera: python3 test_camera.py"
echo "2. Run demo: python3 demo/rpi_demo.py"
echo "3. Start service: sudo systemctl start traffic-monitor.service"
echo ""
echo "📊 For web dashboard:"
echo "streamlit run frontend/rpi_dashboard.py"
echo ""
echo "🔄 Reboot recommended to ensure all changes take effect"
