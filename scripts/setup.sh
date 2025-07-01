#!/bin/bash
# scripts/setup.sh

echo "🚦 Setting up Smart Traffic Monitoring System"

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )(.+)')
if [[ -z "$python_version" ]]; then
    echo "❌ Python 3.7+ is required"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create directories
echo "📁 Creating directories..."
mkdir -p data/{raw,processed,models}
mkdir -p logs
mkdir -p frontend/dist

# Download pre-trained models
echo "🤖 Downloading AI models..."
python -c "
from ultralytics import YOLO
import os

# Download YOLOv8 models
models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
for model in models:
    if not os.path.exists(f'data/models/{model}'):
        print(f'Downloading {model}...')
        YOLO(model)
        os.rename(model, f'data/models/{model}')
        print(f'✅ {model} downloaded')
"

# Setup database
echo "🗄️ Setting up database..."
python -c "
from app.database.connection import create_tables
create_tables()
print('✅ Database tables created')
"

# Create environment file
echo "⚙️ Creating environment configuration..."
cat > .env << EOL
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/traffic_db
REDIS_URL=redis://localhost:6379

# AI Models
YOLO_MODEL_PATH=data/models/yolov8n.pt

# Video Processing
CONFIDENCE_THRESHOLD=0.5
FRAME_SKIP=2

# Traffic Control
MIN_GREEN_TIME=15
MAX_GREEN_TIME=120
YELLOW_TIME=3
EOL

echo "✅ Setup completed successfully!"
echo ""
echo "🚀 To start the system:"
echo "1. Start database: docker-compose up -d postgres redis"
echo "2. Run application: python -m app.main"
echo "3. Open dashboard: http://localhost:8000"
echo ""
echo "📊 For Streamlit dashboard:"
echo "streamlit run frontend/streamlit_dashboard.py"
