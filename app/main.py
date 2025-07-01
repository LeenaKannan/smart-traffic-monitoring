# app/main.py
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
from contextlib import asynccontextmanager

from app.config.settings import settings
from app.database.connection import create_tables
from app.api.routes import app as routes_app
from app.services.stream_processor import StreamProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting Smart Traffic Monitoring System")
    create_tables()
    
    # Initialize default camera streams if configured
    stream_processor = StreamProcessor()
    for i, source in enumerate(settings.VIDEO_SOURCES):
        camera_id = f"camera_{i+1}"
        try:
            await stream_processor.start_stream(camera_id, source)
            logger.info(f"Started default stream: {camera_id}")
        except Exception as e:
            logger.error(f"Failed to start default stream {camera_id}: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Smart Traffic Monitoring System")

# Create FastAPI app
app = FastAPI(
    title="Smart Traffic Monitoring System",
    description="AI-powered traffic monitoring and control system for Indian traffic conditions",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="frontend/dist"), name="static")

# Include API routes
app.mount("/api", routes_app)

@app.get("/")
async def root():
    return {"message": "Smart Traffic Monitoring System API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
