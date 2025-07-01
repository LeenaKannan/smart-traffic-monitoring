# app/api/routes.py
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
import json
import asyncio
import logging

from app.services.stream_processor import StreamProcessor
from app.services.traffic_controller import TrafficController
from app.services.analytics_service import AnalyticsService
from app.config.settings import settings

logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Traffic Monitoring System", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
stream_processor = StreamProcessor()
traffic_controller = TrafficController()
analytics_service = AnalyticsService()

# WebSocket connections
active_connections: List[WebSocket] = []

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logger.info("Starting Smart Traffic Monitoring System")
    
    # Initialize default intersections
    default_intersections = [
        {
            'id': 'intersection_1',
            'config': {
                'lanes': ['north', 'south', 'east', 'west'],
                'signal_groups': {
                    'ns': ['north', 'south'],
                    'ew': ['east', 'west']
                }
            }
        }
    ]
    
    for intersection in default_intersections:
        traffic_controller.initialize_intersection(
            intersection['id'], 
            intersection['config']
        )

# Stream Management Endpoints
@app.post("/api/streams/start")
async def start_stream(camera_id: str, stream_url: str):
    """Start processing video stream"""
    result = await stream_processor.start_stream(camera_id, stream_url)
    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])
    return result

@app.post("/api/streams/stop")
async def stop_stream(camera_id: str):
    """Stop processing video stream"""
    result = await stream_processor.stop_stream(camera_id)
    if 'error' in result:
        raise HTTPException(status_code=404, detail=result['error'])
    return result

@app.get("/api/streams/{camera_id}/status")
async def get_stream_status(camera_id: str):
    """Get stream status"""
    result = stream_processor.get_stream_status(camera_id)
    if 'error' in result:
        raise HTTPException(status_code=404, detail=result['error'])
    return result

@app.get("/api/streams/{camera_id}/results")
async def get_stream_results(camera_id: str, count: int = 10):
    """Get latest processing results"""
    results = stream_processor.get_latest_results(camera_id, count)
    return {"camera_id": camera_id, "results": results}

# Traffic Control Endpoints
@app.post("/api/traffic/intersections")
async def create_intersection(intersection_id: str, config: Dict):
    """Create new intersection"""
    traffic_controller.initialize_intersection(intersection_id, config)
    return {"intersection_id": intersection_id, "status": "created"}

@app.get("/api/traffic/intersections/{intersection_id}/status")
async def get_intersection_status(intersection_id: str):
    """Get intersection signal status"""
    result = traffic_controller.get_signal_status(intersection_id)
    if 'error' in result:
        raise HTTPException(status_code=404, detail=result['error'])
    return result

@app.post("/api/traffic/intersections/{intersection_id}/optimize")
async def optimize_intersection(intersection_id: str, density_data: Dict):
    """Optimize signal timing based on density data"""
    result = await traffic_controller.optimize_signal_timing(intersection_id, density_data)
    if 'error' in result:
        raise HTTPException(status_code=404, detail=result['error'])
    return result

@app.post("/api/traffic/emergency")
async def handle_emergency(intersection_id: str, direction: str):
    """Handle emergency vehicle priority"""
    result = await traffic_controller.handle_emergency_vehicle(intersection_id, direction)
    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])
    return result

# Analytics Endpoints
@app.get("/api/analytics/violations")
async def get_violations(camera_id: Optional[str] = None, 
                        violation_type: Optional[str] = None,
                        limit: int = 100):
    """Get traffic violations"""
    violations = analytics_service.get_violations(camera_id, violation_type, limit)
    return {"violations": violations}

@app.get("/api/analytics/density/{camera_id}")
async def get_density_analytics(camera_id: str, hours: int = 24):
    """Get traffic density analytics"""
    analytics = analytics_service.get_density_analytics(camera_id, hours)
    return analytics

@app.get("/api/analytics/congestion/predict/{camera_id}")
async def predict_congestion(camera_id: str, minutes: int = 15):
    """Predict traffic congestion"""
    prediction = stream_processor.density_estimator.predict_congestion(camera_id, minutes)
    return prediction

# WebSocket for real-time updates
@app.websocket("/ws/{camera_id}")
async def websocket_endpoint(websocket: WebSocket, camera_id: str):
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Get latest results
            results = stream_processor.get_latest_results(camera_id, 1)
            if results:
                await websocket.send_text(json.dumps(results[0]))
            
            await asyncio.sleep(1)  # Send updates every second
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
