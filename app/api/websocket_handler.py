# app/api/websocket_handler.py
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List
import json
import asyncio
import logging

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.camera_subscribers: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, camera_id: str):
        """Connect client to specific camera feed"""
        await websocket.accept()
        
        if camera_id not in self.camera_subscribers:
            self.camera_subscribers[camera_id] = []
        
        self.camera_subscribers[camera_id].append(websocket)
        logger.info(f"Client connected to camera {camera_id}")

    def disconnect(self, websocket: WebSocket, camera_id: str):
        """Disconnect client from camera feed"""
        if camera_id in self.camera_subscribers:
            if websocket in self.camera_subscribers[camera_id]:
                self.camera_subscribers[camera_id].remove(websocket)
        logger.info(f"Client disconnected from camera {camera_id}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific client"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    async def broadcast_to_camera(self, message: Dict, camera_id: str):
        """Broadcast message to all clients subscribed to camera"""
        if camera_id not in self.camera_subscribers:
            return
        
        message_str = json.dumps(message)
        disconnected_clients = []
        
        for connection in self.camera_subscribers[camera_id]:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected_clients.append(connection)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.camera_subscribers[camera_id].remove(client)

    async def broadcast_alert(self, alert: Dict):
        """Broadcast alert to all connected clients"""
        alert_str = json.dumps(alert)
        
        for camera_id, connections in self.camera_subscribers.items():
            for connection in connections:
                try:
                    await connection.send_text(alert_str)
                except Exception:
                    pass

manager = ConnectionManager()
