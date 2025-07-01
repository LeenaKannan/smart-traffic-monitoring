# app/utils/factory.py
from app.config.settings import settings

def get_stream_processor():
    """Factory to get appropriate stream processor"""
    if settings.IS_RASPBERRY_PI:
        from app.services.rpi_stream_processor import RPiStreamProcessor
        return RPiStreamProcessor()
    else:
        from app.services.stream_processor import StreamProcessor
        return StreamProcessor()

def get_vehicle_detector():
    """Factory to get appropriate vehicle detector"""
    if settings.IS_RASPBERRY_PI:
        from app.models.rpi_vehicle_detector import RPiVehicleDetector
        return RPiVehicleDetector()
    else:
        from app.models.vehicle_detector import VehicleDetector
        return VehicleDetector()
