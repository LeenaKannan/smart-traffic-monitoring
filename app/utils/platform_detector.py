# app/utils/platform_detector.py
import platform
import psutil

def get_deployment_mode():
    """Automatically detect deployment environment"""
    machine = platform.machine()
    total_ram = psutil.virtual_memory().total / (1024**3)  # GB
    
    if machine.startswith('arm') and total_ram < 8:
        return "raspberry_pi"
    elif total_ram < 16:
        return "edge_device" 
    else:
        return "server"

# Use in main.py
deployment_mode = get_deployment_mode()
if deployment_mode == "raspberry_pi":
    from app.services.rpi_stream_processor import RPiStreamProcessor as StreamProcessor
else:
    from app.services.stream_processor import StreamProcessor
