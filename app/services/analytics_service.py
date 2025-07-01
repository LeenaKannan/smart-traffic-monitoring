# app/services/analytics_service.py
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from app.database.models import TrafficViolation, TrafficDensity, VehicleDetection
from app.database.connection import AsyncSessionLocal
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class AnalyticsService:
    def __init__(self):
        self.session_factory = AsyncSessionLocal

    async def get_violations(self, camera_id: Optional[str] = None,
                           violation_type: Optional[str] = None,
                           limit: int = 100) -> List[Dict]:
        """Get traffic violations with filtering"""
        async with self.session_factory() as session:
            query = session.query(TrafficViolation)
            
            if camera_id:
                query = query.filter(TrafficViolation.camera_id == camera_id)
            if violation_type:
                query = query.filter(TrafficViolation.violation_type == violation_type)
            
            violations = await query.order_by(desc(TrafficViolation.timestamp)).limit(limit).all()
            
            return [
                {
                    "id": v.id,
                    "camera_id": v.camera_id,
                    "violation_type": v.violation_type,
                    "vehicle_class": v.vehicle_class,
                    "license_plate": v.license_plate,
                    "timestamp": v.timestamp.isoformat(),
                    "status": v.status
                }
                for v in violations
            ]

    async def get_density_analytics(self, camera_id: str, hours: int = 24) -> Dict:
        """Get traffic density analytics for specified time period"""
        async with self.session_factory() as session:
            start_time = datetime.utcnow() - timedelta(hours=hours)
            
            density_records = await session.query(TrafficDensity).filter(
                TrafficDensity.camera_id == camera_id,
                TrafficDensity.timestamp >= start_time
            ).order_by(TrafficDensity.timestamp).all()
            
            if not density_records:
                return {"error": "No data available"}
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame([
                {
                    "timestamp": r.timestamp,
                    "total_vehicles": r.total_vehicles,
                    "congestion_score": r.congestion_score,
                    "density_level": r.density_level
                }
                for r in density_records
            ])
            
            # Calculate statistics
            avg_vehicles = df['total_vehicles'].mean()
            peak_vehicles = df['total_vehicles'].max()
            avg_congestion = df['congestion_score'].mean()
            
            # Find peak hours
            df['hour'] = df['timestamp'].dt.hour
            hourly_avg = df.groupby('hour')['total_vehicles'].mean()
            peak_hour = hourly_avg.idxmax()
            
            return {
                "camera_id": camera_id,
                "period_hours": hours,
                "total_records": len(density_records),
                "average_vehicles": round(avg_vehicles, 2),
                "peak_vehicles": int(peak_vehicles),
                "average_congestion_score": round(avg_congestion, 2),
                "peak_hour": int(peak_hour),
                "hourly_distribution": hourly_avg.to_dict(),
                "density_level_distribution": df['density_level'].value_counts().to_dict()
            }

    async def generate_daily_report(self, date: datetime) -> Dict:
        """Generate comprehensive daily traffic report"""
        async with self.session_factory() as session:
            start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = start_date + timedelta(days=1)
            
            # Vehicle detection stats
            vehicle_stats = await session.query(
                VehicleDetection.vehicle_class,
                func.count(VehicleDetection.id).label('count')
            ).filter(
                VehicleDetection.timestamp >= start_date,
                VehicleDetection.timestamp < end_date
            ).group_by(VehicleDetection.vehicle_class).all()
            
            # Violation stats
            violation_stats = await session.query(
                TrafficViolation.violation_type,
                func.count(TrafficViolation.id).label('count')
            ).filter(
                TrafficViolation.timestamp >= start_date,
                TrafficViolation.timestamp < end_date
            ).group_by(TrafficViolation.violation_type).all()
            
            return {
                "date": date.strftime("%Y-%m-%d"),
                "vehicle_detection_stats": {stat.vehicle_class: stat.count for stat in vehicle_stats},
                "violation_stats": {stat.violation_type: stat.count for stat in violation_stats},
                "total_vehicles_detected": sum(stat.count for stat in vehicle_stats),
                "total_violations": sum(stat.count for stat in violation_stats)
            }
