# frontend/streamlit_dashboard.py
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import cv2
import numpy as np

st.set_page_config(
    page_title="Smart Traffic Monitoring",
    page_icon="🚦",
    layout="wide"
)

class TrafficDashboard:
    def __init__(self):
        self.api_base = "http://localhost:8000/api"
    
    def get_camera_status(self, camera_id):
        """Get camera stream status"""
        try:
            response = requests.get(f"{self.api_base}/streams/{camera_id}/status")
            return response.json()
        except:
            return {"error": "Connection failed"}
    
    def get_violations(self, camera_id=None):
        """Get recent violations"""
        try:
            params = {"limit": 50}
            if camera_id:
                params["camera_id"] = camera_id
            response = requests.get(f"{self.api_base}/analytics/violations", params=params)
            return response.json().get("violations", [])
        except:
            return []
    
    def get_density_data(self, camera_id, hours=24):
        """Get density analytics"""
        try:
            response = requests.get(f"{self.api_base}/analytics/density/{camera_id}?hours={hours}")
            return response.json()
        except:
            return {}

def main():
    st.title("🚦 Smart Traffic Monitoring System")
    st.markdown("Real-time AI-powered traffic monitoring for Indian roads")
    
    dashboard = TrafficDashboard()
    
    # Sidebar
    st.sidebar.header("Camera Selection")
    camera_options = ["camera_1", "camera_2", "camera_3"]
    selected_camera = st.sidebar.selectbox("Select Camera", camera_options)
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=True)
    if auto_refresh:
        st.rerun()
    
    # Main dashboard
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader(f"📹 Live Feed - {selected_camera}")
        
        # Camera status
        status = dashboard.get_camera_status(selected_camera)
        if "error" not in status:
            st.success(f"Status: {status.get('status', 'Unknown')}")
            st.info(f"Uptime: {status.get('uptime', 0):.1f} seconds")
        else:
            st.error("Camera offline")
        
        # Placeholder for video feed
        st.image("https://via.placeholder.com/640x480?text=Live+Traffic+Feed", 
                caption="Live traffic monitoring")
    
    with col2:
        st.subheader("🚨 Recent Violations")
        violations = dashboard.get_violations(selected_camera)
        
        if violations:
            for violation in violations[:5]:
                with st.expander(f"{violation['violation_type']} - {violation['vehicle_class']}"):
                    st.write(f"**Time:** {violation['timestamp']}")
                    st.write(f"**Camera:** {violation['camera_id']}")
                    st.write(f"**Status:** {violation['status']}")
        else:
            st.info("No recent violations")
    
    with col3:
        st.subheader("📊 Quick Stats")
        density_data = dashboard.get_density_data(selected_camera, 1)
        
        if density_data and "error" not in density_data:
            st.metric("Average Vehicles", f"{density_data.get('average_vehicles', 0):.1f}")
            st.metric("Peak Vehicles", density_data.get('peak_vehicles', 0))
            st.metric("Congestion Score", f"{density_data.get('average_congestion_score', 0):.2f}")
        else:
            st.info("No data available")
    
    # Traffic density chart
    st.subheader("📈 Traffic Density Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hourly distribution
        density_data = dashboard.get_density_data(selected_camera, 24)
        if density_data and "hourly_distribution" in density_data:
            hourly_df = pd.DataFrame(
                list(density_data["hourly_distribution"].items()),
                columns=["Hour", "Average Vehicles"]
            )
            fig = px.bar(hourly_df, x="Hour", y="Average Vehicles", 
                        title="Hourly Traffic Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Density level distribution
        if density_data and "density_level_distribution" in density_data:
            density_df = pd.DataFrame(
                list(density_data["density_level_distribution"].items()),
                columns=["Density Level", "Count"]
            )
            fig = px.pie(density_df, values="Count", names="Density Level",
                        title="Traffic Density Levels")
            st.plotly_chart(fig, use_container_width=True)
    
    # Traffic control section
    st.subheader("🚦 Traffic Signal Control")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚨 Emergency Override", type="primary"):
            st.success("Emergency override activated!")
    
    with col2:
        if st.button("⚙️ Optimize Signals"):
            st.info("Signal optimization in progress...")
    
    with col3:
        if st.button("📊 Generate Report"):
            st.info("Generating traffic report...")

if __name__ == "__main__":
    main()
