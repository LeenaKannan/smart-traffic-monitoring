// frontend/src/components/TrafficDashboard.jsx
import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const TrafficDashboard = () => {
  const [cameraFeeds, setCameraFeeds] = useState([]);
  const [violations, setViolations] = useState([]);
  const [densityData, setDensityData] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState('camera_1');

  useEffect(() => {
    // WebSocket connection for real-time updates
    const ws = new WebSocket(`ws://localhost:8000/ws/${selectedCamera}`);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      updateDashboard(data);
    };

    return () => ws.close();
  }, [selectedCamera]);

  const updateDashboard = (data) => {
    // Update density chart
    setDensityData(prev => [...prev.slice(-20), {
      time: new Date(data.timestamp * 1000).toLocaleTimeString(),
      vehicles: data.density_data.total_vehicles,
      congestion: data.density_data.congestion_level
    }]);

    // Update violations
    if (data.violations.length > 0) {
      setViolations(prev => [...data.violations, ...prev].slice(0, 10));
    }
  };

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>Smart Traffic Monitoring System</h1>
        <div className="camera-selector">
          <select
            value={selectedCamera}
            onChange={(e) => setSelectedCamera(e.target.value)}
          >
            <option value="camera_1">Camera 1</option>
            <option value="camera_2">Camera 2</option>
            <option value="camera_3">Camera 3</option>
          </select>
        </div>
      </header>

      <div className="dashboard-grid">
        {/* Live Camera Feed */}
        <div className="camera-feed">
          <h3>Live Feed - {selectedCamera}</h3>
          <div className="video-container">
            <img
              src={`/api/streams/${selectedCamera}/live`}
              alt="Live traffic feed"
              className="live-video"
            />
          </div>
        </div>

        {/* Traffic Density Chart */}
        <div className="density-chart">
          <h3>Traffic Density</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={densityData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="vehicles"
                stroke="#8884d8"
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Violations Panel */}
        <div className="violations-panel">
          <h3>Recent Violations</h3>
          <div className="violations-list">
            {violations.map((violation, index) => (
              <div key={index} className="violation-item">
                <span className="violation-type">{violation.type}</span>
                <span className="violation-vehicle">{violation.vehicle_class}</span>
                <span className="violation-time">
                  {new Date(violation.timestamp * 1000).toLocaleTimeString()}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Signal Control */}
        <div className="signal-control">
          <h3>Traffic Signal Control</h3>
          <div className="intersection-status">
            <div className="signal-light red"></div>
            <div className="signal-light yellow"></div>
            <div className="signal-light green active"></div>
          </div>
          <button className="emergency-btn">Emergency Override</button>
        </div>
      </div>
    </div>
  );
};

export default TrafficDashboard;
