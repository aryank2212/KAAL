import { useState, useEffect } from "react";
import Sidebar from "./components/Sidebar";
import Navbar from "./components/Navbar";
import "./App.css";
import MainPanel from "./components/MainPanel";
import GraphPanel from "./components/GraphPanel";
import MultiGraph from "./components/MultiGraph";
import Controls from "./components/Controls";
import HealthMonitor from "./components/SystemStatus";
import StatusBar from "./components/StatusBar";

function App() {
  const [selectedModel, setSelectedModel] = useState("Model 1");
  const [data, setData] = useState([]);
  const [image, setImage] = useState(null);
  const [isRunning, setIsRunning] = useState(true);
  const [systemMetrics, setSystemMetrics] = useState({
    cpu: 42,
    memory: 67,
    gpu: 31,
    bandwidth: 88,
    latency: 23,
    uptime: "47:12:03",
  });
  const [swarmData, setSwarmData] = useState({
    status: 'NOMINAL',
    alive: 0,
    kia: 0,
    mode: 'NORMAL'
  });

  // Seed data
  useEffect(() => {
    const initial = [];
    for (let i = 1; i <= 12; i++) {
      initial.push({
        time: i,
        value: 10 + Math.sin(i * 0.5) * 15 + Math.random() * 5,
        value2: 5 + Math.cos(i * 0.3) * 10 + Math.random() * 5,
        value3: 20 + Math.sin(i * 0.8) * 8 + Math.random() * 3,
      });
    }
    setData(initial);
    setImage("/api/video_feed");
  }, [selectedModel]);

  // Live data stream
  useEffect(() => {
    if (!isRunning) return;

    const fetchBackendData = async () => {
      try {
        const response = await fetch('/api/python-data');
        if (!response.ok) throw new Error('Failed to fetch data');
        const result = await response.json();
        
        setData(result.data);
        setSystemMetrics(result.systemMetrics);
        setSwarmData(prev => ({
          ...prev,
          status: result.status,
          noise_score: result.noise_score,
          alive: result.alive !== undefined ? result.alive : prev.alive,
          kia: result.kia !== undefined ? result.kia : prev.kia,
        }));
      } catch (err) {
        console.error("Error fetching python data:", err);
      }
    };

    // Fetch immediately on mount or start
    fetchBackendData();

    const interval = setInterval(fetchBackendData, 1500);

    return () => clearInterval(interval);
  }, [isRunning]);

  return (
    <div className="app flex flex-col h-screen overflow-hidden text-white">
      {/* Top Navbar */}
      <Navbar systemMetrics={systemMetrics} isRunning={isRunning} />

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <Sidebar
          selectedModel={selectedModel}
          setSelectedModel={setSelectedModel}
          systemMetrics={systemMetrics}
        />

        {/* Main Content Grid — 8 columns for precise sizing */}
        <div className="flex-1 grid grid-cols-8 grid-rows-[auto_2fr_1fr_auto] gap-3 p-3 overflow-hidden">
          {/* Status Bar — full width */}
          <div className="col-span-8">
            <StatusBar systemMetrics={systemMetrics} isRunning={isRunning} swarmData={swarmData} />
          </div>

          {/* Main Mission Feed — dominant, spans 6 cols + 2 rows */}
          <div className="col-span-6 row-span-2 glass active">
            <MainPanel
              image={image}
              selectedModel={selectedModel}
              isRunning={isRunning}
              swarmData={swarmData}
              onSwarmData={setSwarmData}
            />
          </div>

          {/* Analytics Graph — narrow right panel */}
          <div className="col-span-2 glass">
            <GraphPanel data={data} />
          </div>

          {/* Performance Multi-Graph — narrow right panel */}
          <div className="col-span-2 glass">
            <MultiGraph data={data} />
          </div>

          {/* Controls — bottom left */}
          <div className="col-span-4 glass">
            <Controls
              setSelectedModel={setSelectedModel}
              isRunning={isRunning}
              setIsRunning={setIsRunning}
            />
          </div>

          {/* Health Monitoring System — bottom right, compact with stars */}
          <div className="col-span-4">
            <HealthMonitor systemMetrics={systemMetrics} isRunning={isRunning} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;