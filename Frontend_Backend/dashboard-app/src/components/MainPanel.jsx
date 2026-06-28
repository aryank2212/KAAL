import { useState, useEffect } from "react";
import { motion } from "motion/react";
import KaalSwarmCanvas from "./KaalSwarmCanvas";

function MainPanel({ image, selectedModel, isRunning, swarmData = {}, onSwarmData }) {
  const [isVideo, setIsVideo] = useState(false);
  const [coords, setCoords] = useState({ lat: 28.6139, lon: 77.209 });

  useEffect(() => {
    if (!isRunning) return;
    const interval = setInterval(() => {
      setCoords((prev) => ({
        lat: prev.lat + (Math.random() - 0.5) * 0.01,
        lon: prev.lon + (Math.random() - 0.5) * 0.01,
      }));
    }, 2000);
    return () => clearInterval(interval);
  }, [isRunning]);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, delay: 0.1 }}
      className="h-full flex flex-col"
    >
      {/* Header row */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className="panel-header mb-0">
            <h3 className="panel-title">MISSION FEED</h3>
          </div>
          <div className="h-3 w-px bg-blue-500/20" />
          <span className="text-[9px] tracking-wider text-slate-600" style={{ fontFamily: "var(--font-mono)" }}>
            {selectedModel.toUpperCase()}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsVideo(!isVideo)}
            className="btn text-[10px] py-1.5 px-3"
          >
            {isVideo ? "◉ IMAGE" : "▶ SIMULATION"}
          </button>
          <div className="panel-dot" />
        </div>
      </div>

      {/* Feed viewport */}
      <div className="relative flex-1 scanlines rounded-xl overflow-hidden border border-blue-500/10 bg-black">
        {isVideo ? (
          <div className="w-full h-full relative z-0">
            <KaalSwarmCanvas onTelemetry={onSwarmData} />
          </div>
        ) : (
          <img
            src={image}
            className="w-full h-full object-cover"
            alt="Mission feed"
            style={{ filter: "brightness(0.7) contrast(1.1)" }}
          />
        )}

        {/* HUD Overlay — Corner brackets */}
        <div className="absolute inset-0 pointer-events-none">
          {/* Top-left bracket */}
          <div className="absolute top-3 left-3">
            <div className="w-6 h-6 border-t-2 border-l-2 border-cyan-400/40 rounded-tl-sm" />
          </div>
          {/* Top-right bracket */}
          <div className="absolute top-3 right-3">
            <div className="w-6 h-6 border-t-2 border-r-2 border-cyan-400/40 rounded-tr-sm" />
          </div>
          {/* Bottom-left bracket */}
          <div className="absolute bottom-3 left-3">
            <div className="w-6 h-6 border-b-2 border-l-2 border-cyan-400/40 rounded-bl-sm" />
          </div>
          {/* Bottom-right bracket */}
          <div className="absolute bottom-3 right-3">
            <div className="w-6 h-6 border-b-2 border-r-2 border-cyan-400/40 rounded-br-sm" />
          </div>

          {/* Center crosshair */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
            <div className="w-8 h-8 relative">
              <div className="absolute top-0 left-1/2 -translate-x-1/2 w-px h-2 bg-cyan-400/30" />
              <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-px h-2 bg-cyan-400/30" />
              <div className="absolute left-0 top-1/2 -translate-y-1/2 w-2 h-px bg-cyan-400/30" />
              <div className="absolute right-0 top-1/2 -translate-y-1/2 w-2 h-px bg-cyan-400/30" />
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-1.5 h-1.5 rounded-full border border-cyan-400/40" />
            </div>
          </div>
        </div>

        {/* HUD Data overlays */}
        {/* LIVE badge */}
        <div className="absolute top-3 left-10 flex items-center gap-2">
          <div
            className="live-dot"
            style={{
              width: 6, height: 6,
              background: swarmData.status === 'ENGAGING' ? '#f43f5e' : (swarmData.status === 'JAMMED' ? '#fbbf24' : '')
            }}
          />
          <span
            className="text-[10px] tracking-wider font-medium"
            style={{
              fontFamily: "var(--font-display)",
              color: swarmData.status === 'ENGAGING' ? '#f43f5e' : (swarmData.status === 'JAMMED' ? '#fbbf24' : '#34d399')
            }}
          >
            {swarmData.status || "LIVE"}
          </span>
        </div>

        {/* SAT-LINK badge */}
        <div className="absolute top-3 right-10 flex items-center gap-2 px-2 py-0.5 rounded bg-blue-500/10 border border-blue-500/20">
          <div className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
          <span className="text-[9px] tracking-wider text-blue-300" style={{ fontFamily: "var(--font-display)" }}>
            {swarmData.mode === 'KAAL' ? 'KAAL MESH ACTIVE' : 'STANDARD OPS'}
          </span>
        </div>

        {/* Bottom data strip */}
        <div className="absolute bottom-0 left-0 right-0 flex items-center justify-between px-4 py-2 bg-gradient-to-t from-black/80 to-transparent">
          <div className="flex items-center gap-4">
            <DataTag label="ALT" value="1,200 km" />
            <DataTag label="LAT" value={coords.lat.toFixed(4)} />
            <DataTag label="LON" value={coords.lon.toFixed(4)} />
          </div>

          <div className="flex items-center gap-4">
            <DataTag label="SPD" value="27,400 km/h" />
            <DataTag label="ORBIT" value="LEO-342" />
          </div>
        </div>

        {/* Top vignette */}
        <div className="absolute inset-0 bg-gradient-to-b from-black/30 via-transparent to-transparent pointer-events-none" />
      </div>
    </motion.div>
  );
}

function DataTag({ label, value }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[8px] tracking-wider text-slate-500" style={{ fontFamily: "var(--font-display)" }}>
        {label}
      </span>
      <span className="text-[10px] font-medium text-cyan-300" style={{ fontFamily: "var(--font-mono)" }}>
        {value}
      </span>
    </div>
  );
}

export default MainPanel;