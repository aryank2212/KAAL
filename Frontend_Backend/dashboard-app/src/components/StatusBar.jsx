import { motion } from "motion/react";

export default function StatusBar({ systemMetrics, isRunning, swarmData = {} }) {
  const metrics = [
    { label: "DRONES", value: swarmData.alive || 0, color: "#34d399", icon: "⬡" },
    { label: "KIA", value: swarmData.kia || 0, color: "#f43f5e", icon: "◈" },
    { label: "NOISE", value: swarmData.noise_score || 0, color: (swarmData.noise_score > 110) ? "#f43f5e" : "#fbbf24", icon: "⚡" },
    { label: "MODE", value: swarmData.mode || "NORMAL", color: swarmData.mode === 'KAAL' ? '#22d3ee' : '#cbd5e1', icon: "⏣" },
    { label: "CPU", value: `${Math.round(systemMetrics.cpu)}%`, color: systemMetrics.cpu > 80 ? "#f43f5e" : "#22d3ee", icon: "⬡" },
    { label: "MEM", value: `${Math.round(systemMetrics.memory)}%`, color: systemMetrics.memory > 85 ? "#f43f5e" : "#3b82f6", icon: "◈" },
    { label: "GPU", value: `${Math.round(systemMetrics.gpu)}%`, color: systemMetrics.gpu > 80 ? "#fbbf24" : "#34d399", icon: "◇" },
    { label: "LATENCY", value: `${Math.round(systemMetrics.latency)}ms`, color: systemMetrics.latency > 100 ? "#fbbf24" : "#22d3ee", icon: "◎" },
    { label: "UPTIME", value: systemMetrics.uptime, color: "#34d399", icon: "⏣" },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.2 }}
      className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-red-500/8 bg-[#0a0406]/60"
    >
      {/* Sim running indicator */}
      <div className="flex items-center gap-2 pr-3 border-r border-red-500/10">
        <div
          className="w-1.5 h-4 rounded-full"
          style={{
            background: isRunning ? "#34d399" : "#fbbf24",
            boxShadow: `0 0 8px ${isRunning ? "rgba(52,211,153,0.5)" : "rgba(251,191,36,0.5)"}`,
          }}
        />
        <span className="text-[9px] tracking-[3px] text-slate-500" style={{ fontFamily: "var(--font-display)" }}>
          {isRunning ? "LIVE" : "IDLE"}
        </span>
      </div>

      {/* Metrics row */}
      {metrics.map((m, i) => (
        <div key={m.label} className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] opacity-40">{m.icon}</span>
            <span className="text-[8px] tracking-[2px] text-slate-600" style={{ fontFamily: "var(--font-display)" }}>
              {m.label}
            </span>
            <span
              className="text-[11px] font-medium"
              style={{ fontFamily: "var(--font-mono)", color: m.color, textShadow: `0 0 6px ${m.color}40` }}
            >
              {m.value}
            </span>
          </div>
          {i < metrics.length - 1 && <div className="w-px h-3 bg-red-500/8" />}
        </div>
      ))}

      {/* Spacer */}
      <div className="flex-1" />

      {/* Right info */}
      <div className="flex items-center gap-1.5">
        <div className="w-1 h-1 rounded-full bg-red-400/50" />
        <span className="text-[8px] tracking-[2px] text-slate-600" style={{ fontFamily: "var(--font-display)" }}>
          KAAL v2.4.1
        </span>
      </div>
    </motion.div>
  );
}
