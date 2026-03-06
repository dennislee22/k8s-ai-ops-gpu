import { useState, useRef, useEffect } from "react";

const API_URL = "";
const FONT    = "'IBM Plex Mono', monospace";
const FONT2   = "'IBM Plex Sans', sans-serif";

// ── Themes ────────────────────────────────────────────────────────────────
const THEMES = {
  midnight: {
    name: "Midnight",
    dark: true,
    bg:        "#030712", headerBg:  "#070d1a", border:    "#0f1f35",
    accent:    "#38bdf8", accent2:   "#7dd3fc",
    msgBg:     "#070d1a", inputBg:   "#0a0f1e",
    userBg:    "#0f172a", userBorder:"#1e293b",
    statusBg:  "#040b14", dimText:   "#475569", bodyText:  "#cbd5e1",
    tooltipBg: "#070d1a", tooltipText:"#94a3b8",
    cpuBg:     "#020a14", cpuBorder: "#0f1f35", cpuLabel:  "#7dd3fc",
    textMain:  "#e2e8f0", inputText: "#e2e8f0",
    shimmer:   "linear-gradient(90deg,#38bdf8 0%,#818cf8 25%,#c084fc 50%,#38bdf8 75%,#00C1DE 100%)",
  },
  singapore: {
    // Dark red — deep crimson on near-black, white font
    name: "Singapore",
    dark: true,
    bg:        "#0d0000", headerBg:  "#1a0000", border:    "#4a0000",
    accent:    "#c41e1e", accent2:   "#ffffff",
    msgBg:     "#1a0000", inputBg:   "#110000",
    userBg:    "#220000", userBorder:"#4a0000",
    statusBg:  "#0d0000", dimText:   "#e5e5e5", bodyText:  "#ffffff",
    tooltipBg: "#1a0000", tooltipText:"#ffffff",
    cpuBg:     "#0d0000", cpuBorder: "#4a0000", cpuLabel:  "#ffffff",
    textMain:  "#ffffff", inputText: "#ffffff",
    shimmer:   "linear-gradient(90deg,#c41e1e 0%,#ef4444 30%,#991b1b 60%,#c41e1e 100%)",
  },
  cloudera: {
    // Orange and White — Cloudera brand colours
    name: "Cloudera",
    dark: false,
    bg:        "#ffffff", headerBg:  "#ffffff", border:    "#fed7aa",
    accent:    "#f96702", accent2:   "#ea580c",
    msgBg:     "#fff7ed", inputBg:   "#ffffff",
    userBg:    "#ffedd5", userBorder:"#fed7aa",
    statusBg:  "#fff7ed", dimText:   "#9a3412", bodyText:  "#431407",
    tooltipBg: "#ffffff", tooltipText:"#431407",
    cpuBg:     "#ffffff", cpuBorder: "#fed7aa", cpuLabel:  "#f96702",
    textMain:  "#1c0a00", inputText: "#1c0a00",
    shimmer:   "linear-gradient(90deg,#f96702 0%,#ea580c 30%,#fb923c 60%,#f96702 100%)",
  },
};

const EXAMPLES = [
  "Check overall cluster health",
  "Are there any warning events in the cluster?",
  "Why is my payment-service pod crashing?",
  "Show me node status and resource pressure",
  "Check all deployments for degraded replicas",
  "Any OOMKilled pods in the last hour?",
  "Check Longhorn storage health",
  "Are there any Longhorn volume issues?",
];

// ── Gradient colour for a CPU % value (green→yellow→orange→red) ──────────
function cpuColour(pct) {
  if (pct < 40)  return `hsl(${120 - pct * 1.5},100%,45%)`;   // green→yellow-green
  if (pct < 70)  return `hsl(${60  - (pct-40)*1.5},100%,50%)`; // yellow→orange
  return           `hsl(${25  - (pct-70)*0.8},100%,50%)`;       // orange→red
}

// ── Dot-matrix bar (mimics bpytop raster dots) ────────────────────────────
function DotBar({ pct, width = 120, colour }) {
  const COLS = Math.floor(width / 6);
  const filled = Math.round((pct / 100) * COLS);
  return (
    <div style={{ display: "flex", gap: 1, alignItems: "center" }}>
      {Array.from({ length: COLS }).map((_, i) => {
        const active = i < filled;
        const dotCol = active ? cpuColour((i / COLS) * 100) : "transparent";
        return (
          <div key={i} style={{
            width: 4, height: 8, borderRadius: 1,
            background: active ? dotCol : "transparent",
            border: `1px solid ${active ? dotCol : colour + "22"}`,
          }} />
        );
      })}
    </div>
  );
}

// ── bpytop-style CPU panel ────────────────────────────────────────────────
// metrics and history are passed from App (lifted state — shared with GpuPanel)
function CpuPanel({ t, metrics, history }) {
  if (!metrics) return (
    <div style={{
      width: 280, background: t.cpuBg, borderLeft: `1px solid ${t.cpuBorder}`,
      display: "flex", alignItems: "center", justifyContent: "center",
      fontFamily: FONT, fontSize: 11, color: t.dimText,
    }}>
      connecting...
    </div>
  );

  const cores = metrics.cpu_per_core || [];
  // layout: up to 3 columns
  const cols = cores.length > 12 ? 3 : cores.length > 6 ? 2 : 1;
  const perCol = Math.ceil(cores.length / cols);
  const colGroups = Array.from({ length: cols }, (_, c) =>
    cores.slice(c * perCol, (c + 1) * perCol)
  );
  const freqGhz = (metrics.freq_mhz / 1000).toFixed(1);
  const [la1, la5, la15] = metrics.load_avg || [0, 0, 0];

  // sparkline history graph
  const SPARK_W = 256, SPARK_H = 28;
  const pts = history.map((v, i) => {
    const x = (i / 59) * SPARK_W;
    const y = SPARK_H - (v / 100) * SPARK_H;
    return `${x},${y}`;
  }).join(" ");

  return (
    <div style={{
      width: 290, flexShrink: 0,
      background: t.cpuBg, borderLeft: `1px solid ${t.cpuBorder}`,
      display: "flex", flexDirection: "column",
      fontFamily: FONT, fontSize: 11, overflow: "hidden",
    }}>
      {/* ── Header row ── */}
      <div style={{
        padding: "8px 12px 6px", borderBottom: `1px solid ${t.cpuBorder}`,
        display: "flex", justifyContent: "space-between", alignItems: "center",
      }}>
        <span style={{ color: t.cpuLabel, fontWeight: 700, fontSize: 12 }}>
          CPU MONITOR
        </span>
        <span style={{ color: t.dimText, fontSize: 11 }}>
          {freqGhz} GHz · {metrics.cpu_count}C
        </span>
      </div>

      {/* ── Sparkline + total bar ── */}
      <div style={{ padding: "8px 12px 4px" }}>
        {/* Sparkline */}
        <div style={{
          background: t.headerBg, border: `1px solid ${t.cpuBorder}`,
          borderRadius: 4, marginBottom: 6, overflow: "hidden", position: "relative",
        }}>
          <svg width={SPARK_W} height={SPARK_H} style={{ display: "block" }}>
            {history.length > 1 && (
              <>
                <defs>
                  <linearGradient id="sparkfill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={cpuColour(metrics.cpu_total)} stopOpacity="0.4"/>
                    <stop offset="100%" stopColor={cpuColour(metrics.cpu_total)} stopOpacity="0.05"/>
                  </linearGradient>
                </defs>
                <polyline
                  points={pts}
                  fill="none"
                  stroke={cpuColour(metrics.cpu_total)}
                  strokeWidth="1.5"
                />
                <polygon
                  points={`0,${SPARK_H} ${pts} ${SPARK_W},${SPARK_H}`}
                  fill="url(#sparkfill)"
                />
              </>
            )}
          </svg>
          <div style={{
            position: "absolute", top: 3, right: 5,
            fontSize: 10, color: t.dimText,
          }}>60s</div>
        </div>

        {/* Overall CPU bar */}
        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8 }}>
          <span style={{ color: t.cpuLabel, minWidth: 28, fontSize: 11 }}>CPU</span>
          <div style={{ flex: 1 }}>
            <DotBar pct={metrics.cpu_total} width={200} colour={t.cpuBorder} />
          </div>
          <span style={{
            minWidth: 36, textAlign: "right",
            color: cpuColour(metrics.cpu_total), fontWeight: 700, fontSize: 12,
          }}>
            {metrics.cpu_total}%
          </span>
        </div>
      </div>

      {/* ── Per-core grid ── */}
      <div style={{
        flex: 1, overflowY: "auto", padding: "0 12px",
        display: "flex", gap: 10,
      }}>
        {colGroups.map((group, ci) => (
          <div key={ci} style={{ flex: 1, display: "flex", flexDirection: "column", gap: 3 }}>
            {group.map((pct, i) => {
              const coreIdx = ci * perCol + i;
              return (
                <div key={coreIdx} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <span style={{ color: t.dimText, minWidth: 22, fontSize: 10 }}>
                    C{coreIdx + 1}
                  </span>
                  <div style={{ flex: 1, position: "relative", height: 8 }}>
                    {/* Track */}
                    <div style={{
                      position: "absolute", inset: 0,
                      background: t.cpuBorder + "33", borderRadius: 2,
                    }} />
                    {/* Fill */}
                    <div style={{
                      position: "absolute", top: 0, left: 0, bottom: 0,
                      width: `${pct}%`, borderRadius: 2,
                      background: cpuColour(pct),
                      transition: "width 0.8s ease, background 0.8s ease",
                    }} />
                    {/* Dot pattern overlay */}
                    <div style={{
                      position: "absolute", inset: 0,
                      backgroundImage: `repeating-linear-gradient(90deg, transparent, transparent 4px, ${t.cpuBg}66 4px, ${t.cpuBg}66 5px)`,
                      borderRadius: 2,
                    }} />
                  </div>
                  <span style={{
                    minWidth: 30, textAlign: "right",
                    color: pct > 80 ? cpuColour(pct) : t.dimText,
                    fontWeight: pct > 80 ? 700 : 400, fontSize: 10,
                  }}>
                    {pct}%
                  </span>
                </div>
              );
            })}
          </div>
        ))}
      </div>

      {/* ── RAM Monitor ── */}
      <div style={{ padding: "8px 12px 6px", borderTop: `1px solid ${t.cpuBorder}` }}>
        <div style={{ color: t.cpuLabel, fontWeight: 700, fontSize: 13, marginBottom: 6 }}>RAM MONITOR</div>
        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
          <span style={{ color: t.cpuLabel, fontSize: 13 }}>RAM</span>
          <div style={{ flex: 1, height: 8, background: t.cpuBorder + "33", borderRadius: 2, position: "relative" }}>
            <div style={{
              position: "absolute", top: 0, left: 0, bottom: 0,
              width: `${metrics.mem_pct}%`, borderRadius: 2,
              background: cpuColour(metrics.mem_pct),
              transition: "width 1s ease",
            }} />
          </div>
          <span style={{ color: t.dimText, fontSize: 10, minWidth: 72, textAlign: "right" }}>
            {metrics.mem_used_gb}/{metrics.mem_total_gb} GB
          </span>
        </div>
      </div>

      {/* ── Load average ── */}
      <div style={{
        padding: "6px 12px 8px", borderTop: `1px solid ${t.cpuBorder}`,
        display: "flex", justifyContent: "space-between", alignItems: "center",
      }}>
        <span style={{ color: t.cpuLabel, fontWeight: 700, fontSize: 13 }}>LOAD AVG</span>
        <span style={{ color: t.dimText, fontSize: 13, letterSpacing: "0.05em" }}>
          {la1} &nbsp; {la5} &nbsp; {la15}
        </span>
      </div>
    </div>
  );
}


// ── GPU colour (cool blue→cyan→green for normal, yellow→red for hot) ─────
function gpuColour(pct) {
  if (pct < 50) return `hsl(${200 - pct * 0.4},100%,55%)`;  // blue→cyan
  if (pct < 80) return `hsl(${180 - (pct-50)*4},100%,50%)`; // cyan→green→yellow
  return          `hsl(${60  - (pct-80)*3},100%,50%)`;       // yellow→red
}

function tempColour(c) {
  if (c < 60) return "#22c55e";
  if (c < 75) return "#f59e0b";
  return "#ef4444";
}

// ── GPU Monitor Panel ─────────────────────────────────────────────────────
function GpuPanel({ gpus, t }) {
  if (!gpus || gpus.length === 0) return null;

  return (
    <div style={{
      width: 290, flexShrink: 0,
      background: t.cpuBg, borderLeft: `1px solid ${t.cpuBorder}`,
      display: "flex", flexDirection: "column",
      fontFamily: FONT, fontSize: 11, overflow: "hidden",
    }}>
      {/* Header */}
      <div style={{
        padding: "8px 12px 6px", borderBottom: `1px solid ${t.cpuBorder}`,
        display: "flex", justifyContent: "space-between", alignItems: "center",
      }}>
        <span style={{ color: t.cpuLabel, fontWeight: 700, fontSize: 12 }}>GPU MONITOR</span>
        <span style={{ color: t.dimText, fontSize: 10 }}>{gpus.length} device{gpus.length > 1 ? "s" : ""}</span>
      </div>

      <div style={{ flex: 1, overflowY: "auto", padding: "10px 12px", display: "flex", flexDirection: "column", gap: 14 }}>
        {gpus.map((gpu) => {
          const utilCol  = gpuColour(gpu.util_pct);
          const memCol   = gpuColour(gpu.mem_pct);
          const tCol     = tempColour(gpu.temp_c);
          const shortName = gpu.name.replace(/NVIDIA\s*/i, "").replace(/GeForce\s*/i, "").trim();

          return (
            <div key={gpu.index} style={{
              background: t.headerBg, border: `1px solid ${t.cpuBorder}`,
              borderRadius: 8, padding: "10px 12px", display: "flex", flexDirection: "column", gap: 8,
            }}>
              {/* GPU name + index */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ color: t.cpuLabel, fontWeight: 700, fontSize: 11 }}>
                  GPU {gpu.index}
                </span>
                <span style={{ color: t.dimText, fontSize: 9, maxWidth: 170, overflow: "hidden",
                               textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {shortName}
                </span>
              </div>

              {/* Utilisation */}
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                  <span style={{ color: t.dimText, fontSize: 10 }}>Utilisation</span>
                  <span style={{ color: utilCol, fontWeight: 700, fontSize: 11 }}>{gpu.util_pct}%</span>
                </div>
                <div style={{ height: 10, background: t.cpuBorder + "33", borderRadius: 3, position: "relative" }}>
                  <div style={{
                    position: "absolute", top: 0, left: 0, bottom: 0,
                    width: `${gpu.util_pct}%`, borderRadius: 3,
                    background: utilCol,
                    boxShadow: `0 0 6px ${utilCol}88`,
                    transition: "width 1s ease",
                  }} />
                  {/* dot-stripe overlay */}
                  <div style={{
                    position: "absolute", inset: 0, borderRadius: 3,
                    backgroundImage: `repeating-linear-gradient(90deg,transparent,transparent 4px,${t.cpuBg}55 4px,${t.cpuBg}55 5px)`,
                  }} />
                </div>
              </div>

              {/* VRAM */}
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                  <span style={{ color: t.dimText, fontSize: 10 }}>VRAM</span>
                  <span style={{ color: memCol, fontSize: 10 }}>{gpu.mem_used_gb}/{gpu.mem_total_gb} GB</span>
                </div>
                <div style={{ height: 8, background: t.cpuBorder + "33", borderRadius: 3, position: "relative" }}>
                  <div style={{
                    position: "absolute", top: 0, left: 0, bottom: 0,
                    width: `${gpu.mem_pct}%`, borderRadius: 3,
                    background: memCol, transition: "width 1s ease",
                  }} />
                </div>
              </div>

              {/* Temp + Power row */}
              <div style={{ display: "flex", gap: 12, marginTop: 2 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                  <span style={{ fontSize: 12 }}>🌡</span>
                  <span style={{ color: tCol, fontWeight: 700, fontSize: 11 }}>{gpu.temp_c}°C</span>
                </div>
                {gpu.power_w != null && (
                  <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                    <span style={{ fontSize: 12 }}>⚡</span>
                    <span style={{ color: t.dimText, fontSize: 11 }}>{gpu.power_w} W</span>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Footer */}
      <div style={{
        padding: "6px 12px 8px", borderTop: `1px solid ${t.cpuBorder}`,
        display: "flex", justifyContent: "space-between", alignItems: "center",
      }}>
        <span style={{ color: t.cpuLabel, fontWeight: 700, fontSize: 12 }}>INFERENCE</span>
        <span style={{ color: "#22c55e", fontSize: 11 }}>⚡ GPU Active</span>
      </div>
    </div>
  );
}

// ── Rancher logo ──────────────────────────────────────────────────────────
const RancherLogo = ({ size = 38 }) => (
  <img src="/rancher-logo.svg" alt="Rancher" width={size} height={size}
    style={{ objectFit: "contain", display: "block" }} />
);

// ── Theme switcher ────────────────────────────────────────────────────────
function ThemeSwitcher({ current, onChange, t }) {
  return (
    <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
      {Object.entries(THEMES).map(([key, theme]) => (
        <button key={key} onClick={() => onChange(key)} title={theme.name}
          style={{
            width: 18, height: 18, borderRadius: "50%", border: "none",
            cursor: "pointer", padding: 0,
            background: key === "midnight"
              ? "linear-gradient(135deg,#38bdf8,#818cf8)"
              : key === "singapore"
              ? "linear-gradient(135deg,#ef4444,#fbbf24)"
              : "linear-gradient(135deg,#f96702,#fdd0aa)",
            outline: current === key ? `2px solid ${t.accent}` : `2px solid ${t.border}`,
            outlineOffset: 2, transition: "outline 0.2s",
          }} />
      ))}
      <span style={{ fontSize: 10, color: t.dimText, fontFamily: FONT, marginLeft: 2 }}>
        {THEMES[current].name}
      </span>
    </div>
  );
}

// ── CPU/GPU tooltip ───────────────────────────────────────────────────────
function ModelTooltip({ model, numGpu, t }) {
  const [show, setShow] = useState(false);
  const isGpu = numGpu > 0;
  return (
    <div style={{ position: "relative", display: "inline-flex", alignItems: "center" }}
      onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)}
    >
      <div style={{
        width: 17, height: 17, borderRadius: "50%",
        background: t.headerBg, border: `1px solid ${t.border}`,
        display: "flex", alignItems: "center", justifyContent: "center",
        cursor: "help", fontSize: 11, color: t.dimText, fontFamily: FONT, flexShrink: 0,
      }}>?</div>
      {show && (
        <div style={{
          position: "absolute", top: 24, left: 0, zIndex: 200,
          background: t.tooltipBg, border: `1px solid ${t.border}`,
          borderRadius: 10, padding: "12px 14px", width: 270,
          fontFamily: FONT2, fontSize: 12, color: t.tooltipText, lineHeight: 1.65,
          boxShadow: "0 8px 32px #0006",
        }}>
          <div style={{ color: t.accent2, fontWeight: 600, marginBottom: 6 }}>
            {isGpu ? "🖥️  GPU Inference" : "⚙️  CPU Inference"}
          </div>
          <div>Model: <span style={{ color: t.bodyText }}>{model}</span></div>
          {!isGpu ? (
            <div style={{ marginTop: 8, color: "#f59e0b", borderTop: `1px solid ${t.border}`, paddingTop: 8 }}>
              ⚠ Running on CPU. Responses may take 30–90 seconds. Please be patient.
            </div>
          ) : (
            <div style={{ marginTop: 8, color: "#22c55e", borderTop: `1px solid ${t.border}`, paddingTop: 8 }}>
              ✓ GPU acceleration active. Responses will be faster.
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Tool badge ────────────────────────────────────────────────────────────
function ToolBadge({ tool, t }) {
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 4,
      padding: "2px 8px", borderRadius: 999, fontSize: 10,
      background: t.headerBg, border: `1px solid ${t.accent}44`, color: t.accent2,
      fontFamily: FONT, fontWeight: 700,
    }}>
      {tool === "search_documentation" ? "📄" : "⚙"} {tool}
    </span>
  );
}

// ── Status ticker ─────────────────────────────────────────────────────────
function StatusTicker({ updates, t }) {
  const [visible, setVisible] = useState(0);
  useEffect(() => {
    setVisible(0);
    if (!updates.length) return;
    const timers = updates.map((_, i) => setTimeout(() => setVisible(i + 1), i * 1800));
    return () => timers.forEach(clearTimeout);
  }, [updates]);
  const shown = updates.slice(0, visible);
  return (
    <div style={{
      background: t.statusBg, border: `1px solid ${t.border}`,
      borderRadius: 12, padding: "12px 16px", marginBottom: 4,
      fontFamily: FONT, fontSize: 12, display: "flex", flexDirection: "column", gap: 6,
    }}>
      <div style={{ color: t.dimText, fontSize: 10, letterSpacing: "0.12em", marginBottom: 2 }}>AGENT STATUS</div>
      {shown.map((u, i) => (
        <div key={i} style={{
          display: "flex", alignItems: "center", gap: 8,
          color: i === shown.length - 1 ? t.accent2 : t.dimText,
          animation: "fadeUp 0.3s ease",
        }}>
          <span style={{ opacity: i === shown.length - 1 ? 1 : 0.35 }}>
            {i === shown.length - 1 ? "›" : "✓"}
          </span>
          {u}
        </div>
      ))}
      <div style={{ display: "flex", gap: 4, marginTop: 2 }}>
        {[0,1,2].map(i => (
          <div key={i} style={{
            width: 4, height: 4, borderRadius: "50%", background: t.dimText,
            animation: `pls 1.2s ${i * 0.2}s infinite`,
          }} />
        ))}
      </div>
    </div>
  );
}

// ── Message ───────────────────────────────────────────────────────────────
function Message({ msg, t }) {
  if (msg.role === "user") return (
    <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 16 }}>
      <div style={{
        maxWidth: "72%", background: t.userBg, border: `1px solid ${t.userBorder}`,
        borderRadius: "14px 14px 2px 14px", padding: "11px 16px",
        fontSize: 14, color: t.bodyText, fontFamily: FONT2, lineHeight: 1.6,
      }}>{msg.text}</div>
    </div>
  );
  if (msg.role === "thinking") return <StatusTicker updates={msg.updates || []} t={t} />;
  if (msg.role === "assistant") return (
    <div style={{ display: "flex", gap: 10, alignItems: "flex-start", marginBottom: 20 }}>
      <div style={{
        width: 28, height: 28, borderRadius: 8, background: t.headerBg,
        border: `1px solid ${t.accent}`, display: "flex", alignItems: "center",
        justifyContent: "center", flexShrink: 0, marginTop: 2, boxShadow: `0 0 10px ${t.accent}33`,
      }}>
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
          <path d="M12 2L3 7v10l9 5 9-5V7L12 2z" stroke={t.accent} strokeWidth="1.8" strokeLinejoin="round"/>
          <circle cx="12" cy="12" r="3" stroke={t.accent} strokeWidth="1.8"/>
        </svg>
      </div>
      <div style={{ flex: 1 }}>
        <div style={{
          background: t.msgBg, border: `1px solid ${t.border}`,
          borderRadius: "2px 14px 14px 14px", padding: "14px 18px",
          fontSize: 14, color: t.bodyText, fontFamily: FONT2, lineHeight: 1.75,
          whiteSpace: "pre-wrap",
        }}>{msg.text}</div>
        {msg.elapsed != null && (
          <div style={{ fontSize: 10, color: t.dimText, fontFamily: FONT, marginTop: 5 }}>
            ⏱ {msg.elapsed}s total
          </div>
        )}
        {msg.tools && msg.tools.length > 0 && (
          <div style={{ marginTop: 6, display: "flex", flexWrap: "wrap", gap: 5, alignItems: "center" }}>
            <span style={{ fontSize: 10, color: t.dimText, fontFamily: FONT }}>called:</span>
            {[...new Set(msg.tools)].map((tool, i) => <ToolBadge key={i} tool={tool} t={t} />)}
            <span style={{ fontSize: 10, color: t.dimText, fontFamily: FONT }}>
              · {msg.iterations} iteration{msg.iterations !== 1 ? "s" : ""}
            </span>
          </div>
        )}
      </div>
    </div>
  );
  return null;
}

// ── App ───────────────────────────────────────────────────────────────────
export default function App() {
  const [messages, setMessages]   = useState([]);
  const [input, setInput]         = useState("");
  const [loading, setLoading]     = useState(false);
  const [model, setModel]         = useState("qwen2.5:7b");
  const [numGpu, setNumGpu]       = useState(0);
  const [themeName, setThemeName] = useState("midnight");
  const [sysMetrics, setSysMetrics] = useState(null);
  const [cpuHistory, setCpuHistory] = useState([]);
  const t = THEMES[themeName];
  const bottomRef  = useRef(null);
  const metricsRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then(r => r.json())
      .then(d => { setModel(d.model || "qwen2.5:7b"); if (d.num_gpu !== undefined) setNumGpu(d.num_gpu); })
      .catch(() => {});
  }, []);

  // ── Metrics polling (lifted here so GPU panel can share the same data) ──
  useEffect(() => {
    const poll = async () => {
      try {
        const r = await fetch(`${API_URL}/metrics`);
        if (!r.ok) return;
        const d = await r.json();
        setSysMetrics(d);
        setCpuHistory(h => [...h.slice(-59), d.cpu_total]);
      } catch {}
    };
    poll();
    metricsRef.current = setInterval(poll, 1500);
    return () => clearInterval(metricsRef.current);
  }, []);

  async function send(queryOverride) {
    const query = queryOverride || input.trim();
    if (!query || loading) return;
    setInput("");
    setLoading(true);

    const q = query.toLowerCase();
    let contextHint = "🔍 Analysing query...";
    if (q.includes("longhorn"))    contextHint = "🔍 Checking Longhorn storage...";
    else if (q.includes("vault"))  contextHint = "🔍 Checking vault-system namespace...";
    else if (q.includes("health")) contextHint = "🔍 Scanning cluster health...";
    else if (q.includes("log"))    contextHint = "🔍 Fetching pod logs...";
    else if (q.includes("node"))   contextHint = "🔍 Checking node status...";
    else if (q.includes("event"))  contextHint = "🔍 Fetching cluster events...";
    else if (q.includes("deploy")) contextHint = "🔍 Checking deployments...";

    setMessages(m => [...m,
      { role: "user", text: query },
      { role: "thinking", updates: [`🤖 Model: ${model}`, contextHint] },
    ]);

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: query }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || "Request failed");
      const data = await res.json();
      setMessages(m => {
        const filtered = m.filter(x => x.role !== "thinking");
        return [...filtered, {
          role: "assistant", text: data.response,
          tools: data.tools_used, iterations: data.iterations,
          elapsed: data.elapsed_seconds, updates: data.status_updates || [],
        }];
      });
    } catch (err) {
      setMessages(m => {
        const filtered = m.filter(x => x.role !== "thinking");
        return [...filtered, {
          role: "assistant",
          text: `⚠ Error: ${err.message}\n\nMake sure the backend is running:\n  cd backend && uvicorn main:app --port 8000`,
          tools: [], iterations: 0,
        }];
      });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{
      background: t.bg, height: "100vh", display: "flex", flexDirection: "column",
      fontFamily: FONT2, color: t.textMain,
      transition: "background 0.3s", overflow: "hidden",
    }}>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet" />
      <style>{`
        @keyframes pls { 0%,100%{opacity:1} 50%{opacity:0.2} }
        @keyframes fadeUp { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
        @keyframes shimmer { 0%{background-position:-500px 0} 100%{background-position:500px 0} }
        ::-webkit-scrollbar{width:4px} ::-webkit-scrollbar-track{background:${t.bg}}
        ::-webkit-scrollbar-thumb{background:${t.border};border-radius:4px}
        textarea:focus{outline:none}
        .shiny {
          background: ${t.shimmer};
          background-size: 500px 100%;
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          animation: shimmer 4s linear infinite;
        }
      `}</style>

      {/* ── Header ── */}
      <div style={{
        padding: "10px 24px", borderBottom: `1px solid ${t.border}`,
        background: t.headerBg, display: "flex", alignItems: "center", gap: 14, flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
          <div style={{ borderRadius: 10, overflow: "hidden" }}>
            <RancherLogo size={38} />
          </div>
          <img src="/k8s-logo.svg" alt="Kubernetes" width={34} height={34}
            style={{ objectFit: "contain", display: "block" }} />
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            {!t.dark ? (
              <span style={{ fontSize: 20, fontWeight: 700, fontFamily: FONT, letterSpacing: "-0.02em", color: "#f96702" }}>
                Cloudera ECS AI Ops
              </span>
            ) : (
              <span className="shiny" style={{ fontSize: 20, fontWeight: 700, fontFamily: FONT, letterSpacing: "-0.02em" }}>
                Cloudera ECS AI Ops
              </span>
            )}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: t.dimText, fontFamily: FONT }}>
            Air-gapped · {model}
            <ModelTooltip model={model} numGpu={numGpu} t={t} />
          </div>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 16 }}>
          <a href="mailto:dennislee@cloudera.com"
            style={{ fontSize: 11, color: t.dimText, fontFamily: FONT, textDecoration: "none", transition: "color 0.2s" }}
            onMouseEnter={e => e.target.style.color = t.accent2}
            onMouseLeave={e => e.target.style.color = t.dimText}
          >✉ dennislee@cloudera.com</a>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ fontSize: 10, color: t.dimText, fontFamily: FONT }}>THEME</span>
            <ThemeSwitcher current={themeName} onChange={setThemeName} t={t} />
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
            <div style={{
              width: 6, height: 6, borderRadius: "50%",
              background: loading ? "#f59e0b" : "#22c55e",
              boxShadow: `0 0 8px ${loading ? "#f59e0b" : "#22c55e"}`,
              animation: loading ? "pls 1s infinite" : "none",
            }} />
            <span style={{ fontSize: 11, color: t.dimText, fontFamily: FONT }}>
              {loading ? "thinking" : "ready"}
            </span>
          </div>
        </div>
      </div>

      {/* ── Body: chat + right panel ── */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>

        {/* ── Chat column ── */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          <div style={{ flex: 1, overflow: "auto", padding: "20px 24px" }}>
            {messages.map((msg, i) => (
              <div key={i} style={{ animation: "fadeUp 0.25s ease" }}>
                <Message msg={msg} t={t} />
              </div>
            ))}
            {messages.length === 0 && (
              <div style={{ marginTop: 8 }}>
                <div style={{ fontSize: 11, color: t.dimText, fontFamily: FONT, marginBottom: 10, letterSpacing: "0.1em" }}>
                  EXAMPLE QUERIES
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 7 }}>
                  {EXAMPLES.map((ex, i) => (
                    <button key={i} onClick={() => send(ex)} disabled={loading}
                      style={{
                        background: t.headerBg, border: `1px solid ${t.border}`,
                        borderRadius: 8, padding: "7px 12px", fontSize: 12,
                        color: t.dimText, cursor: "pointer", fontFamily: FONT, transition: "all 0.2s",
                      }}
                      onMouseEnter={e => { e.currentTarget.style.borderColor = t.accent; e.currentTarget.style.color = t.accent2; }}
                      onMouseLeave={e => { e.currentTarget.style.borderColor = t.border; e.currentTarget.style.color = t.dimText; }}
                    >{ex}</button>
                  ))}
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {/* ── Input ── */}
          <div style={{ borderTop: `1px solid ${t.border}`, padding: "14px 24px", background: t.headerBg, display: "flex", gap: 10 }}>
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
              placeholder="Ask about cluster health, Longhorn storage, pod issues, logs, events... (Enter to send)"
              disabled={loading} rows={2}
              style={{
                flex: 1, background: t.inputBg,
                border: `1px solid ${input ? t.accent + "44" : t.border}`,
                borderRadius: 10, padding: "10px 14px",
                color: t.inputText,
                fontSize: 14, fontFamily: FONT2, resize: "none",
                transition: "border-color 0.2s", lineHeight: 1.5,
              }}
            />
            <button onClick={() => send()} disabled={loading || !input.trim()}
              style={{
                background: loading ? t.headerBg : t.msgBg,
                border: `1px solid ${loading ? t.border : t.accent}`,
                borderRadius: 10, padding: "10px 18px",
                color: loading ? t.dimText : t.accent2,
                fontSize: 13, fontFamily: FONT, fontWeight: 700,
                cursor: loading ? "not-allowed" : "pointer",
                transition: "all 0.2s", alignSelf: "stretch",
              }}
            >
              {loading ? "..." : "→ Send"}
            </button>
          </div>
        </div>

        {/* ── Right: CPU + GPU panels ── */}
        <div style={{ display: "flex", flexShrink: 0 }}>
          <CpuPanel t={t} metrics={sysMetrics} history={cpuHistory} />
          <GpuPanel gpus={sysMetrics?.gpus} t={t} />
        </div>
      </div>
    </div>
  );
}
