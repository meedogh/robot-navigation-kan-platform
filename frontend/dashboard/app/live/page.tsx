"use client";

import { useEffect, useRef, useState } from "react";
import { getJSON } from "../../lib/api";

// Derive the WebSocket URL from NEXT_PUBLIC_API_URL (same default as lib/api.ts),
// converting the http(s) scheme to ws(s).
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const WS_BASE_URL = API_BASE_URL.replace(/^http/, "ws");

const SIZE = 560;
const WORLD = 20; // matches env v2 world_size; coords range [-WORLD/2, WORLD/2]

function toCanvas(v: number, world: number) {
  return ((v + world / 2) / world) * SIZE;
}

// Y mapping with the world's +y rendered upward (the world uses y-up math;
// without the flip the robot's heading would be mirrored against its motion).
function toCanvasY(v: number, world: number) {
  return SIZE - ((v + world / 2) / world) * SIZE;
}

const ACTION_NAMES = [
  "Forward",
  "Forward-Left",
  "Forward-Right",
  "Turn Left",
  "Turn Right",
  "Stop",
];

// Manual play keyboard bindings (WASD / QE / arrows).
const KEY_ACTIONS: Record<string, number> = {
  w: 0, ArrowUp: 0,
  q: 1,
  e: 2,
  a: 3, ArrowLeft: 3,
  d: 4, ArrowRight: 4,
  s: 5, ArrowDown: 5,
};

type Mode = "agent" | "job" | "manual";

type Frame = {
  env_label?: string;
  manual?: boolean;
  q_values?: number[] | null;
  suggested_action?: number | null;
  sensors?: number[] | null;
  sensor_range?: number;
  [key: string]: any;
};

export default function Live() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const trailRef = useRef<{ x: number; y: number }[]>([]);
  const modeRef = useRef<Mode>("agent");
  const [mode, setMode] = useState<Mode>("agent");
  const [model, setModel] = useState<"kan" | "mlp">("kan");
  const [status, setStatus] = useState("idle");
  const [stats, setStats] = useState({ reward: 0, step: 0, action: "-", reached: false, collision: false, distance: 0, trainingStep: 0 });
  const [frame, setFrame] = useState<Frame | null>(null);
  const [lastInput, setLastInput] = useState<string>("-");

  modeRef.current = mode;

  function draw(f: Frame) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Frames may come from a job trained with a custom world size.
    const world =
      typeof f.world_size === "number" && f.world_size > 0 ? f.world_size : WORLD;
    const scale = SIZE / world;

    ctx.clearRect(0, 0, SIZE, SIZE);

    // grid
    ctx.strokeStyle = "#101623";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const p = (i / 10) * SIZE;
      ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, SIZE); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(SIZE, p); ctx.stroke();
    }

    // target
    ctx.fillStyle = "#38d39f";
    ctx.beginPath();
    ctx.arc(toCanvas(f.target_x, world), toCanvasY(f.target_y, world), 14, 0, Math.PI * 2);
    ctx.fill();

    // obstacles (v2/v3 envs have multiple obstacles with different shapes)
    ctx.fillStyle = "#ff5c7a";
    for (const ob of f.obstacles ?? []) {
      if (ob.shape === "rect" && typeof ob.width === "number") {
        ctx.save();
        ctx.translate(toCanvas(ob.x, world), toCanvasY(ob.y, world));
        ctx.rotate(-(ob.angle ?? 0));
        const w = Math.max(4, ob.width * scale);
        const h = Math.max(4, ob.height * scale);
        ctx.fillRect(-w / 2, -h / 2, w, h);
        ctx.restore();
      } else {
        const r = Math.max(5, ob.radius * scale);
        ctx.beginPath();
        ctx.arc(toCanvas(ob.x, world), toCanvasY(ob.y, world), r, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // sensor beams (front / left / right), length = normalized reading * range
    if (Array.isArray(f.sensors)) {
      const range =
        typeof f.sensor_range === "number" && f.sensor_range > 0
          ? f.sensor_range
          : 0.5 * world;
      const angles = [0, (2 * Math.PI) / 3, -(2 * Math.PI) / 3];
      const rx = toCanvas(f.robot_x, world);
      const ry = toCanvasY(f.robot_y, world);
      f.sensors.forEach((s, i) => {
        const len = Math.max(2, s * range * scale);
        // Head in the robot's world heading; negate sin for the y-up render.
        const a = f.robot_angle + angles[i];
        const ex = rx + Math.cos(a) * len;
        const ey = ry - Math.sin(a) * len;
        ctx.strokeStyle = s < 0.35 ? "rgba(255,92,122,0.8)" : "rgba(79,140,255,0.45)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(rx, ry);
        ctx.lineTo(ex, ey);
        ctx.stroke();
      });
    }

    // trajectory trail (fading), reset each episode
    if (f.done) {
      trailRef.current = [];
    } else {
      const trail = trailRef.current;
      trail.push({ x: f.robot_x, y: f.robot_y });
      while (trail.length > 48) trail.shift();
      for (let i = 0; i < trail.length - 1; i++) {
        const alpha = 0.15 + 0.55 * (i / Math.max(1, trail.length - 1));
        ctx.strokeStyle = `rgba(56,211,159,${alpha})`;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(toCanvas(trail[i].x, world), toCanvasY(trail[i].y, world));
        ctx.lineTo(toCanvas(trail[i + 1].x, world), toCanvasY(trail[i + 1].y, world));
        ctx.stroke();
      }
    }

    // robot (triangle pointing in heading direction, matching its movement)
    const rx = toCanvas(f.robot_x, world);
    const ry = toCanvasY(f.robot_y, world);
    const ang = f.robot_angle;
    ctx.save();
    ctx.translate(rx, ry);
    ctx.rotate(-ang);
    ctx.fillStyle = f.manual ? "#ffd166" : "#4f8cff";
    ctx.beginPath();
    ctx.moveTo(16, 0);
    ctx.lineTo(-10, 10);
    ctx.lineTo(-10, -10);
    ctx.closePath();
    ctx.fill();
    ctx.restore();

    setStats({
      reward: f.episode_reward,
      step: f.step,
      action: f.action >= 0 ? ACTION_NAMES[f.action] ?? "-" : "-",
      reached: f.reached_target,
      collision: f.collision,
      distance: Math.hypot(f.target_x - f.robot_x, f.target_y - f.robot_y),
      trainingStep: f.training_step ?? 0,
    });
    setFrame(f);
  }

  function send(msg: object) {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }

  function disconnect() {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setStatus("closed");
  }

  async function connect() {
    disconnect();
    setStatus("connecting");

    const currentMode = modeRef.current;

    let opening: object;
    if (currentMode === "job") {
      opening = { live: true };
    } else if (currentMode === "manual") {
      // Prefer the environment from the selected model's most recent run
      // config - that includes any custom map (v3) the user designed.
      let envConfig: any = undefined;
      try {
        const last = await getJSON<Record<string, any>>(
          `/api/training/last-config/${model}`
        );
        if (last && Object.keys(last).length > 0) {
          envConfig = last; // flat config; backend flattens it via env_factory
        }
      } catch {
        // no previous run - play the default env instead
      }
      opening = { manual: true, model, env_config: envConfig };
    } else {
      opening = { model };
    }

    const ws = new WebSocket(`${WS_BASE_URL}/ws/live`);
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify(opening));
      setStatus(
        currentMode === "manual"
          ? "manual — drive with WASD/QE or the buttons"
          : currentMode === "job"
          ? "live (active job)"
          : "live"
      );
    };
    ws.onmessage = (ev) => {
      const f: Frame = JSON.parse(ev.data);
      if (f.error) {
        setStatus("error: " + f.error);
        return;
      }
      draw(f);
    };
    ws.onclose = () => setStatus("closed");
  }

  // Keyboard input for manual play.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (mode !== "manual") return;
      const action = KEY_ACTIONS[e.key];
      if (action === undefined) return;
      e.preventDefault();
      setLastInput(`${e.key} → ${ACTION_NAMES[action]}`);
      send({ action });
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mode]);

  useEffect(() => () => disconnect(), []);

  const qValues = frame?.q_values ?? null;
  const qMax = qValues ? Math.max(...qValues.map((q) => Math.abs(q)), 1e-6) : 1;
  const suggested = frame?.suggested_action;

  return (
    <div>
      <h1>Live Simulation</h1>
      <p className="subtitle">
        Watch the trained agent, stream active jobs — or grab the keyboard and
        drive the robot yourself on the very same map
      </p>

      <div className="card">
        <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 16, flexWrap: "wrap" }}>
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as Mode)}
            style={{ padding: "10px 14px", borderRadius: 10, background: "#1a2233", color: "#e6ebf4", border: "1px solid #232d42" }}
          >
            <option value="agent">Agent (saved model plays)</option>
            <option value="job">Active job (live training)</option>
            <option value="manual">Manual play (you drive)</option>
          </select>
          <select
            className="secondary"
            value={model}
            disabled={mode === "job"}
            onChange={(e) => setModel(e.target.value as any)}
            style={{ padding: "10px 14px", borderRadius: 10, background: "#1a2233", color: "#e6ebf4", border: "1px solid #232d42", opacity: mode === "job" ? 0.5 : 1 }}
          >
            <option value="kan">KAN</option>
            <option value="mlp">MLP</option>
          </select>
          <button onClick={connect}>Start</button>
          <button className="secondary" onClick={disconnect}>Stop</button>
          <span className="muted">status: {status}</span>
          {frame?.env_label && (
            <span className="pill running" title="Environment used for this simulation">
              env: {frame.env_label}
            </span>
          )}
        </div>

        <canvas ref={canvasRef} className="live" width={SIZE} height={SIZE} />

        <div className="legend">
          <span className="robot">Robot {mode === "manual" ? "(you)" : ""}</span>
          <span className="target">Target</span>
          <span className="obstacle">Obstacle</span>
          <span className="muted">beam = obstacle sensor (short + red = close)</span>
        </div>

        <div className="grid cols-3" style={{ marginTop: 20 }}>
          <div className="stat">
            <div className="label">Episode Reward</div>
            <div className="value">{stats.reward.toFixed(2)}</div>
          </div>
          <div className="stat">
            <div className="label">{mode === "manual" ? "Your Input" : "Action"}</div>
            <div className="value" style={{ fontSize: 20 }}>
              {mode === "manual" && lastInput !== "-" ? lastInput : stats.action}
            </div>
          </div>
          <div className="stat">
            <div className="label">Status</div>
            <div className="value" style={{ fontSize: 20 }}>
              {stats.reached ? "Reached" : stats.collision ? "Collision" : "Running"}
            </div>
          </div>
        </div>

        <div className="muted" style={{ marginTop: 8 }}>
          distance to target: {stats.distance.toFixed(2)} · episode step {stats.step}
          {stats.trainingStep > 0 && (
            <> · training progress: step <b>{stats.trainingStep.toLocaleString()}</b></>
          )}
        </div>

        {/* Obstacle sensor readout (numbers under the canvas beams) */}
        {Array.isArray(frame?.sensors) && (
          <div style={{ marginTop: 12 }}>
            <div className="muted">Obstacle sensors (normalized, 1 = clear)</div>
            <div style={{ display: "flex", gap: 10 }}>
              {["front", "left", "right"].map((name, i) => {
                const v = frame.sensors![i];
                return (
                  <div key={name} style={{ flex: 1, minWidth: 90 }}>
                    <span style={{ fontSize: 12, color: "#8b93a7" }}>{name}</span>
                    <div style={{ background: "#121826", borderRadius: 6, height: 8 }}>
                      <div
                        style={{
                          width: `${Math.min(100, v * 100)}%`,
                          height: 8,
                          borderRadius: 6,
                          background: v < 0.35 ? "#ff5c7a" : "#38d39f",
                        }}
                      />
                    </div>
                    <span style={{ fontSize: 10, color: "#8b93a7" }}>{v.toFixed(2)}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Q-value bars: what the network "thinks" in the current state */}
        {qValues && (
          <div style={{ marginTop: 16 }}>
            <div className="muted" style={{ marginBottom: 6 }}>
              Q-values in this state
              {mode === "manual" && suggested !== null && suggested !== undefined && (
                <> — agent would choose: <b>{ACTION_NAMES[suggested]}</b></>
              )}
            </div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              {qValues.map((q, i) => {
                const widthPct = 8 + (Math.abs(q) / qMax) * 92;
                const isBest = i === suggested;
                return (
                  <div key={i} style={{ flex: 1, minWidth: 90 }}>
                    <div style={{ fontSize: 11, marginBottom: 2, color: isBest ? "#38d39f" : "#8b93a7" }}>
                      {ACTION_NAMES[i] ?? i}
                    </div>
                    <div style={{ background: "#121826", borderRadius: 6, height: 10 }}>
                      <div
                        style={{
                          width: `${widthPct}%`,
                          height: 10,
                          borderRadius: 6,
                          background: isBest ? "#38d39f" : "#4f8cff",
                          opacity: isBest ? 1 : 0.65,
                        }}
                      />
                    </div>
                    <div style={{ fontSize: 10, color: "#8b93a7", marginTop: 2 }}>{q.toFixed(2)}</div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Manual play controls: on-screen buttons mirror the keyboard */}
        {mode === "manual" && (
          <div style={{ marginTop: 16 }}>
            <div className="muted" style={{ marginBottom: 6 }}>
              Drive the robot (keyboard: W/Q/E/A/D/S or arrows)
            </div>
            <div className="actions" style={{ flexWrap: "wrap" }}>
              {ACTION_NAMES.map((name, i) => (
                <button key={name} onClick={() => { setLastInput(`button → ${name}`); send({ action: i }); }}>
                  {name}
                </button>
              ))}
              <button className="secondary" onClick={() => send({ reset: true })}>
                Reset Episode
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}