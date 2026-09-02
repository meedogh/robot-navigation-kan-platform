"use client";

import { useEffect, useRef, useState } from "react";

const SIZE = 560;
const WORLD = 20; // matches env v2 world_size; coords range [-WORLD/2, WORLD/2]

function toCanvas(v: number, world: number) {
  return ((v + world / 2) / world) * SIZE;
}

export default function Live() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [model, setModel] = useState<"kan" | "mlp">("kan");
  const [liveMode, setLiveMode] = useState(false);
  const [status, setStatus] = useState("idle");
  const [stats, setStats] = useState({ reward: 0, step: 0, action: "-", reached: false, collision: false });
  const [stream, setStream] = useState<{ phase: string; model: string } | null>(null);

  const ACTION_NAMES = ["Forward", "Forward-Left", "Forward-Right", "Turn Left", "Turn Right", "Stop"];

  function draw(frame: any) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Frames may come from a job trained with a custom world size.
    const world =
      typeof frame.world_size === "number" && frame.world_size > 0
        ? frame.world_size
        : WORLD;

    ctx.clearRect(0, 0, SIZE, SIZE);

    // grid
    ctx.strokeStyle = "#101623";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const p = (i / 10) * SIZE;
      ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, SIZE); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(SIZE, p); ctx.stroke();
    }

    // obstacles (v2 env has multiple obstacles with different radii)
    ctx.fillStyle = "#ff5c7a";
    for (const ob of frame.obstacles ?? []) {
      const r = Math.max(5, ob.radius * (SIZE / world));
      ctx.beginPath();
      ctx.arc(toCanvas(ob.x, world), toCanvas(ob.y, world), r, 0, Math.PI * 2);
      ctx.fill();
    }

    // target
    ctx.fillStyle = "#38d39f";
    ctx.beginPath();
    ctx.arc(toCanvas(frame.target_x, world), toCanvas(frame.target_y, world), 14, 0, Math.PI * 2);
    ctx.fill();

    // robot (triangle pointing in heading direction)
    const rx = toCanvas(frame.robot_x, world);
    const ry = toCanvas(frame.robot_y, world);
    const ang = frame.robot_angle;
    ctx.save();
    ctx.translate(rx, ry);
    ctx.rotate(-ang);
    ctx.fillStyle = "#4f8cff";
    ctx.beginPath();
    ctx.moveTo(16, 0);
    ctx.lineTo(-10, 10);
    ctx.lineTo(-10, -10);
    ctx.closePath();
    ctx.fill();
    ctx.restore();

    setStats({
      reward: frame.episode_reward,
      step: frame.step,
      action: ACTION_NAMES[frame.action] ?? "-",
      reached: frame.reached_target,
      collision: frame.collision,
    });
    setStream(
      frame.phase
        ? { phase: frame.phase, model: frame.model_name ?? frame.model ?? "" }
        : null
    );
  }

  function connect() {
    disconnect();
    setStatus("connecting");
    const ws = new WebSocket("ws://127.0.0.1:8000/ws/live");
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify(liveMode ? { live: true } : { model }));
      setStatus(liveMode ? "live (active job)" : "live");
    };
    ws.onmessage = (ev) => {
      const frame = JSON.parse(ev.data);
      if (frame.error) {
        setStatus("error: " + frame.error);
        return;
      }
      draw(frame);
    };
    ws.onclose = () => setStatus("closed");
  }

  function disconnect() {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }

  useEffect(() => () => disconnect(), []);

  return (
    <div>
      <h1>Live Simulation</h1>
      <p className="subtitle">Watch the trained agent navigate in real time over WebSocket</p>

      <div className="card">
        <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 16, flexWrap: "wrap" }}>
          <label className="toggle-row">
            <input
              type="checkbox"
              checked={liveMode}
              onChange={(e) => setLiveMode(e.target.checked)}
            />
            Stream active job (training/eval)
          </label>
          <select
            className="secondary"
            value={model}
            disabled={liveMode}
            onChange={(e) => setModel(e.target.value as any)}
            style={{ padding: "10px 14px", borderRadius: 10, background: "#1a2233", color: "#e6ebf4", border: "1px solid #232d42", opacity: liveMode ? 0.5 : 1 }}
          >
            <option value="kan">KAN</option>
            <option value="mlp">MLP</option>
          </select>
          <button onClick={connect}>Start</button>
          <button className="secondary" onClick={disconnect}>Stop</button>
          <span className="muted">status: {status}</span>
          {liveMode && stream && (
            <span className="pill running">
              {stream.model ? `${stream.phase}: ${stream.model}` : stream.phase}
            </span>
          )}
        </div>

        <canvas ref={canvasRef} className="live" width={SIZE} height={SIZE} />

        <div className="legend">
          <span className="robot">Robot</span>
          <span className="target">Target</span>
          <span className="obstacle">Obstacle</span>
        </div>

        <div className="grid cols-3" style={{ marginTop: 20 }}>
          <div className="stat">
            <div className="label">Episode Reward</div>
            <div className="value">{stats.reward.toFixed(2)}</div>
          </div>
          <div className="stat">
            <div className="label">Action</div>
            <div className="value" style={{ fontSize: 20 }}>{stats.action}</div>
          </div>
          <div className="stat">
            <div className="label">Status</div>
            <div className="value" style={{ fontSize: 20 }}>
              {stats.reached ? "Reached" : stats.collision ? "Collision" : "Running"}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}