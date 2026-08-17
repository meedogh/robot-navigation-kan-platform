"use client";

import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";
import { getJSON } from "../../lib/api";

export default function Training() {
  const [data, setData] = useState<{ mlp: any[]; kan: any[] }>({ mlp: [], kan: [] });
  const [error, setError] = useState("");

  useEffect(() => {
    getJSON<any>("/api/results/training")
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  const merged = (data.mlp || []).map((m, i) => ({
    step: m.training_step,
    mlp_reward: m.mean_reward,
    kan_reward: data.kan[i]?.mean_reward,
    mlp_success: (m.success_rate ?? 0) * 100,
    kan_success: (data.kan[i]?.success_rate ?? 0) * 100,
    mlp_collision: (m.collision_rate ?? 0) * 100,
    kan_collision: (data.kan[i]?.collision_rate ?? 0) * 100,
  }));

  if (error) return <div className="card">Error: {error}</div>;

  return (
    <div>
      <h1>Training Comparison</h1>
      <p className="subtitle">Reward, success rate and collision rate over training steps</p>

      <div className="card">
        <h2>Mean Reward</h2>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={merged}>
            <CartesianGrid stroke="#232d42" />
            <XAxis dataKey="step" stroke="#8b96ad" />
            <YAxis stroke="#8b96ad" />
            <Tooltip contentStyle={{ background: "#121826", border: "1px solid #232d42" }} />
            <Legend />
            <Line type="monotone" dataKey="mlp_reward" stroke="#4f8cff" name="MLP" dot={false} />
            <Line type="monotone" dataKey="kan_reward" stroke="#38d39f" name="KAN" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="card">
        <h2>Success Rate (%)</h2>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={merged}>
            <CartesianGrid stroke="#232d42" />
            <XAxis dataKey="step" stroke="#8b96ad" />
            <YAxis stroke="#8b96ad" domain={[0, 100]} />
            <Tooltip contentStyle={{ background: "#121826", border: "1px solid #232d42" }} />
            <Legend />
            <Line type="monotone" dataKey="mlp_success" stroke="#4f8cff" name="MLP" dot={false} />
            <Line type="monotone" dataKey="kan_success" stroke="#38d39f" name="KAN" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="card">
        <h2>Collision Rate (%)</h2>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={merged}>
            <CartesianGrid stroke="#232d42" />
            <XAxis dataKey="step" stroke="#8b96ad" />
            <YAxis stroke="#8b96ad" />
            <Tooltip contentStyle={{ background: "#121826", border: "1px solid #232d42" }} />
            <Legend />
            <Line type="monotone" dataKey="mlp_collision" stroke="#4f8cff" name="MLP" dot={false} />
            <Line type="monotone" dataKey="kan_collision" stroke="#38d39f" name="KAN" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}