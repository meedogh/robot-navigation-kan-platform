"use client";

import { useEffect, useState } from "react";
import { getJSON } from "../lib/api";

type Row = {
  model_name?: string;
  model_type?: string;
  success_rate?: number;
  collision_rate?: number;
  mean_reward?: number;
  mean_steps?: number;
  mean_final_distance?: number;
  parameter_count?: number;
  avg_inference_ms?: number;
};

export default function Overview() {
  const [rows, setRows] = useState<Row[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    getJSON<any>("/api/results/final")
      .then((data) => setRows(Array.isArray(data) ? data : [data]))
      .catch((e) => setError(e.message));
  }, []);

  const best = rows.reduce<Row | null>((acc, r) => {
    if (!acc) return r;
    return (r.success_rate ?? 0) > (acc.success_rate ?? 0) ? r : acc;
  }, null);

  return (
    <div>
      <h1>Project Overview</h1>
      <p className="subtitle">
        Robot Navigation Simulation Platform using Reinforcement Learning and
        Kolmogorov–Arnold Networks (KANs)
      </p>

      {error && <div className="card">Error: {error}</div>}

      {best && (
        <div className="grid cols-3">
          <div className="stat">
            <div className="label">Best Model</div>
            <div className="value">{best.model_type?.toUpperCase()}</div>
            <div className="sub">highest success rate</div>
          </div>
          <div className="stat">
            <div className="label">Success Rate</div>
            <div className="value">
              {((best.success_rate ?? 0) * 100).toFixed(1)}%
            </div>
            <div className="sub">over fixed evaluation episodes</div>
          </div>
          <div className="stat">
            <div className="label">Mean Reward</div>
            <div className="value">{best.mean_reward?.toFixed(2)}</div>
            <div className="sub">average episode return</div>
          </div>
        </div>
      )}

      <div className="card" style={{ marginTop: 20 }}>
        <h2>MLP vs KAN — Final Evaluation</h2>
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Type</th>
              <th>Success</th>
              <th>Collision</th>
              <th>Reward</th>
              <th>Steps</th>
              <th>Final Dist.</th>
              <th>Params</th>
              <th>Inference (ms)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                <td>{r.model_name ?? "-"}</td>
                <td>
                  <span className={`badge ${r.model_type}`}>
                    {r.model_type?.toUpperCase()}
                  </span>
                </td>
                <td>{((r.success_rate ?? 0) * 100).toFixed(1)}%</td>
                <td>{((r.collision_rate ?? 0) * 100).toFixed(1)}%</td>
                <td>{r.mean_reward?.toFixed(2)}</td>
                <td>{r.mean_steps?.toFixed(1)}</td>
                <td>{r.mean_final_distance?.toFixed(2)}</td>
                <td>{r.parameter_count?.toLocaleString()}</td>
                <td>{r.avg_inference_ms?.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}