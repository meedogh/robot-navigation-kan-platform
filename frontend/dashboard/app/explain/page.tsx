"use client";

import { useEffect, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Legend,
} from "recharts";
import { getJSON } from "../../lib/api";

type Explain = {
  labels: string[];
  importance: number[];
  top_features: { feature: string; importance: number; xs: number[]; ys: number[] }[];
};

export default function Explain() {
  const [data, setData] = useState<Explain | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getJSON<Explain>("/api/explain/kan")
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="card">Error: {error}</div>;
  if (!data) return <div className="card">Loading…</div>;

  const importanceRows = data.labels
    .map((label, i) => ({ label, importance: data.importance[i] }))
    .sort((a, b) => b.importance - a.importance);

  const maxImp = importanceRows[0]?.importance || 1;

  return (
    <div>
      <h1>KAN Explainability</h1>
      <p className="subtitle">
        How each observation feature influences the policy — a key advantage of KANs over MLPs
      </p>

      <div className="card">
        <h2>Feature Importance</h2>
        {importanceRows.map((r) => (
          <div key={r.label} style={{ marginBottom: 10 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13 }}>
              <span>{r.label}</span>
              <span className="muted">{(r.importance * 100).toFixed(1)}%</span>
            </div>
            <div style={{ background: "#1a2233", borderRadius: 6, height: 8, marginTop: 4 }}>
              <div
                style={{
                  width: `${(r.importance / maxImp) * 100}%`,
                  background: "#38d39f",
                  height: 8,
                  borderRadius: 6,
                }}
              />
            </div>
          </div>
        ))}
      </div>

      <div className="card">
        <h2>Learned Influence of Top Features</h2>
        <p className="subtitle">
          Each curve shows how strongly a top input feature drives the KAN output as its value changes.
        </p>
        {data.top_features.map((f) => {
          const chartData = f.xs.map((x, i) => ({ x, y: f.ys[i] }));
          return (
            <div key={f.feature} style={{ marginBottom: 30 }}>
              <div style={{ marginBottom: 8 }}>
                <span className="badge kan">{f.feature}</span>{" "}
                <span className="muted">importance {(f.importance * 100).toFixed(1)}%</span>
              </div>
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={chartData}>
                  <CartesianGrid stroke="#232d42" />
                  <XAxis dataKey="x" stroke="#8b96ad" type="number" domain={["dataMin", "dataMax"]} />
                  <YAxis stroke="#8b96ad" />
                  <Tooltip contentStyle={{ background: "#121826", border: "1px solid #232d42" }} />
                  <Line type="monotone" dataKey="y" stroke="#4f8cff" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          );
        })}
      </div>
    </div>
  );
}