"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
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
import { getJSON, postJSON } from "../../lib/api";

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

type Job = {
  job_id: string;
  job_type: string;
  state: string;
  message: string;
  error?: string | null;
  started_at: number;
  finished_at?: number | null;
  log_len?: number;
  payload?: any;
  progress?: number;
  progress_label?: string;
  live_enabled?: boolean;
};

export default function Results() {
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState(false);
  const [rows, setRows] = useState<Row[]>([]);
  const [progressRows, setProgressRows] = useState<any[]>([]);
  const [error, setError] = useState("");
  const [episodes, setEpisodes] = useState("20");
  const [seedBase, setSeedBase] = useState("999999");
  const [checkpoints, setCheckpoints] = useState<any[]>([]);
  const [liveEval, setLiveEval] = useState(true);

  const loadFinal = useCallback(async () => {
    try {
      const data = await getJSON<any>("/api/results/final");
      setRows(Array.isArray(data) ? data : [data]);
      setError("");
    } catch (e: any) {
      if (e?.response?.status === 404) {
        setError("No final evaluation yet. Train and evaluate a model first.");
      } else {
        setError(e.message);
      }
    }
  }, []);

  const loadCheckpoints = useCallback(async () => {
    try {
      const data = await getJSON<any[]>("/api/checkpoints");
      setCheckpoints(data);
    } catch {
      // ignore transient errors
    }
  }, []);

  useEffect(() => {
    async function tick() {
      try {
        const s = await getJSON<{ busy: boolean; job: Job | null }>("/api/training/status");
        setBusy(s.busy);
        setJob(s.job);
      } catch {
        // ignore transient poll errors
      }
    }

    tick();
    const timer = setInterval(tick, 1500);
    loadFinal();
    loadCheckpoints();
    return () => clearInterval(timer);
  }, [loadFinal, loadCheckpoints]);

  // Poll the training evaluation rows while a train job exists.
  useEffect(() => {
    if (!job || job.job_type !== "train") {
      return;
    }

    let cancelled = false;

    async function pollProgress() {
      try {
        const data = await getJSON<{ state: string; rows: any[] }>("/api/training/progress");
        if (!cancelled) setProgressRows(data.rows ?? []);
      } catch {
        // ignore transient poll errors
      }
    }

    pollProgress();
    const timer = setInterval(pollProgress, 1500);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [job?.job_id, job?.job_type]);

  // Refresh derived data when jobs finish.
  useEffect(() => {
    if (job?.job_type === "evaluate" && job.state === "completed") {
      loadFinal();
    }
    if (job?.job_type === "train" && job.state === "completed") {
      loadCheckpoints();
    }
  }, [job?.job_id, job?.job_type, job?.state, loadFinal, loadCheckpoints]);
async function startEvaluation() {
    setError("");
    try {
      await postJSON("/api/evaluate/start", {
        episodes: parseInt(episodes, 10),
        seed_base: parseInt(seedBase, 10),
        live: liveEval,
      });
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? e.message);
    }
  }

  async function stopJob() {
    try {
      await postJSON("/api/training/stop");
    } catch (e: any) {
      setError(e.message);
    }
  }

  async function toggleJobLive(enabled: boolean) {
    // Optimistic update so the checkbox feels instant; the backend mirrors it.
    setJob((prev) => (prev ? { ...prev, live_enabled: enabled } : prev));
    try {
      await postJSON("/api/live/toggle", { enabled });
    } catch (e: any) {
      setError(e.message);
    }
  }

  const chartData = progressRows.map((r) => ({
    step: r.training_step,
    reward: r.mean_reward,
    success: (r.success_rate ?? 0) * 100,
  }));

  const trainRunning = job?.job_type === "train" && job.state === "running";
  const evaluateRunning = job?.job_type === "evaluate" && job.state === "running";

  return (
    <div>
      <h1>Training Results</h1>
      <p className="subtitle">Live training progress, saved checkpoints and final evaluation</p>

      {error && <div className="card">Error: {error}</div>}

      {job && (
        <div className="card">
          <h2>{job.job_type === "train" ? "Training Job" : "Evaluation Job"}</h2>
          <div className="status-line">
            <span className={`pill ${job.state === "running" ? "running" : job.state}`}>
              {job.state}
            </span>
            <span className="muted">
              {job.job_type} | {job.job_id}
            </span>
            <span>{job.message}</span>
          </div>
          {job.state === "running" && typeof job.progress === "number" && (
            <div>
              <div className="progress-track">
                <div
                  className="progress-fill"
                  style={{ width: `${Math.round((job.progress ?? 0) * 100)}%` }}
                />
              </div>
              <div className="progress-label">
                {job.progress_label ?? `${Math.round((job.progress ?? 0) * 100)}%`}
              </div>
            </div>
          )}
          {job.state === "running" && (
            <div className="actions">
              <label className="toggle-row">
                <input
                  type="checkbox"
                  checked={!!job.live_enabled}
                  onChange={(e) => toggleJobLive(e.target.checked)}
                />
                Live view during this job
              </label>
              {job.live_enabled && (
                <Link href="/live">
                  <button className="secondary">Watch Live</button>
                </Link>
              )}
            </div>
          )}
          {trainRunning && (
            <div className="actions">
              <button onClick={stopJob}>Stop Training</button>
            </div>
          )}
          {job.error && job.state === "failed" && (
            <pre className="muted" style={{ marginTop: 10, fontSize: 12 }}>
              {job.error}
            </pre>
          )}
        </div>
      )}

      {trainRunning ? (
        <div className="card">
          <h2>Live Training Progress</h2>
          <p className="subtitle">Evaluation points collected so far during this run</p>
          {chartData.length === 0 ? (
            <p className="muted">Waiting for the first evaluation point...</p>
          ) : (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid stroke="#232d42" />
                <XAxis dataKey="step" stroke="#8b96ad" />
                <YAxis yAxisId="reward" stroke="#8b96ad" />
                <YAxis yAxisId="success" orientation="right" stroke="#8b96ad" domain={[0, 100]} />
                <Tooltip contentStyle={{ background: "#121826", border: "1px solid #232d42" }} />
                <Legend />
                <Line yAxisId="reward" type="monotone" dataKey="reward" stroke="#4f8cff" name="Mean reward" dot={false} />
                <Line yAxisId="success" type="monotone" dataKey="success" stroke="#38d39f" name="Success rate %" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      ) : (
        job?.job_type === "train" && chartData.length > 0 && (
          <div className="card">
            <h2>Progress from last run</h2>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={chartData}>
                <CartesianGrid stroke="#232d42" />
                <XAxis dataKey="step" stroke="#8b96ad" />
                <YAxis yAxisId="reward" stroke="#8b96ad" />
                <YAxis yAxisId="success" orientation="right" stroke="#8b96ad" domain={[0, 100]} />
                <Tooltip contentStyle={{ background: "#121826", border: "1px solid #232d42" }} />
                <Legend />
                <Line yAxisId="reward" type="monotone" dataKey="reward" stroke="#4f8cff" name="Mean reward" dot={false} />
                <Line yAxisId="success" type="monotone" dataKey="success" stroke="#38d39f" name="Success rate %" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )
      )}
<div className="card">
        <h2>Evaluate Saved Models</h2>
        <p className="subtitle">
          Run every saved checkpoint for a fixed number of episodes and refresh the
          final evaluation table below.
        </p>
        <div className="actions" style={{ marginTop: 0 }}>
          <div className="field" style={{ width: 140 }}>
            <label htmlFor="eval-episodes">Episodes</label>
            <input
              id="eval-episodes"
              type="number"
              min={1}
              value={episodes}
              onChange={(e) => setEpisodes(e.target.value)}
            />
          </div>
          <div className="field" style={{ width: 160 }}>
            <label htmlFor="eval-seed">Seed base</label>
            <input
              id="eval-seed"
              type="number"
              value={seedBase}
              onChange={(e) => setSeedBase(e.target.value)}
            />
          </div>
          <label className="toggle-row" style={{ marginTop: 24 }}>
            <input
              type="checkbox"
              checked={liveEval}
              onChange={(e) => setLiveEval(e.target.checked)}
            />
            Live view during evaluation
          </label>
          <button onClick={startEvaluation} disabled={busy} style={{ marginTop: 24 }}>
            {evaluateRunning ? "Evaluating..." : "Evaluate Saved Models"}
          </button>
        </div>
      </div>

      <div className="card">
        <h2>Model Checkpoints</h2>
        {checkpoints.length === 0 ? (
          <p className="muted">No checkpoints found. Train a model from the Setup page first.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>File</th>
                <th>Size</th>
                <th>Modified</th>
              </tr>
            </thead>
            <tbody>
              {checkpoints.map((c) => (
                <tr key={c.name}>
                  <td>{c.name}</td>
                  <td>{(c.size / 1024).toFixed(1)} KB</td>
                  <td>{new Date(c.modified * 1000).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="card">
        <h2>Final Evaluation</h2>
        {rows.length === 0 ? (
          <p className="muted">
            No evaluation data yet. Review the checkpoints card above, then click
            "Evaluate Saved Models".
          </p>
        ) : (
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
        )}
      </div>

      <div className="actions">
        <Link href="/training">
          <button className="secondary">Training Comparison</button>
        </Link>
        <Link href="/live">
          <button className="secondary">Live Simulation</button>
        </Link>
        <Link href="/setup">
          <button className="secondary">New Training Run</button>
        </Link>
      </div>
    </div>
  );
}