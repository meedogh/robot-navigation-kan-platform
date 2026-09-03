"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { getJSON, postJSON } from "../../lib/api";

type FieldDef = {
  key: string;
  label: string;
  type?: "int" | "float" | "select" | "text";
  options?: { value: string; label: string }[];
  step?: number;
  min?: number;
  max?: number;
  hint?: string;
  allowNull?: boolean;
};

const FIELD_GROUPS: { title: string; fields: FieldDef[] }[] = [
  {
    title: "Model and Run",
    fields: [
      {
        key: "model_type", label: "Model", type: "select",
        options: [{ value: "mlp", label: "MLP" }, { value: "kan", label: "KAN" }],
      },
      { key: "total_steps", label: "Total steps", type: "int", min: 1, hint: "Total environment steps to train for" },
      { key: "seed", label: "Random seed", type: "int" },
      { key: "eval_every", label: "Eval every (steps)", type: "int", min: 1 },
      { key: "eval_episodes", label: "Eval episodes", type: "int", min: 1 },
      { key: "eval_seed_base", label: "Eval seed base", type: "int", allowNull: true, hint: "Leave blank to auto-derive from the training seed" },
    ],
  },
  {
    title: "Environment",
    fields: [
      { key: "env_world_size", label: "World size", type: "float", min: 2.01, step: 0.5 },
      { key: "env_max_steps", label: "Max steps per episode", type: "int", min: 1 },
      { key: "env_frame_skip", label: "Frame skip", type: "int", min: 1 },
      { key: "env_min_obstacles", label: "Min obstacles", type: "int", min: 0 },
      { key: "env_max_obstacles", label: "Max obstacles", type: "int", min: 0 },
      { key: "env_sensor_range", label: "Sensor range", type: "float", min: 0.1, step: 0.5 },
      { key: "env_robot_radius", label: "Robot radius", type: "float", min: 0.01, step: 0.05 },
      { key: "env_target_radius", label: "Target radius", type: "float", min: 0.05, step: 0.05 },
      { key: "env_max_speed", label: "Max speed", type: "float", min: 0.01, step: 0.05 },
      { key: "env_turn_angle_deg", label: "Turn angle (deg)", type: "float", min: 1, step: 5 },
    ],
  },
  {
    title: "Environment Source",
    fields: [
      {
        key: "env_source", label: "Environment", type: "select",
        options: [
          { value: "builtin", label: "Built-in (Python)" },
          { value: "module", label: "External module (Unity / Gazebo / custom)" },
        ],
        hint: "External environments must follow the Gymnasium API contract (see README)",
      },
      {
        key: "env_variant", label: "Built-in variant", type: "select",
        options: [{ value: "v2", label: "v2 (6 actions)" }, { value: "v1", label: "v1 (4 actions)" }],
      },
      {
        key: "env_module", label: "Module path", type: "text",
        hint: "package.module:ClassName — importable from the backend Python environment, e.g. my_adapters.unity_env:UnityNavEnv",
      },
    ],
  },
  {
    title: "DQN Hyperparameters",
    fields: [
      { key: "learning_rate", label: "Learning rate", type: "float", min: 0.00000001, step: 0.0001 },
      { key: "gamma", label: "Discount factor", type: "float", min: 0, max: 0.9999, step: 0.01 },
      { key: "buffer_size", label: "Replay buffer size", type: "int", min: 1 },
      { key: "batch_size", label: "Batch size", type: "int", min: 1 },
      { key: "epsilon_start", label: "Epsilon start", type: "float", min: 0, max: 1, step: 0.05 },
      { key: "epsilon_end", label: "Epsilon end", type: "float", min: 0, max: 1, step: 0.05 },
      { key: "epsilon_decay_steps", label: "Epsilon decay steps", type: "int", min: 1 },
      { key: "target_update_interval", label: "Target update interval", type: "int", min: 1 },
    ],
  },
  {
    title: "Network Architecture",
    fields: [
      { key: "mlp_hidden_dim", label: "MLP hidden dim", type: "int", min: 4 },
      { key: "kan_hidden_dim", label: "KAN hidden dim", type: "int", min: 4 },
      { key: "kan_grid_size", label: "KAN grid size", type: "int", min: 2 },
      { key: "kan_grid_range", label: "KAN grid range", type: "float", min: 0.5, step: 0.5 },
    ],
  },
];

type Job = {
  job_id: string;
  job_type: string;
  state: string;
  message: string;
  error?: string | null;
  started_at: number;
  finished_at?: number | null;
  progress?: number;
  progress_label?: string;
  live_enabled?: boolean;
};

function downloadJSON(filename: string, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export default function Setup() {
  const [defaults, setDefaults] = useState<Record<string, string | number | null>>({});
  const [presets, setPresets] = useState<Record<string, Record<string, string | number | null>>>({});
  const [form, setForm] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [job, setJob] = useState<Job | null>(null);
  const [liveEnabled, setLiveEnabled] = useState(true);
  const fileRef = useRef<HTMLInputElement>(null);

  async function refreshStatus() {
    try {
      const data: { busy: boolean; job: Job | null } = await getJSON("/api/training/status");
      if (data.job) setJob(data.job);
    } catch {
      // ignore poll errors so the page does not break on a transient backend hiccup
    }
  }

  useEffect(() => {
    getJSON<{ defaults: any; presets: any }>("/api/training/config")
      .then((data) => {
        const d = data.defaults ?? {};
        setDefaults(d);
        setPresets(data.presets ?? {});
        const filled: Record<string, string> = {};
        for (const key of Object.keys(d)) {
          filled[key] = d[key] === null ? "" : String(d[key]);
        }
        setForm(filled);
        setLoading(false);
      })
      .catch((e) => {
        setError(e.message);
        setLoading(false);
      });
    refreshStatus();
    const timer = setInterval(refreshStatus, 3000);
    return () => clearInterval(timer);
  }, []);

  function fieldOf(key: string): FieldDef | undefined {
    for (const group of FIELD_GROUPS) {
      const f = group.fields.find((x) => x.key === key);
      if (f) return f;
    }
    return undefined;
  }

  function toConfig(): Record<string, string | number | null> {
    const cfg: Record<string, string | number | null> = {};
    for (const key of Object.keys(form)) {
      const raw = form[key];
      const f = fieldOf(key);
      if (raw === "" && f?.allowNull) {
        cfg[key] = null;
        continue;
      }
      if (!f) {
        cfg[key] = raw;
        continue;
      }
      if (f.type === "int") cfg[key] = raw === "" ? null : parseInt(raw, 10);
      else if (f.type === "float") cfg[key] = raw === "" ? null : parseFloat(raw);
      else cfg[key] = raw;
    }
    return cfg;
  }

  function applyPreset(name: string) {
    const patch = presets[name] ?? {};
    const base: Record<string, string> = {};
    for (const key of Object.keys(defaults)) {
      base[key] = defaults[key] === null ? "" : String(defaults[key]);
    }
    for (const key of Object.keys(form)) {
      base[key] = form[key];
    }
    for (const [key, value] of Object.entries(patch)) {
      base[key] = value === null ? "" : String(value);
    }
    setForm(base);
  }

  async function loadLastRun() {
    const model = form.model_type || "mlp";
    try {
      const data = await getJSON<Record<string, string | number | null>>(
        `/api/training/last-config/${model}`
      );
      const filled: Record<string, string> = {};
      for (const [key, value] of Object.entries(data)) {
        filled[key] = value === null ? "" : String(value);
      }
      setForm((prev) => ({ ...prev, ...filled }));
      setError("");
    } catch (e: any) {
      if (e?.response?.status === 404) {
        setError(`No previous run config found for ${model.toUpperCase()}.`);
      } else {
        setError(e.message);
      }
    }
  }

  async function start() {
    setError("");
    try {
      const data = await postJSON<Job>("/api/training/start", {
        config: toConfig(),
        live: liveEnabled,
      });
      setJob(data);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? e.message);
    }
  }

  async function stop() {
    try {
      await postJSON("/api/training/stop");
    } catch (e: any) {
      setError(e.message);
    }
  }

  async function exportConfig() {
    setError("");
    setNotice("");
    try {
      const run = await postJSON<any>("/api/config/export", {
        config: toConfig(),
        name: `robotnav-${form.model_type || "mlp"}-run`,
        description: "Exported from the RobotNav dashboard Setup page",
      });
      const filename = `${String(run.name ?? "robotnav-run").replace(
        /[^A-Za-z0-9_.-]+/g,
        "_"
      )}.json`;
      downloadJSON(filename, run);
      setNotice(
        `Run config exported to ${filename}. Import it here later, or run ` +
          `"python -m rl.train_custom_dqn --config ${filename}" anywhere.`
      );
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? e.message);
    }
  }

  async function importConfig(file: File) {
    setError("");
    setNotice("");
    try {
      const parsed = JSON.parse(await file.text());
      const data = await postJSON<{
        ok: boolean;
        name: string;
        config: Record<string, string | number | null>;
        warnings: string[];
      }>("/api/config/import", parsed);

      const filled: Record<string, string> = {};
      for (const [key, value] of Object.entries(data.config ?? {})) {
        filled[key] = value === null || value === undefined ? "" : String(value);
      }
      setForm((prev) => ({ ...prev, ...filled }));

      const prefix = `Imported run config "${data.name ?? file.name}".`;
      setNotice(
        data.warnings?.length ? `${prefix} ${data.warnings.join(" ")}` : prefix
      );
    } catch (e: any) {
      if (e instanceof SyntaxError) {
        setError(`${file.name} is not valid JSON.`);
      } else {
        setError(e?.response?.data?.detail ?? e.message);
      }
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

  const running = job?.state === "running";
return (
    <div>
      <h1>Setup and Train</h1>
      <p className="subtitle">
        Tune the training and evaluation parameters, then start training from the page.
        Training runs on the backend and replaces the saved checkpoint and log for the
        selected model.
      </p>

      {error && <div className="card">Error: {error}</div>}

      {notice && !error && <div className="card">{notice}</div>}

      {job && (
        <div className="card">
          <h2>Job Status</h2>
          <div className="status-line">
            <span className={`pill ${running ? "running" : job.state}`}>
              {running ? "running" : job.state}
            </span>
            <span className="muted">
              {job.job_type} | {job.job_id}
            </span>
            <span>{job.message}</span>
          </div>
          {typeof job.progress === "number" && (
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
          {running && (
            <div className="actions">
              <label className="toggle-row">
                <input
                  type="checkbox"
                  checked={!!job.live_enabled}
                  onChange={(e) => toggleJobLive(e.target.checked)}
                />
                Live view during this job
              </label>
            </div>
          )}
          {running && (
            <div className="actions">
              <button onClick={stop}>Stop Training</button>
              <Link href="/results">
                <button className="secondary">View Live Progress</button>
              </Link>
              {job.live_enabled && (
                <Link href="/live">
                  <button className="secondary">Watch Live</button>
                </Link>
              )}
            </div>
          )}
          {!running && job.state === "completed" && (
            <div className="actions">
              <Link href="/results">
                <button>View Results</button>
              </Link>
              <Link href="/live">
                <button className="secondary">Run Live Simulation</button>
              </Link>
            </div>
          )}
          {job.error && job.state === "failed" && (
            <pre className="muted" style={{ marginTop: 10, fontSize: 12 }}>
              {job.error}
            </pre>
          )}
        </div>
      )}

      {!loading && (
        <div className="card">
          <div className="actions" style={{ marginTop: 0, marginBottom: 4 }}>
            {Object.keys(presets).map((name) => (
              <button key={name} className="secondary" onClick={() => applyPreset(name)}>
                Preset: {name}
              </button>
            ))}
            <button className="secondary" onClick={loadLastRun}>
              Load Last Run
            </button>
            <button className="secondary" onClick={exportConfig} disabled={running}>
              Export Config
            </button>
            <button
              className="secondary"
              onClick={() => fileRef.current?.click()}
              disabled={running}
            >
              Import Config
            </button>
            <input
              ref={fileRef}
              type="file"
              accept="application/json,.json"
              style={{ display: "none" }}
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) importConfig(file);
                e.target.value = "";
              }}
            />
          </div>

          {FIELD_GROUPS.map((group) => (
            <div key={group.title} style={{ marginBottom: 20 }}>
              <h2 className="section-title">{group.title}</h2>
              <div className="form-grid">
                {group.fields.map((f) => {
                  const value = form[f.key] ?? "";
                  return (
                    <div key={f.key} className="field">
                      <label htmlFor={`field-${f.key}`}>{f.label}</label>
                      {f.type === "select" ? (
                        <select
                          id={`field-${f.key}`}
                          value={value}
                          onChange={(e) => setForm((p) => ({ ...p, [f.key]: e.target.value }))}
                        >
                          {f.options?.map((o) => (
                            <option key={o.value} value={o.value}>
                              {o.label}
                            </option>
                          ))}
                        </select>
                      ) : f.type === "text" ? (
                        <input
                          id={`field-${f.key}`}
                          type="text"
                          value={value}
                          placeholder="package.module:ClassName"
                          onChange={(e) => setForm((p) => ({ ...p, [f.key]: e.target.value }))}
                        />
                      ) : (
                        <input
                          id={`field-${f.key}`}
                          type="number"
                          step={f.step}
                          min={f.min}
                          max={f.max}
                          value={value}
                          onChange={(e) => setForm((p) => ({ ...p, [f.key]: e.target.value }))}
                        />
                      )}
                      {f.hint && <div className="hint">{f.hint}</div>}
                    </div>
                  );
                })}
              </div>
            </div>
          ))}

          <div className="actions">
            <button onClick={start} disabled={running}>
              {running ? "Training in Progress..." : "Start Training"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}