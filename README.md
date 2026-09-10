# Robot Navigation RL + KAN Platform

A graduation project that trains Deep Q-Network (DQN) agents to navigate a 2D robot to a target while avoiding obstacles, and compares two policy-network architectures:

- **MLP** — a standard multi-layer perceptron Q-network
- **KAN** — a [Kolmogorov–Arnold Network](https://arxiv.org/abs/2404.19756) Q-network (learned univariate basis functions instead of fixed weights)

A **FastAPI** backend serves training results, a KAN explainability endpoint, and a live WebSocket simulation; a **Next.js** dashboard visualizes everything (training curves, final evaluation, live agent, feature importance).

## Project structure

```
├── simulation/                  # Gymnasium environment
│   ├── envs/robot_navigation_env.py       # simple legacy env (4 actions, raycast sensors)
│   ├── envs/robot_navigation_env_v2.py    # ★ main env: domain randomization, circle + rotated-rectangle obstacles
│   ├── env_factory.py                     # ★ builds envs from run configs (builtin registry
│   │                                      #   + external Unity/Gazebo/custom module loading)
│   └── test_random_agent.py               # quick smoke test with a random agent
├── rl/                          # RL code
│   ├── dqn/                     # custom DQN agent + replay buffer
│   ├── policies/                # MLP & KAN Q-networks (+ KAN layer implementation)
│   ├── train_custom_dqn.py      # ★ main training entry point (MLP or KAN) - config-driven, cancellable
│   ├── run_seeds.py             # ★ multi-seed sweep runner (resumable, writes seed_summary.csv)
│   ├── config_io.py             # ★ portable run-config export / import (CLI + schema)
│   ├── train_dqn_baseline.py    # stable-baselines3 DQN baseline
│   ├── evaluate_agent.py        # evaluate the SB3 baseline
│   ├── evaluate_saved_models.py # evaluate all saved custom checkpoints
│   ├── compare_agents.py        # seed-aggregated MLP vs KAN comparison + statistics
│   └── model_factory.py         # central Q-network factory (keeps architectures consistent)
├── backend/                     # FastAPI app (results API, KAN explainer, live WS sim)
├── my_adapters/                 # external environment adapters & examples
│   ├── custom_rect_env.py       # ★ example custom env source (fixed rectangular walls)
│   ├── unity_env.py             # Unity bridge Gymnasium adapter
│   └── bridge_server.py         # TCP bridge reference server (protocol v1)
├── frontend/dashboard/          # Next.js dashboard (Overview / Training / Live / Explain)
└── experiments/
    ├── configs/                 # example portable run-config files
    ├── checkpoints/             # trained .pt models (gitignored)
    └── results/                 # training logs & evaluation CSVs (sample results included)
```

## Prerequisites

- **Python 3.10+** (developed on 3.11)
- **Node.js 20.9+** and npm (developed on Node 24)
- A CUDA GPU is *optional* — PyTorch uses it automatically if available, everything also runs on CPU.

## 1. Set up Python

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
```

```bash
python3 -m venv .venv
source .venv/bin/activate           # Linux / macOS
pip install -r requirements.txt
```

Smoke-test the environment (optional):

```powershell
cd simulation
python test_random_agent.py         # runs 5 random episodes
cd ..
```

## 2. Train the agents (optional)

> The repo ships with sample training results in `experiments/results/`, so the
> dashboard works out of the box. Train only if you want to regenerate them.
> Note: `experiments/checkpoints/` is gitignored, so trained models must be
> produced locally for the **Live Simulation** and **Explainability** pages.

**Important:** scripts use package imports and relative output paths, so always
run them **from the repository root** with `python -m`:

```powershell
python -m rl.train_custom_dqn --model mlp --total-steps 100000
python -m rl.train_custom_dqn --model kan --total-steps 100000
```

Useful flags: `--model {mlp,kan}`, `--total-steps`, `--seed`, `--eval-every`, `--eval-episodes`,
plus `--config run.json` (train from a portable run config file, see below) and
`--export-config run.json` (write the resolved config and exit).

Outputs:

| File | Description |
| --- | --- |
| `experiments/checkpoints/custom_dqn_{mlp,kan}.pt` / `..._best.pt` | final / best checkpoints |
| `experiments/results/custom_dqn_{mlp,kan}_train_log.csv` | evaluation log every N steps |
| `experiments/results/custom_dqn_{mlp,kan}_eval_episodes.csv` | raw per-episode evaluation results (used by the statistics) |

Optional extras — SB3 baseline and analysis:

```powershell
python -m rl.train_dqn_baseline       # trains experiments/checkpoints/dqn_mlp_baseline (SB3)
python -m rl.evaluate_agent           # evaluates the SB3 baseline
python -m rl.evaluate_saved_models    # evaluates every saved custom checkpoint (100 episodes)
python -m rl.run_seeds                # multi-seed sweep (see next section)
python -m rl.compare_agents           # seed-aggregated curves + statistics
```

Useful training flags: `--loss-type {huber,smooth_l1,mse}` and `--huber-delta`
(default: `huber`, robust to the ±100 terminal rewards).

### Multi-seed experiments (thesis-grade statistics)

A single RL run is noisy: the same configuration can produce very different
policies depending on the random seed. For the MLP vs KAN claim, sweep several
seeds and compare distributions instead of single numbers:

```powershell
python -m rl.run_seeds                # 5 default seeds x {mlp, kan}, resumable
python -m rl.compare_agents           # aggregated curves + statistical tests
```

- `rl.run_seeds` stores each run under `experiments/checkpoints/seed_runs/seed_<s>/`
  and `experiments/results/seed_runs/seed_<s>/`, appends to
  `experiments/results/seed_runs/seed_summary.csv`, and **skips** (model, seed)
  pairs whose final checkpoint already exists — interrupt and re-run freely.
  Useful flags: `--models kan --seeds 42 123 2024 --total-steps 300000
  --eval-episodes 30`, plus `--config <flat/run-config JSON>` for environment
  and hyperparameter overrides.
- `rl.compare_agents` plots reward / success curves with ± std bands across
  seeds and writes `experiments/results/seed_comparison_stats.csv`: pooled
  success rate with a bootstrap 95% CI, per-seed std, collision rate, and a
  two-sided permutation test on the pooled final-evaluation episode returns
  (non-parametric, no scipy needed — appropriate for the bimodal returns).

### Environment variations and custom sources

**Obstacle shape variations (builtin v2).** Obstacles are sampled as circles
*or* rectangles; two flat-config keys control the mix (also on the Setup page):

| Key | Meaning |
| --- | --- |
| `env_rect_obstacle_ratio` | probability that an obstacle is a rectangle (0 = all circles, 1 = all rectangles, default 0.5) |
| `env_rect_rotation` | `1` = rectangles spawn at random angles, `0` = axis-aligned (default 1) |

Collision, raycast sensors and the BFS solvability check are all shape-aware,
and the live dashboard renders rotated rectangles. Example config file:

```json
{ "env_rect_obstacle_ratio": 1.0, "env_rect_rotation": 1 }
```

**Custom environment sources.** Set *Environment Source → External module* on
the Setup page (or `"env_source": "module"` in a run config) and give the
module path as `package.module:ClassName`. The class must follow the Gymnasium
contract (6 actions, 10-D observation layout, `obstacles` attribute) documented
in `simulation/env_factory.py::builtin_env_spec`. The repo ships a working
example with fixed rectangular walls:

```text
my_adapters.custom_rect_env:RectNavEnv
```

Before committing to a long run, click **Test Environment** on the Setup page —
it calls `POST /api/env/test`, which instantiates the env, resets it, takes
three random steps and reports the observed spec (obs dim, action count,
obstacle shapes) or a readable error.

## Portable run configs (export / import)

A **run config** is a single JSON file that fully describes a run: the training /
DQN / network hyperparameters **and** the environment (world size, physics,
obstacle counts — or a pointer to an external simulator environment). Share it,
commit it, or open it on another machine and reproduce the exact run.

Example files live in `experiments/configs/`:

- `default_builtin.json` — built-in environment with platform defaults
- `external_env_template.json` — template pointing at an external environment

### File format

```json
{
  "format": "robotnav-run-config",
  "schema_version": 1,
  "name": "my-run",
  "description": "…",
  "training":   { "model_type": "kan", "learning_rate": 0.0005, "…": "…" },
  "environment": {
    "source": "builtin",
    "variant": "v2",
    "params": { "world_size": 20.0, "robot_radius": 0.35, "…": "…" },
    "spec":   { "observation": { "shape": [10], "features": ["…"] }, "action": { "n": 6 } }
  }
}
```

- `"source": "builtin"` — a registered Python environment (`variant: "v1"|"v2"`).
- `"source": "module"` — any external Gymnasium environment class, see the
  next section.
- `environment.spec` documents the observation layout, action semantics and
  physics of the environment — the contract an external implementation must
  satisfy so checkpoints / training / live view stay compatible.

Import validation is strict: unknown or misspelled keys, bad types and
unsupported schema versions are rejected with precise error messages.

### Ways to export

| Where | How |
| --- | --- |
| Dashboard Setup page | **Export Config** button — downloads the current form as a run config |
| Dashboard Setup page | **Load Last Run** → **Export Config** — re-export the previous run |
| CLI (defaults) | `python -m rl.config_io export --out myrun.json --name my-run` |
| CLI (last run of a model) | `python -m rl.config_io export --out myrun.json --from-model kan` |
| CLI (effective config incl. flag overrides) | `python -m rl.train_custom_dqn --model kan --total-steps 50000 --export-config myrun.json` |
| API | `POST /api/config/export` (flat config in → file download) · `GET /api/config/export/{mlp\|kan}` (last run) |

### Ways to import

| Where | How |
| --- | --- |
| Dashboard Setup page | **Import Config** button — validates the file and fills the form (warnings shown for external environments) |
| CLI | `python -m rl.train_custom_dqn --config myrun.json` (CLI flags like `--total-steps` override the file when given) |
| Validate only | `python -m rl.config_io validate myrun.json` — prints the resolved flat config |
| API | `POST /api/config/import` (run config in → validated flat config out) |

Checkpoint compatibility is handled automatically: each training run records
its full config (including the environment) next to the checkpoint, so
evaluation, live simulation and KAN explainability always re-create the exact
environment a model was trained in.

## External simulators (Unity / Gazebo / any Python environment)

Training is **environment-agnostic**: the trainer, evaluation, live view and
explainability work against *any* Gymnasium environment. A world built in
Unity (ML-Agents), Gazebo/ROS, Webots, or plain Python runs here as-is —
training, inference and the dashboard included.

### The contract

Point a run config at your environment class and make sure it satisfies:

1. **Gymnasium API** — `reset(seed=None, options=None) -> (obs, info)` and
   `step(action) -> (obs, reward, terminated, truncated, info)`, with
   `observation_space` / `action_space` attributes.
2. **Matching observation layout** — same feature order and scaling as the
   built-in env (see `environment.spec` in the run config: a 10-D box with
   features `robot_x, robot_y, robot_angle, target_x, target_y,
   distance_to_target, angle_to_target, front_sensor, left_sensor,
   right_sensor`).
3. **Matching action semantics** — the discrete action set from the spec
   (`forward, forward-left, forward-right, turn-left, turn-right, stop`).
   A different action count *is allowed* — networks are sized from
   `env.action_space.n` — but checkpoints trained on the built-in env will
   not transfer to a different action set.
4. **`info` keys** — `reached_target`, `collision`, `distance_to_target`
   (used by the evaluation metrics).
5. **Optional visualization attributes** — expose `world_size`, `robot_pos`,
   `robot_angle`, `target_pos` and `obstacles` (list of `(pos, radius)`) and
   the Live page renders your environment exactly like the built-in one
   (omitting them degrades gracefully).

### Registering your environment

Set `environment.source` to `"module"` with a class path (see
`experiments/configs/external_env_template.json`):

```json
{
  "environment": {
    "source": "module",
    "module": "my_adapters.unity_env:UnityNavEnv",
    "params": { "world_size": 20.0, "max_steps": 300 }
  }
}
```

The class is imported lazily on the machine that runs training. Constructor
parameters the class doesn't accept are filtered out automatically (with a
printed note), and `observation_space` / `action_space` are checked against
`environment.spec` with warnings on mismatch.

```powershell
python -m rl.train_custom_dqn --config unity_run.json
```

Everything downstream — evaluation, live WebSocket simulation, KAN
explainability — uses the same environment recorded in the run.

### Unity / Gazebo bridge (shipped)

The platform includes a dependency-free TCP bridge. The simulator runs a small
server; `my_adapters.unity_env.UnityNavEnv` (a Gymnasium client) connects to
it and behaves exactly like the builtin environment for the rest of the
platform.

Why not the ML-Agents Python API? `mlagents_envs` 1.1.0 pins Python
3.10.1–3.10.12 with `protobuf<3.21` and `numpy<1.24` — incompatible with this
project's stack (Python 3.13, protobuf 7, numpy 2). The socket bridge has zero
Python dependencies and works with **any** runtime that can open a TCP socket:
Unity, Gazebo/ROS, Webots, or a simulator on another machine.

```
RobotNav platform (Python)                        External simulator
-------------------------------------------       -------------------------------
rl/train_custom_dqn.py   <- Gymnasium API <-      Unity scene + unity_bridge/RobotNavBridge.cs
rl/evaluate_saved_models.py   |                    Gazebo/ROS node (same protocol)
dashboard (train / eval / live)  |                or python -m my_adapters.bridge_server
my_adapters/unity_env.py  <== TCP, length-prefixed JSON (protocol v1) ==>
```

**Protocol v1** (full spec: `my_adapters/bridge_protocol.py`): every message
is a 4-byte big-endian length followed by UTF-8 JSON. The simulator sends a
`hello` handshake (obs dim, action count, world size) on connect; the client
sends `reset` / `step` / `close`; the simulator replies with `state` messages
carrying `obs`, `reward`, `terminated`, `truncated`, the evaluation `info`
keys and optional visualization data for the Live page.

### Unity quick start

1. Copy `unity_bridge/RobotNavBridge.cs` into your Unity project
   (`Assets/Scripts/`), add an empty GameObject to a new scene, attach the
   script and press **Play**. Ground, boundary walls, robot, target and the
   randomized obstacles are created automatically; the world constants
   (size, sensor range, speeds, ...) are Inspector fields matching the
   builtin env defaults.
2. The C# script is a 1:1 port of `robot_navigation_env_v2.py` — same
   domain-randomized map generation with a BFS solvability check, same
   kinematics, same reward, same analytic sector sensors. Every connection
   gets its own simulation session (the platform opens one connection per
   environment instance, e.g. train + eval).
3. Copy `experiments/configs/unity_env_template.json`, adjust `host` / `port`
   under `environment.params` if needed, and train:

   ```powershell
   python -m rl.train_custom_dqn --config experiments/configs/unity_env_template.json
   ```

   The Unity scene must be running **before** training starts, on the machine
   that runs the training process.

### Test the bridge without Unity

```powershell
python -m my_adapters.bridge_server --port 5577   # terminal 1: reference simulator
python -m my_adapters.smoke_test                  # terminal 2: end-to-end check
```

`my_adapters.bridge_server` wraps the builtin v2 environment in the same
protocol the Unity script speaks — it is the parity reference for any new
simulator implementation and proves the whole pipeline without Unity.

### Gazebo / ROS

Implement the same protocol on the simulator side: map the six discrete
actions to `/cmd_vel` (or direct joint commands), publish the normalized 10-D
observation from `/odom` + LiDAR scans, and report `reached_target` /
`collision` in `info`. `my_adapters/bridge_protocol.py` is stdlib-only and
self-contained — port it to `rclpy`/`rospy` or any other language 1:1. The
run config only cares about the adapter class path:

```json
{ "source": "module", "module": "my_adapters.unity_env:UnityNavEnv", "params": { "host": "127.0.0.1", "port": 5577 } }
```

### "Train here, run it there" — ONNX model export

```powershell
python -m rl.model_export --model mlp          # or --model kan
```

| Dashboard | API |
| --- | --- |
| Results page → **Export Models (ONNX)** buttons (MLP / KAN) | `GET /api/model/export/{mlp\|kan}` |

The exported graph (input `observations` → output `q_values`, dynamic batch
axis) runs inside Unity via **Microsoft Sentis**, or any ONNX runtime — a
model trained on this platform can then drive a simulation built in the
external tool with no Python involved. The custom KAN layer (piecewise-linear
basis functions + einsum) exports faithfully too: both model types are
verified against onnxruntime after export (max \|ΔQ\| < 1e-4). Files are
written next to the checkpoint by default, or to `--out`.

### Running it through the dashboard

Import the run config on the **Setup** page (Import Config). The backend
warns if the module can't be imported on that machine, then trains, evaluates
and streams live frames from *your* environment. The simulator must live on
the machine that runs the backend; `host` / `port` are part of
`environment.params` in the config file (the Setup form covers the module
path; bridge connection parameters come from the imported file).

## 3. Run the backend (FastAPI)

From the repository root (port **8000** — the dashboard expects this):

```powershell
python -m uvicorn backend.main:app --reload --port 8000
```

Health check: <http://127.0.0.1:8000/> · Interactive docs: <http://127.0.0.1:8000/docs>

| Endpoint | Description |
| --- | --- |
| `GET /api/results/training` | MLP + KAN training logs |
| `GET /api/results/training/mlp` · `/training/kan` | individual training logs |
| `GET /api/results/final` | final saved-model evaluation table |
| `GET /api/comparison/summary` | best-model summary |
| `GET /api/results/comparison-image` | MLP vs KAN comparison plot |
| `GET /api/training/config` | default tuning knobs + preset runs |
| `GET /api/training/last-config/{mlp\|kan}` | config of the most recent run |
| `POST /api/config/export` | flat config in → portable run-config JSON download |
| `POST /api/config/import` | run config JSON in → validated flat config + warnings |
| `GET /api/config/export/{mlp\|kan}` | last run config as a downloadable run-config file |
| `GET /api/model/export/{mlp\|kan}` | best checkpoint as an ONNX download (Unity Sentis-ready) — also the Results page → Export Models (ONNX) buttons |
| `POST /api/training/start` | start a background training job `{"config": {...}}` |
| `GET /api/training/status` | current job status + live/training busy flag |
| `GET /api/training/progress` | evaluation rows collected so far |
| `POST /api/training/stop` | request a clean early stop of the running job |
| `POST /api/evaluate/start` | evaluate all saved checkpoints `{"episodes": 100}` |
| `GET /api/evaluate/status` | evaluation job status |
| `GET /api/checkpoints` | files in `experiments/checkpoints/` |
| `POST /api/env/test` | probe an environment config (builtin/module): spec + 3 random steps — Setup page "Test Environment" |
| `GET /api/explain/kan` | KAN feature importance + learned curves *(requires a KAN checkpoint)* |
| `WS /ws/live` | live episode streaming — client sends `{"model": "kan" \| "mlp"}` *(requires a checkpoint)* |

## 4. Run the dashboard (Next.js)

```powershell
cd frontend\dashboard
npm install
npm run dev
```

Open <http://localhost:3000>. Pages:

| Route | Page |
| --- | --- |
| `/` | Overview — best model stats + final evaluation table |
| `/training` | Reward / success / collision curves for MLP vs KAN |
| `/live` | Real-time agent simulation over WebSocket |
| `/explain` | KAN explainability — feature importance & learned functions |
| `/setup` | Setup and Train — tweak training/evaluation parameters and start a run |
| `/results` | Results — live training progress, saved checkpoints, final evaluation, run evaluation + ONNX model export |

Production build:

```powershell
npm run build
npm start
```

To point the dashboard at a different backend, set `NEXT_PUBLIC_API_URL` in
`frontend/dashboard/.env.local` (e.g. `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000`).
Both the REST base URL and the **Live Simulation** page's WebSocket URL derive
from `NEXT_PUBLIC_API_URL` (the http(s) scheme is converted to ws(s) automatically).

## Quick start (two terminals)

```powershell
# Terminal 1 — backend
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
python -m uvicorn backend.main:app --port 8000

# Terminal 2 — dashboard
cd frontend\dashboard
npm install; npm run dev
```

Then open <http://localhost:3000>.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'simulation'`** — you ran a script directly
  (e.g. `python rl/train_custom_dqn.py`). Run it as a module from the repo root:
  `python -m rl.train_custom_dqn ...`.
- **Live page shows `error: No checkpoint for model 'kan'`** or the Explainability page
  shows `404: No KAN checkpoint found` — no trained model exists in
  `experiments/checkpoints/`. Train one first (see step 2).
- **Backend not reachable from the dashboard** — make sure uvicorn is on port 8000;
  CORS is already open (`allow_origins=["*"]`).
- **Old v1 checkpoints behave oddly or won't load** — the v1 environment's
  front/left/right observations were fixed to true raycast sector distances in
  [0, 1] (matching the v2 implementation), which changes the observation
  semantics. Retrain any v1 models; v2 checkpoints are unaffected.
- **Missing `frontend/dashboard/lib/` after cloning** — older `.gitignore` versions had a
  Python `lib/` rule that excluded it; the negation at the end of `.gitignore` now keeps
  it tracked.

