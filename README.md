# Robot Navigation RL + KAN Platform

A graduation project that trains Deep Q-Network (DQN) agents to navigate a 2D robot to a target while avoiding obstacles, and compares two policy-network architectures:

- **MLP** — a standard multi-layer perceptron Q-network
- **KAN** — a [Kolmogorov–Arnold Network](https://arxiv.org/abs/2404.19756) Q-network (learned univariate basis functions instead of fixed weights)

A **FastAPI** backend serves training results, a KAN explainability endpoint, and a live WebSocket simulation; a **Next.js** dashboard visualizes everything (training curves, final evaluation, live agent, feature importance).

## Project structure

```
├── simulation/                  # Gymnasium environment
│   ├── envs/robot_navigation_env.py       # env used for training/eval (4 actions, 10-D obs)
│   ├── envs/robot_navigation_env_v2.py    # experimental larger env (domain randomization)
│   ├── env_factory.py                     # ★ builds envs from run configs (builtin registry
│   │                                      #   + external Unity/Gazebo/custom module loading)
│   └── test_random_agent.py               # quick smoke test with a random agent
├── rl/                          # RL code
│   ├── dqn/                     # custom DQN agent + replay buffer
│   ├── policies/                # MLP & KAN Q-networks (+ KAN layer implementation)
│   ├── train_custom_dqn.py      # ★ main training entry point (MLP or KAN) - config-driven, cancellable
│   ├── config_io.py             # ★ portable run-config export / import (CLI + schema)
│   ├── train_dqn_baseline.py    # stable-baselines3 DQN baseline
│   ├── evaluate_agent.py        # evaluate the SB3 baseline
│   ├── evaluate_saved_models.py # evaluate all saved custom checkpoints
│   ├── compare_agents.py        # plot MLP vs KAN comparison
│   └── model_factory.py         # central Q-network factory (keeps architectures consistent)
├── backend/                     # FastAPI app (results API, KAN explainer, live WS sim)
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

Optional extras — SB3 baseline and analysis:

```powershell
python -m rl.train_dqn_baseline       # trains experiments/checkpoints/dqn_mlp_baseline (SB3)
python -m rl.evaluate_agent           # evaluates the SB3 baseline
python -m rl.evaluate_saved_models    # evaluates every saved custom checkpoint (100 episodes)
python -m rl.compare_agents           # renders experiments/results/mlp_vs_kan_comparison.png
```

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

### Unity ML-Agents sketch

Create the scene with an Agent whose *observations* match the spec above and a
Discrete Action Space of 6. Then wrap the ML-Agents Python API in a Gym env:

```python
# my_adapters/unity_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

class UnityNavEnv(gym.Env):
    """Gym adapter for a RobotNav-contract Unity scene (see README 'The contract')."""

    def __init__(self, scene_path=None, world_size=20.0, max_steps=300, **ignored):
        channel = EngineConfigurationChannel()
        self._env = UnityEnvironment(file_name=scene_path, side_channels=[channel])
        channel.set_configuration_parameters(time_scale=100.0)  # train fast
        self._env.reset()
        self._behavior = next(iter(self._env.behavior_specs))
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(10,), dtype=np.float32)
        self.max_steps = max_steps
        self._step_count = 0

    def reset(self, seed=None, options=None):
        self._env.reset()
        self._step_count = 0
        return self._observation(), {}

    def step(self, action):
        self._set_action(int(action))
        self._env.step()
        self._step_count += 1
        obs, reward, done = self._collect()   # map Unity rewards / end-episode flags
        return obs, float(reward), bool(done), self._step_count >= self.max_steps, {}
```

(Full ML-Agents API details — decision requests, terminator flags, action
branching — are documented by Unity; fill in the `_`-prefixed helpers from
your scene's Agent setup. `simulation/envs/robot_navigation_env_v2.py` is the
reference implementation of the contract.)

### Gazebo / ROS sketch

The same pattern applies: a thin `gym.Env` that publishes velocity commands
(mapping the 6 discrete actions to `/cmd_vel` twist messages), reads `/odom` +
LiDAR scans to build the 10-D observation, and detects goal-reached /
collision for `info`. Whether you bridge with `rospy`/`rclpy` directly or use
an existing ROS-Gym bridge, the run config only cares about the class path:

```json
{ "source": "module", "module": "my_adapters.gazebo_env:GazeboNavEnv", "params": {} }
```

### Running it through the dashboard

Import the run config on the **Setup** page (Import Config). The backend
warns if the module can't be imported on that machine, then trains, evaluates
and streams live frames from *your* environment. Simulators like Unity or
Gazebo must live on the machine that runs the backend.

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
| `POST /api/training/start` | start a background training job `{"config": {...}}` |
| `GET /api/training/status` | current job status + live/training busy flag |
| `GET /api/training/progress` | evaluation rows collected so far |
| `POST /api/training/stop` | request a clean early stop of the running job |
| `POST /api/evaluate/start` | evaluate all saved checkpoints `{"episodes": 100}` |
| `GET /api/evaluate/status` | evaluation job status |
| `GET /api/checkpoints` | files in `experiments/checkpoints/` |
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
| `/results` | Results — live training progress, saved checkpoints, final evaluation + run evaluation |

Production build:

```powershell
npm run build
npm start
```

To point the dashboard at a different backend, set `NEXT_PUBLIC_API_URL` in
`frontend/dashboard/.env.local` (e.g. `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000`).
The **Live Simulation** page's WebSocket URL is hardcoded to `ws://127.0.0.1:8000/ws/live`
in `app/live/page.tsx` — edit it there if you change ports.

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
- **Empty training scripts?** `rl/train_kan_custom.py` and `rl/train_mlp_custom.py` are
  empty placeholders — use `rl/train_custom_dqn.py` for both models.
- **Missing `frontend/dashboard/lib/` after cloning** — older `.gitignore` versions had a
  Python `lib/` rule that excluded it; the negation at the end of `.gitignore` now keeps
  it tracked.

