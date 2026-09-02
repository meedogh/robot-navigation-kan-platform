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
│   └── test_random_agent.py               # quick smoke test with a random agent
├── rl/                          # RL code
│   ├── dqn/                     # custom DQN agent + replay buffer
│   ├── policies/                # MLP & KAN Q-networks (+ KAN layer implementation)
│   ├── train_custom_dqn.py      # ★ main training entry point (MLP or KAN)
│   ├── train_dqn_baseline.py    # stable-baselines3 DQN baseline
│   ├── evaluate_agent.py        # evaluate the SB3 baseline
│   ├── evaluate_saved_models.py # evaluate all saved custom checkpoints
│   └── compare_agents.py        # plot MLP vs KAN comparison
├── backend/                     # FastAPI app (results API, KAN explainer, live WS sim)
├── frontend/dashboard/          # Next.js dashboard (Overview / Training / Live / Explain)
└── experiments/
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

Useful flags: `--model {mlp,kan}`, `--total-steps`, `--seed`, `--eval-every`, `--eval-episodes`.

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

