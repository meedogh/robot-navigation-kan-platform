"""Multi-seed experiment runner for the MLP vs KAN comparison.

Single-seed RL results are notoriously noisy: the same configuration can
produce very different final policies depending on the random seed.  This
script trains each architecture across several seeds, storing every run in its
own directory (``<checkpoint-root>/seed_<seed>`` / ``<results-root>/seed_<seed>``)
so the dashboard keeps working on the latest single runs while
``rl.compare_agents`` aggregates across seeds.

Usage (always from the repository root)::

    python -m rl.run_seeds                            # 5 seeds x {mlp, kan}
    python -m rl.run_seeds --seeds 42 123 2024        # specific seeds
    python -m rl.run_seeds --models kan               # one architecture only
    python -m rl.run_seeds --total-steps 100000 --eval-episodes 30
    python -m rl.run_seeds --config experiments/results/custom_dqn_mlp_config.json

Runs are resumable: a (model, seed) pair whose final checkpoint already exists
is skipped unless ``--force`` is passed.  After every run a summary row is
appended to ``<results-root>/seed_summary.csv``, so an interrupted sweep still
yields usable data for the completed runs.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from rl.config_io import load_run_config_flat
from rl.train_custom_dqn import train


DEFAULT_SEEDS = [42, 123, 2024, 777, 31337]

# Run-control keys are set by CLI arguments / the loop, never passed through
# from a base config file.
_RUN_CONTROL_KEYS = {
    "model_type",
    "seed",
    "total_steps",
    "eval_every",
    "eval_episodes",
    "eval_seed_base",
}


def load_base_config(path: Optional[str]) -> Dict[str, Any]:
    """Load an optional flat-config / run-config JSON with env + hyperparameters."""
    if path is None:
        return {}

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    data = json.loads(path.read_text())

    # Portable run-config file -> flatten it.
    if isinstance(data, dict) and data.get("format") == "robotnav-run-config":
        flat = load_run_config_flat(path)
    else:
        if not isinstance(data, dict):
            raise ValueError(f"Config file must contain a JSON object: {path}")
        flat = dict(data)

    for key in _RUN_CONTROL_KEYS:
        flat.pop(key, None)
    return flat


def summarize_run(
    model_type: str,
    seed: int,
    results_dir: Path,
    status: str,
) -> Dict[str, Any]:
    """Build one seed_summary.csv row from a finished run's log files."""
    log_path = results_dir / f"custom_dqn_{model_type}_train_log.csv"
    episodes_path = results_dir / f"custom_dqn_{model_type}_eval_episodes.csv"

    row: Dict[str, Any] = {
        "model_type": model_type,
        "seed": seed,
        "status": status,
        "train_log": str(log_path) if log_path.exists() else None,
        "episodes_log": str(episodes_path) if episodes_path.exists() else None,
    }

    if log_path.exists():
        df = pd.read_csv(log_path)
        if not df.empty:
            final = df.iloc[-1]
            best = df.sort_values(
                ["success_rate", "mean_reward"], ascending=False
            ).iloc[0]

            row.update({
                "final_step": int(final["training_step"]),
                "final_success_rate": float(final["success_rate"]),
                "final_collision_rate": float(final["collision_rate"]),
                "final_mean_reward": float(final["mean_reward"]),
                "final_std_reward": float(final["std_reward"]),
                "final_mean_steps": float(final["mean_steps"]),
                "best_step": int(best["training_step"]),
                "best_success_rate": float(best["success_rate"]),
                "best_mean_reward": float(best["mean_reward"]),
            })

    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rl.run_seeds",
        description=(
            "Train MLP and KAN DQN agents across multiple seeds and write a "
            "summary CSV for the comparison script."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["mlp", "kan"],
        default=["mlp", "kan"],
        help="Architectures to train (default: both)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help=f"Random seeds to sweep (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Optional flat config / portable run-config JSON providing "
            "environment and hyperparameter overrides (run-control keys are "
            "ignored; CLI flags win)."
        ),
    )
    parser.add_argument(
        "--checkpoint-root",
        default="experiments/checkpoints/seed_runs",
        help="Directory receiving per-seed checkpoint folders",
    )
    parser.add_argument(
        "--results-root",
        default="experiments/results/seed_runs",
        help="Directory receiving per-seed result folders + seed_summary.csv",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if the final checkpoint for a (model, seed) exists",
    )
    args = parser.parse_args()

    base_config = load_base_config(args.config)

    checkpoint_root = Path(args.checkpoint_root)
    results_root = Path(args.results_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    summary_path = results_root / "seed_summary.csv"
    summary_rows: List[Dict[str, Any]] = []
    failures = 0

    for model_type in args.models:
        for seed in args.seeds:
            checkpoint_dir = checkpoint_root / f"seed_{seed}"
            results_dir = results_root / f"seed_{seed}"
            final_ckpt = checkpoint_dir / f"custom_dqn_{model_type}.pt"

            print("=" * 80)
            print(f"Run: model={model_type} seed={seed}")

            if final_ckpt.exists() and not args.force:
                print(f"  final checkpoint exists, skipping ({final_ckpt})")
                status = "skipped-existing"
            else:
                run_config = dict(base_config)
                run_config["model_type"] = model_type
                run_config["seed"] = seed
                if args.total_steps is not None:
                    run_config["total_steps"] = args.total_steps
                if args.eval_every is not None:
                    run_config["eval_every"] = args.eval_every
                if args.eval_episodes is not None:
                    run_config["eval_episodes"] = args.eval_episodes

                try:
                    result = train(
                        model_type=model_type,
                        seed=seed,
                        checkpoint_dir=checkpoint_dir,
                        results_dir=results_dir,
                        config=run_config,
                    )
                    status = result["status"]
                except Exception as exc:  # keep the sweep going
                    print(f"  ERROR: {exc}")
                    status = f"failed: {exc}"
                    failures += 1

            summary_rows.append(
                summarize_run(model_type, seed, results_dir, status)
            )
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
            print(f"  status: {status} (summary -> {summary_path})")

    print("=" * 80)
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        print("\n===== Seed Summary =====")
        print(df.to_string(index=False))
    print(f"\nSummary saved to: {summary_path}")
    if failures:
        print(f"WARNING: {failures} run(s) failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())