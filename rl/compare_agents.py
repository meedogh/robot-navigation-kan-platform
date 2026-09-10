"""MLP vs KAN comparison with multi-seed aggregation and statistics.

Data sources (per architecture, first match wins):

1. **Seed runs** - ``<results-root>/seed_<seed>/custom_dqn_<model>_train_log.csv``
   produced by ``rl.run_seeds``.  When present, curves are aggregated as
   mean ± std across seeds and the statistics section runs on the pooled
   final-evaluation episodes from each seed's ``..._eval_episodes.csv``.
2. **Legacy single run** - ``<results-dir>/custom_dqn_<model>_train_log.csv``.
   Plotted as-is; statistics need per-episode data, so they are skipped with
   a hint to run ``python -m rl.run_seeds``.

Outputs:

- ``experiments/results/mlp_vs_kan_comparison.png`` - reward / success curves
  (with ± std bands when multiple seeds exist)
- ``experiments/results/seed_comparison_stats.csv`` - final-metric means,
  bootstrap CIs and a permutation test on the episode returns
- a printed summary table

Usage::

    python -m rl.compare_agents [--results-root DIR] [--results-dir DIR]
                                [--out PNG] [--n-permutations N] [--show]
"""

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib
import numpy as np
import pandas as pd

RESULTS_DIR = Path("experiments/results")
MODELS = ("mlp", "kan")
COLORS = {"mlp": "#4f8cff", "kan": "#ff9f43"}


def load_seed_logs(model_type: str, results_root: Path) -> List[pd.DataFrame]:
    """Training-log frames from every seed directory that has one."""
    logs = []
    if results_root.exists():
        for seed_dir in sorted(results_root.glob("seed_*")):
            path = seed_dir / f"custom_dqn_{model_type}_train_log.csv"
            if path.exists():
                df = pd.read_csv(path)
                if not df.empty:
                    df["seed"] = seed_dir.name
                    logs.append(df)
    return logs


def load_seed_final_episodes(
    model_type: str, results_root: Path
) -> Optional[pd.DataFrame]:
    """Pooled final-evaluation episodes across all seeds (or None)."""
    frames = []
    if results_root.exists():
        for seed_dir in sorted(results_root.glob("seed_*")):
            path = seed_dir / f"custom_dqn_{model_type}_eval_episodes.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if df.empty or "training_step" not in df.columns:
                continue
            final_step = df["training_step"].max()
            df = df[df["training_step"] == final_step].copy()
            df["seed"] = seed_dir.name
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_legacy_log(model_type: str, results_dir: Path) -> Optional[pd.DataFrame]:
    path = results_dir / f"custom_dqn_{model_type}_train_log.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def aggregate_curves(logs: List[pd.DataFrame]) -> pd.DataFrame:
    """Mean ± std of the eval metrics at each training step across seeds."""
    df = pd.concat(logs, ignore_index=True)
    return (
        df.groupby("training_step")
        .agg(
            reward_mean=("mean_reward", "mean"),
            reward_std=("mean_reward", "std"),
            success_mean=("success_rate", "mean"),
            success_std=("success_rate", "std"),
            collision_mean=("collision_rate", "mean"),
            n_runs=("mean_reward", "count"),
        )
        .reset_index()
    )


def permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    n_permutations: int = 10_000,
    seed: int = 0,
) -> tuple:
    """Two-sided permutation test on the difference in means.

    Returns (observed_diff, p_value).  Non-parametric: makes no normality
    assumption, which matters here because episode returns are bimodal
    (+100 success / -100 collision shaped).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    observed = float(a.mean() - b.mean())

    pooled = np.concatenate([a, b])
    n_a = len(a)
    rng = np.random.default_rng(seed)

    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(pooled)
        diff = perm[:n_a].mean() - perm[n_a:].mean()
        if abs(diff) >= abs(observed):
            count += 1

    p_value = (count + 1) / (n_permutations + 1)
    return observed, float(p_value)


def bootstrap_success_ci(
    successes: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple:
    """Percentile-bootstrap CI for a success probability from episode outcomes."""
    arr = np.asarray(successes, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_bootstrap, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ___MAIN_BELOW___


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rl.compare_agents",
        description="Plot the MLP vs KAN comparison and run the statistics.",
    )
    parser.add_argument(
        "--results-root",
        default="experiments/results/seed_runs",
        help="Directory with per-seed result folders (from rl.run_seeds)",
    )
    parser.add_argument(
        "--results-dir",
        default=str(RESULTS_DIR),
        help="Directory with legacy single-run logs + output location",
    )
    parser.add_argument("--out", default=None, help="Output PNG path")
    parser.add_argument("--n-permutations", type=int, default=10_000)
    parser.add_argument(
        "--show", action="store_true", help="Display the plot window (default: headless)"
    )
    args = parser.parse_args()

    if not args.show:
        # Headless-safe backend (servers / CI); must be set before pyplot use.
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_dir = Path(args.results_dir)
    out_path = Path(args.out) if args.out else results_dir / "mlp_vs_kan_comparison.png"

    # ------------------------------------------------------------------
    # Load + aggregate
    # ------------------------------------------------------------------
    curves = {}
    seed_counts = {}
    for model in MODELS:
        seed_logs = load_seed_logs(model, Path(args.results_root))
        if seed_logs:
            logs = seed_logs
        else:
            legacy = load_legacy_log(model, results_dir)
            logs = [legacy] if legacy is not None else []
        seed_counts[model] = len(seed_logs)
        curves[model] = aggregate_curves(logs) if logs else None

    if all(c is None for c in curves.values()):
        print("No training logs found for mlp or kan - nothing to compare.")
        print("Train first: python -m rl.train_custom_dqn ... or python -m rl.run_seeds")
        return 1

    # ------------------------------------------------------------------
    # Plot (mean lines + ± std bands across seeds when available)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for model in MODELS:
        agg = curves[model]
        if agg is None:
            continue
        color = COLORS[model]
        n = seed_counts[model]
        label = f"{model.upper()}" + (f" ({n} seeds)" if n > 1 else "")

        axes[0].plot(agg["training_step"], agg["reward_mean"], label=label, color=color)
        axes[1].plot(
            agg["training_step"], agg["success_mean"] * 100, label=label, color=color
        )

        if n > 1:
            std = agg["reward_std"].fillna(0.0)
            axes[0].fill_between(
                agg["training_step"],
                agg["reward_mean"] - std,
                agg["reward_mean"] + std,
                color=color,
                alpha=0.15,
            )
            std = agg["success_std"].fillna(0.0) * 100
            axes[1].fill_between(
                agg["training_step"],
                agg["success_mean"] * 100 - std,
                agg["success_mean"] * 100 + std,
                color=color,
                alpha=0.15,
            )

    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Mean Reward")
    axes[0].set_title("Reward Comparison")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Success Rate %")
    axes[1].set_title("Success Rate Comparison")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Comparison plot saved to: {out_path}")

    # ___STATS_BELOW___

    # ------------------------------------------------------------------
    # Statistics on the final evaluations
    # ------------------------------------------------------------------
    stats_rows = []
    episodes = {}
    for model in MODELS:
        episodes[model] = load_seed_final_episodes(model, Path(args.results_root))

    if all(ep is None for ep in episodes.values()):
        print(
            "\nNo per-episode evaluation data found - skipping statistics.\n"
            "Run the multi-seed sweep first:  python -m rl.run_seeds"
        )
    else:
        for model in MODELS:
            ep = episodes[model]
            if ep is None or ep.empty:
                print(
                    f"[stats] {model.upper()}: no final-eval episode rows found, "
                    "skipping this model."
                )
                continue

            returns = ep["reward"].to_numpy(dtype=float)
            successes = ep["success"].astype(float).to_numpy()

            per_seed = ep.groupby("seed")["success"].mean()
            ci_lo, ci_hi = bootstrap_success_ci(successes)

            row = {
                "model_type": model,
                "n_seeds": int(ep["seed"].nunique()),
                "n_episodes_final_eval": int(len(ep)),
                "success_rate": float(successes.mean()),
                "success_ci_low": ci_lo,
                "success_ci_high": ci_hi,
                "success_rate_seed_std": float(per_seed.std(ddof=1))
                if len(per_seed) > 1
                else 0.0,
                "mean_reward": float(returns.mean()),
                "std_reward": float(returns.std(ddof=1)),
                "collision_rate": float(ep["collision"].astype(float).mean()),
            }
            stats_rows.append(row)

        if len(stats_rows) == 2:
            mlp_ret = episodes["mlp"]["reward"].to_numpy(dtype=float)
            kan_ret = episodes["kan"]["reward"].to_numpy(dtype=float)
            diff, p_value = permutation_test(
                mlp_ret, kan_ret, n_permutations=args.n_permutations
            )
            stats_rows[0]["reward_diff_vs_kan"] = diff
            stats_rows[0]["permutation_p_value"] = p_value
            stats_rows[1]["reward_diff_vs_kan"] = -diff
            stats_rows[1]["permutation_p_value"] = p_value

        stats_df = pd.DataFrame(stats_rows)
        stats_path = results_dir / "seed_comparison_stats.csv"
        stats_df.to_csv(stats_path, index=False)

        print("\n===== Final-Evaluation Statistics (pooled across seeds) =====")
        print(stats_df.to_string(index=False))
        print(f"\nStatistics saved to: {stats_path}")

    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())