import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch

from simulation.env_factory import create_env
from simulation.envs.robot_navigation_env_v2 import RobotNavigationEnv  # noqa: F401 (kept for backwards compatibility)
from rl.dqn.dqn_agent import DQNAgent
from rl.frames import build_frame
from rl.model_factory import save_arch


DEFAULT_TRAINING_CONFIG: Dict[str, Any] = {
    # run
    "model_type": "mlp",
    "total_steps": 100_000,
    "seed": 42,
    "eval_every": 5_000,
    "eval_episodes": 20,
    "eval_seed_base": None,

    # DQN agent hyperparameters
    "learning_rate": 0.0005,
    "gamma": 0.99,
    "buffer_size": 50_000,
    "batch_size": 64,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 50_000,
    "target_update_interval": 500,

    # network architecture
    "mlp_hidden_dim": 64,
    "kan_hidden_dim": 32,
    "kan_grid_size": 12,
    "kan_grid_range": 2.0,

    # environment
    "env_world_size": 20.0,
    "env_max_steps": 300,
    "env_frame_skip": 3,
    "env_min_obstacles": 3,
    "env_max_obstacles": 6,
    "env_sensor_range": 12.0,
    "env_robot_radius": 0.35,
    "env_target_radius": 0.8,
    "env_max_speed": 0.35,
    "env_turn_angle_deg": 30.0,

    # environment source: the builtin registry, or any external Gymnasium
    # environment class (Unity ML-Agents, Gazebo/ROS bridge, custom module).
    "env_source": "builtin",
    "env_variant": "v2",
    "env_module": None,
}

_INT_KEYS = {
    "total_steps", "seed", "eval_every", "eval_episodes",
    "buffer_size", "batch_size", "epsilon_decay_steps", "target_update_interval",
    "mlp_hidden_dim", "kan_hidden_dim", "kan_grid_size",
    "env_max_steps", "env_frame_skip", "env_min_obstacles", "env_max_obstacles",
}

_FLOAT_KEYS = {
    "learning_rate", "gamma",
    "epsilon_start", "epsilon_end", "kan_grid_range",
    "env_world_size", "env_sensor_range",
    "env_robot_radius", "env_target_radius", "env_max_speed", "env_turn_angle_deg",
}


def validate_training_config(raw: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge user config with defaults and coerce / validate the values."""
    config = {**DEFAULT_TRAINING_CONFIG, **(raw or {})}

    if config["model_type"] not in ("mlp", "kan"):
        raise ValueError(
            f"model_type must be 'mlp' or 'kan', got {config['model_type']!r}"
        )

    for key in _INT_KEYS:
        config[key] = int(config[key])
    for key in _FLOAT_KEYS:
        config[key] = float(config[key])

    if config["eval_seed_base"] is not None:
        config["eval_seed_base"] = int(config["eval_seed_base"])

    if config["total_steps"] < 1:
        raise ValueError("total_steps must be >= 1")
    if config["eval_every"] < 1:
        raise ValueError("eval_every must be >= 1")
    if config["eval_episodes"] < 1:
        raise ValueError("eval_episodes must be >= 1")
    if config["batch_size"] < 1:
        raise ValueError("batch_size must be >= 1")
    if config["buffer_size"] < config["batch_size"]:
        raise ValueError("buffer_size must be >= batch_size")
    if not (0.0 <= config["epsilon_start"] <= 1.0):
        raise ValueError("epsilon_start must be in [0, 1]")
    if not (0.0 <= config["epsilon_end"] <= 1.0):
        raise ValueError("epsilon_end must be in [0, 1]")
    if config["learning_rate"] <= 0.0:
        raise ValueError("learning_rate must be > 0")
    if not (0.0 < config["gamma"] < 1.0):
        raise ValueError("gamma must be in (0, 1)")
    if config["env_max_obstacles"] < config["env_min_obstacles"]:
        raise ValueError("env_max_obstacles must be >= env_min_obstacles")
    if config["env_world_size"] <= 2.0:
        raise ValueError("env_world_size must be > 2.0")

    return config


def evaluate_agent(
    agent,
    env,
    episodes: int = 20,
    seed_base=None,
    frame_callback=None,
    model_type: str = "unknown",
    phase: str = "evaluation"
):
    rewards = []
    successes = []
    collisions = []
    steps_list = []
    final_distances = []

    for ep in range(episodes):
        if seed_base is not None:
            obs, _ = env.reset(seed=seed_base + ep)
        else:
            obs, _ = env.reset()

        done = False
        episode_reward = 0.0
        episode_steps = 0
        info = {}

        while not done:
            action = agent.select_action(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += float(reward)
            episode_steps += 1

            done = terminated or truncated

            if frame_callback is not None:
                frame_callback(
                    build_frame(
                        env=env,
                        model_type=model_type,
                        action=action,
                        reward=float(reward),
                        episode_reward=episode_reward,
                        step=episode_steps,
                        info=info,
                        done=done,
                        phase=phase,
                        source=phase,
                    )
                )

        rewards.append(episode_reward)
        successes.append(bool(info.get("reached_target", False)))
        collisions.append(bool(info.get("collision", False)))
        steps_list.append(episode_steps)
        final_distances.append(float(info.get("distance_to_target", float("nan"))))

    metrics = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "mean_steps": float(np.mean(steps_list)),
        "mean_final_distance": float(np.nanmean(final_distances)),
    }

    return metrics


def train(
    model_type: str = "mlp",
    total_steps: int = 100_000,
    seed: int = 42,
    eval_every: int = 5_000,
    eval_episodes: int = 20,
    eval_seed_base: Optional[int] = None,
    checkpoint_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    cancel_event: Optional[threading.Event] = None,
    config: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    frame_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    status_callback: Optional[Callable[[Dict[str, Any]], None]] = None
):
    """Train a DQN agent.

    All tuning knobs live in `config` (merged over the legacy keyword args).
    Callers may pass an absolute `checkpoint_dir` / `results_dir`, a
    `cancel_event` (threading.Event) to request a clean early stop, a
    `progress_callback` invoked with each evaluation row, and optionally a
    `frame_callback` for live visualization (throttle a ~30 fps there) and a
    `status_callback` invoked once per training step for smooth progress bars.
    """
    overrides = {
        "model_type": model_type,
        "total_steps": total_steps,
        "seed": seed,
        "eval_every": eval_every,
        "eval_episodes": eval_episodes,
        "eval_seed_base": eval_seed_base,
    }
    # eval_seed_base=None means "auto", but keep an explicit user override.
    if eval_seed_base is None:
        if config is not None and config.get("eval_seed_base") is not None:
            overrides["eval_seed_base"] = config["eval_seed_base"]

    config = validate_training_config({**overrides, **(config or {})})

    model_type = config["model_type"]
    total_steps = config["total_steps"]
    seed = config["seed"]
    eval_every = config["eval_every"]
    eval_episodes = config["eval_episodes"]
    eval_seed_base = config["eval_seed_base"] or (seed + 100_000)

    torch.manual_seed(seed)
    np.random.seed(seed)

    stopped = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    env_kwargs = {
        "world_size": config["env_world_size"],
        "max_steps": config["env_max_steps"],
        "frame_skip": config["env_frame_skip"],
        "min_obstacles": config["env_min_obstacles"],
        "max_obstacles": config["env_max_obstacles"],
        "sensor_range": config["env_sensor_range"],
        "robot_radius": config["env_robot_radius"],
        "target_radius": config["env_target_radius"],
        "max_speed": config["env_max_speed"],
        "turn_angle_deg": config["env_turn_angle_deg"],
    }

    # env_source = "builtin" -> RobotNavigationEnv (v2) with env_kwargs;
    # env_source = "module"  -> any external Gymnasium environment class,
    # e.g. a Unity ML-Agents or Gazebo/ROS adapter (see rl/config_io.py).
    if config.get("env_source", "builtin") == "builtin":
        env = RobotNavigationEnv(**env_kwargs)
        eval_env = RobotNavigationEnv(**env_kwargs)
    else:
        env = create_env(config)
        eval_env = create_env(config)

    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("experiments/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(results_dir) if results_dir else Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    best_success = -1.0
    best_reward = -1e9
    best_step = 0

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        network_type=model_type,
        device=device,
        config=config
    )

    obs, _ = env.reset(seed=seed)

    logs = []
    last_frame_time = 0.0
    live_episode_reward = 0.0
    live_episode_steps = 0

    print(f"Training {model_type.upper()} agent on device: {device}")

    for step in range(1, total_steps + 1):
        if cancel_event is not None and cancel_event.is_set():
            print("Training stopped by user request.")
            stopped = True
            break

        action = agent.select_action(obs, training=True)

        next_obs, reward, terminated, truncated, info = env.step(action)

        done_step = bool(terminated or truncated)

        live_episode_reward += float(reward)
        live_episode_steps += 1

        if frame_callback is not None:
            now = time.monotonic()
            if now - last_frame_time >= 1.0 / 30.0:
                last_frame_time = now
                frame_callback(
                    build_frame(
                        env=env,
                        model_type=model_type,
                        action=action,
                        reward=float(reward),
                        episode_reward=live_episode_reward,
                        step=live_episode_steps,
                        info=info,
                        done=done_step,
                        phase="training",
                        source="training",
                        training_step=step,
                    )
                )

        done_for_buffer = bool(terminated)

        agent.buffer.add(
            obs,
            action,
            float(reward),
            next_obs,
            done_for_buffer
        )

        obs = next_obs

        if terminated or truncated:
            obs, _ = env.reset()
            live_episode_reward = 0.0
            live_episode_steps = 0

        agent.update()

        if status_callback is not None:
            status_callback({
                "training_step": step,
                "total_steps": total_steps,
                "phase": "training",
            })

        if step % eval_every == 0:
            metrics = evaluate_agent(
                agent,
                eval_env,
                episodes=eval_episodes,
                seed_base=eval_seed_base,
                frame_callback=frame_callback,
                model_type=model_type,
                phase="evaluation",
            )

            row = {
                "model_type": model_type,
                "training_step": step,
                **metrics
            }

            logs.append(row)

            if progress_callback is not None:
                progress_callback(dict(row))

            print(
                f"[{model_type.upper()}] Step {step} | "
                f"Reward: {metrics['mean_reward']:.2f} | "
                f"Success: {metrics['success_rate'] * 100:.2f}% | "
                f"Collision: {metrics['collision_rate'] * 100:.2f}% | "
                f"Steps: {metrics['mean_steps']:.2f}"
            )

            is_better = (
                metrics["success_rate"] > best_success
                or (
                    metrics["success_rate"] == best_success
                    and metrics["mean_reward"] > best_reward
                )
            )

            if is_better:
                best_success = metrics["success_rate"]
                best_reward = metrics["mean_reward"]
                best_step = step

                best_model_path = checkpoint_dir / f"custom_dqn_{model_type}_best.pt"
                torch.save(agent.policy.state_dict(), best_model_path)

                print(
                    f"New best {model_type.upper()} model | "
                    f"Step: {best_step} | "
                    f"Success: {best_success * 100:.2f}% | "
                    f"Reward: {best_reward:.2f}"
                )

    if not stopped:
        final_metrics = evaluate_agent(
            agent,
            eval_env,
            episodes=eval_episodes,
            seed_base=eval_seed_base,
            frame_callback=frame_callback,
            model_type=model_type,
            phase="evaluation",
        )

        final_row = {
            "model_type": model_type,
            "training_step": total_steps,
            **final_metrics
        }

        logs.append(final_row)

        if progress_callback is not None:
            progress_callback(dict(final_row))

        print("\n===== Final Evaluation =====")
        print(final_row)

    # Persist the exact architecture so evaluation / live sim / explain loaders
    # can reconstruct the network even when custom architecture was used.
    save_arch(checkpoint_dir, model_type, config)

    model_path = checkpoint_dir / f"custom_dqn_{model_type}.pt"
    torch.save(agent.policy.state_dict(), model_path)

    config_path = results_dir / f"custom_dqn_{model_type}_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    if logs:
        df = pd.DataFrame(logs)
        csv_path = results_dir / f"custom_dqn_{model_type}_train_log.csv"
        df.to_csv(csv_path, index=False)
    else:
        csv_path = None

    print(f"Model saved to: {model_path}")
    if csv_path:
        print(f"Training log saved to: {csv_path}")

    return {
        "status": "stopped" if stopped else "completed",
        "model_type": model_type,
        "model_path": str(model_path),
        "log_path": str(csv_path) if csv_path else None,
        "config_path": str(config_path),
        "rows_logged": len(logs),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a DQN agent (MLP or KAN). Supports portable run "
        "config files (see rl/config_io.py)."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a portable run config JSON (exported via "
        "python -m rl.config_io export or the dashboard). CLI flags below "
        "override values from the file when given."
    )

    parser.add_argument(
        "--export-config",
        type=str,
        default=None,
        help="Resolve the effective training config to this JSON file, then "
        "exit without training."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["mlp", "kan"]
    )

    parser.add_argument(
        "--total-steps",
        type=int,
        default=None
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=None
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None
    )

    args = parser.parse_args()

    # 1) Start from the run config file (if any), 2) apply explicitly given
    #    CLI flags on top, 3) let validate_training_config fill the defaults.
    config: Dict[str, Any] = {}
    if args.config is not None:
        from rl.config_io import load_run_config_flat

        config = load_run_config_flat(args.config)
        print(
            f"Loaded run config: {args.config} "
            f"(environment source: {config.get('env_source', 'builtin')!r}"
            + (
                f", module: {config.get('env_module')!r}"
                if config.get("env_source") == "module"
                else ""
            )
            + ")"
        )

    cli_overrides: Dict[str, Any] = {}
    if args.model is not None:
        cli_overrides["model_type"] = args.model
    if args.total_steps is not None:
        cli_overrides["total_steps"] = args.total_steps
    if args.seed is not None:
        cli_overrides["seed"] = args.seed
    if args.eval_every is not None:
        cli_overrides["eval_every"] = args.eval_every
    if args.eval_episodes is not None:
        cli_overrides["eval_episodes"] = args.eval_episodes
    config = {**config, **cli_overrides}

    if args.export_config is not None:
        from rl.config_io import export_run_config

        run = export_run_config(
            args.export_config,
            config=config,
            description="Exported via python -m rl.train_custom_dqn --export-config",
        )
        print(f"Run config written to: {args.export_config}")
        print(f"  environment source: {run['environment']['source']!r}")
        raise SystemExit(0)

    train(
        model_type=config.get("model_type", "mlp"),
        total_steps=config.get("total_steps", 100_000),
        seed=config.get("seed", 42),
        eval_every=config.get("eval_every", 5_000),
        eval_episodes=config.get("eval_episodes", 20),
        eval_seed_base=config.get("eval_seed_base"),
        config=config,
    )