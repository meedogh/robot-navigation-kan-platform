import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from simulation.env_factory import create_env
from rl.config_io import env_config_from_checkpoint_dir
from rl.frames import build_frame
from rl.model_factory import create_qnetwork_from_arch, load_arch


CHECKPOINT_DIR = Path("experiments/checkpoints")
RESULTS_DIR = Path("experiments/results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(model_type: str, obs_dim: int, action_dim: int, arch: Optional[Dict[str, Any]] = None):
    return create_qnetwork_from_arch(model_type, obs_dim, action_dim, arch)


def load_model(model_type: str, model_path: Path, obs_dim: int, action_dim: int, arch: Optional[Dict[str, Any]] = None):
    model = create_model(model_type, obs_dim, action_dim, arch)

    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


def select_action(model, obs):
    obs_tensor = torch.tensor(
        obs,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        start_time = time.perf_counter()
        q_values = model(obs_tensor)
        action = int(q_values.argmax(dim=1).item())
        inference_time = time.perf_counter() - start_time

    return action, inference_time


def evaluate_model(
    model,
    episodes: int = 100,
    seed_base: int = 999_999,
    frame_callback=None,
    on_episode=None,
    model_type: str = "unknown",
    model_name: Optional[str] = None,
    env_config: Optional[Dict[str, Any]] = None,
):
    # env_config comes from the config JSON saved next to the checkpoint, so
    # the model is evaluated in the exact environment it was trained in
    # (builtin or an external Unity / Gazebo / custom module environment).
    env = create_env(env_config)

    rewards = []
    successes = []
    collisions = []
    steps_list = []
    final_distances = []
    inference_times = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_base + ep)

        done = False
        episode_reward = 0.0
        episode_steps = 0
        info = {}

        while not done:
            action, inference_time = select_action(model, obs)

            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += float(reward)
            episode_steps += 1
            inference_times.append(inference_time)

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
                        phase="evaluation",
                        source="evaluation-job",
                        model_name=model_name,
                    )
                )

        rewards.append(episode_reward)
        successes.append(bool(info.get("reached_target", False)))
        collisions.append(bool(info.get("collision", False)))
        steps_list.append(episode_steps)
        final_distances.append(float(info.get("distance_to_target", np.nan)))

        if on_episode is not None:
            on_episode(ep + 1)

    env.close()

    metrics = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "mean_steps": float(np.mean(steps_list)),
        "mean_final_distance": float(np.nanmean(final_distances)),
        "avg_inference_ms": float(np.mean(inference_times) * 1000.0),
        "std_inference_ms": float(np.std(inference_times) * 1000.0),
    }

    return metrics


def main(
    checkpoint_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    episodes: int = 100,
    seed_base: int = 999_999,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    frame_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    episode_callback: Optional[Callable[[int, int], None]] = None
) -> List[Dict[str, Any]]:
    """Evaluate every saved custom checkpoint and write the final evaluation CSV.

    Returns the per-model metric rows. `progress_callback` is invoked once per
    evaluated model, `frame_callback` on every environment step (for live
    visualization), and `episode_callback(done, total)` once per episode (for
    smooth progress bars).
    """
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else CHECKPOINT_DIR
    results_dir = Path(results_dir) if results_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    candidate_models = [
        ("mlp_final", "mlp", "custom_dqn_mlp.pt"),
        ("kan_final", "kan", "custom_dqn_kan.pt"),
        ("mlp_best", "mlp", "custom_dqn_mlp_best.pt"),
        ("kan_best", "kan", "custom_dqn_kan_best.pt"),
    ]

    found = [
        (name, model_type, filename)
        for name, model_type, filename in candidate_models
        if (checkpoint_dir / filename).exists()
    ]

    total_episodes = len(found) * episodes
    completed_episodes = 0

    def _on_episode(_episode_number: int) -> None:
        nonlocal completed_episodes
        completed_episodes += 1
        if episode_callback is not None:
            episode_callback(completed_episodes, total_episodes)

    results = []

    for name, model_type, filename in found:
        model_path = checkpoint_dir / filename

        # Each model may have been trained in a different environment (custom
        # world size or an external simulator) - resolve it per checkpoint.
        env_config = env_config_from_checkpoint_dir(checkpoint_dir, model_type)
        probe_env = create_env(env_config)
        obs_dim = probe_env.observation_space.shape[0]
        action_dim = probe_env.action_space.n
        probe_env.close()

        print(f"Evaluating {name} ...")

        arch = load_arch(checkpoint_dir, model_type)
        model = load_model(
            model_type=model_type,
            model_path=model_path,
            obs_dim=obs_dim,
            action_dim=action_dim,
            arch=arch
        )

        parameter_count = sum(p.numel() for p in model.parameters())

        metrics = evaluate_model(
            model,
            episodes=episodes,
            seed_base=seed_base,
            frame_callback=frame_callback,
            on_episode=_on_episode,
            model_type=model_type,
            model_name=name,
            env_config=env_config,
        )

        row = {
            "model_name": name,
            "model_type": model_type,
            "checkpoint": str(model_path),
            "parameter_count": parameter_count,
            **metrics
        }

        results.append(row)

        if progress_callback is not None:
            progress_callback(dict(row))

        print(f"Finished {name}")
        print(row)
        print("-" * 80)

    if len(results) == 0:
        print("No saved models found.")
        return results

    df = pd.DataFrame(results)

    output_path = results_dir / "final_saved_model_evaluation.csv"
    df.to_csv(output_path, index=False)

    print("\n===== Final Saved Model Evaluation =====")
    print(df.to_string(index=False))

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()