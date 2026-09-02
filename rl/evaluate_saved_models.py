import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# from simulation.envs.robot_navigation_env import RobotNavigationEnv
from simulation.envs.robot_navigation_env_v2 import RobotNavigationEnv
from rl.policies.mlp_network import MLPQNetwork
from rl.policies.kan_network import KANQNetwork


CHECKPOINT_DIR = Path("experiments/checkpoints")
RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(model_type: str, obs_dim: int, action_dim: int):
    if model_type == "mlp":
        return MLPQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=64
        )

    if model_type == "kan":
        return KANQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=32,
            grid_size=12,
            grid_range=2.0
        )

    raise ValueError("model_type must be 'mlp' or 'kan'")


def load_model(model_type: str, model_path: Path, obs_dim: int, action_dim: int):
    model = create_model(model_type, obs_dim, action_dim)

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


def evaluate_model(model, episodes: int = 100, seed_base: int = 999_999):
    env = RobotNavigationEnv()

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

        rewards.append(episode_reward)
        successes.append(bool(info.get("reached_target", False)))
        collisions.append(bool(info.get("collision", False)))
        steps_list.append(episode_steps)
        final_distances.append(float(info.get("distance_to_target", np.nan)))

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


def main():
    dummy_env = RobotNavigationEnv()
    obs_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.n
    dummy_env.close()

    candidate_models = [
        ("mlp_final", "mlp", "custom_dqn_mlp.pt"),
        ("kan_final", "kan", "custom_dqn_kan.pt"),
        ("mlp_best", "mlp", "custom_dqn_mlp_best.pt"),
        ("kan_best", "kan", "custom_dqn_kan_best.pt"),
    ]

    results = []

    for name, model_type, filename in candidate_models:
        model_path = CHECKPOINT_DIR / filename

        if not model_path.exists():
            print(f"Skipping {name}: {model_path} not found.")
            continue

        print(f"Evaluating {name} ...")

        model = load_model(
            model_type=model_type,
            model_path=model_path,
            obs_dim=obs_dim,
            action_dim=action_dim
        )

        parameter_count = sum(p.numel() for p in model.parameters())

        metrics = evaluate_model(
            model,
            episodes=100,
            seed_base=999_999
        )

        row = {
            "model_name": name,
            "model_type": model_type,
            "checkpoint": str(model_path),
            "parameter_count": parameter_count,
            **metrics
        }

        results.append(row)

        print(f"Finished {name}")
        print(row)
        print("-" * 80)

    if len(results) == 0:
        print("No saved models found.")
        return

    df = pd.DataFrame(results)

    output_path = RESULTS_DIR / "final_saved_model_evaluation.csv"
    df.to_csv(output_path, index=False)

    print("\n===== Final Saved Model Evaluation =====")
    print(df.to_string(index=False))

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()