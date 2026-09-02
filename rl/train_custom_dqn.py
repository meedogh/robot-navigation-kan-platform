import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# from simulation.envs.robot_navigation_env import RobotNavigationEnv
from simulation.envs.robot_navigation_env_v2 import RobotNavigationEnv
from rl.dqn.dqn_agent import DQNAgent


def evaluate_agent(agent, env, episodes: int = 20, seed_base=None):
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
    eval_episodes: int = 20
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = RobotNavigationEnv()
    eval_env = RobotNavigationEnv()
    
    checkpoint_dir = Path("experiments/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    eval_seed_base = seed + 100_000

    best_success = -1.0
    best_reward = -1e9
    best_step = 0

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        network_type=model_type,
        device=device
    )

    obs, _ = env.reset(seed=seed)

    logs = []

    print(f"Training {model_type.upper()} agent on device: {device}")

    for step in range(1, total_steps + 1):
        action = agent.select_action(obs, training=True)

        next_obs, reward, terminated, truncated, info = env.step(action)

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

        loss = agent.update()

        if step % eval_every == 0:
            metrics = evaluate_agent(
                agent,
                eval_env,
                episodes=eval_episodes,
                seed_base=eval_seed_base
            )

            row = {
                "model_type": model_type,
                "training_step": step,
                **metrics
            }

            logs.append(row)

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

    final_metrics = evaluate_agent(
        agent,
        eval_env,
        episodes=eval_episodes,
        seed_base=eval_seed_base
    )

    final_row = {
        "model_type": model_type,
        "training_step": total_steps,
        **final_metrics
    }

    logs.append(final_row)

    print("\n===== Final Evaluation =====")
    print(final_row)

    checkpoint_dir = Path("experiments/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_path = checkpoint_dir / f"custom_dqn_{model_type}.pt"
    torch.save(agent.policy.state_dict(), model_path)

    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(logs)
    csv_path = results_dir / f"custom_dqn_{model_type}_train_log.csv"
    df.to_csv(csv_path, index=False)

    print(f"Model saved to: {model_path}")
    print(f"Training log saved to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["mlp", "kan"]
    )

    parser.add_argument(
        "--total-steps",
        type=int,
        default=100_000
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=5_000
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20
    )

    args = parser.parse_args()

    train(
        model_type=args.model,
        total_steps=args.total_steps,
        seed=args.seed,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes
    )