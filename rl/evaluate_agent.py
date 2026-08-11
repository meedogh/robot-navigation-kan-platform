import numpy as np
import pandas as pd
from pathlib import Path

from stable_baselines3 import DQN

from simulation.envs.robot_navigation_env import RobotNavigationEnv


MODEL_PATH = "experiments/checkpoints/dqn_mlp_baseline"
RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(model_path, episodes=20):
    env = RobotNavigationEnv()
    model = DQN.load(model_path)

    records = []

    for episode in range(episodes):
        obs, info = env.reset()

        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))

            total_reward += float(reward)
            steps += 1

            done = terminated or truncated

        success = bool(info.get("reached_target", False))
        collision = bool(info.get("collision", False))
        distance_to_target = float(info.get("distance_to_target", np.nan))

        records.append(
            {
                "episode": episode + 1,
                "reward": total_reward,
                "steps": steps,
                "success": success,
                "collision": collision,
                "final_distance_to_target": distance_to_target,
            }
        )

        print(
            f"Episode {episode + 1} | "
            f"Reward: {total_reward:.2f} | "
            f"Steps: {steps} | "
            f"Success: {success} | "
            f"Collision: {collision}"
        )

    df = pd.DataFrame(records)

    output_path = RESULTS_DIR / "dqn_mlp_baseline_evaluation.csv"
    df.to_csv(output_path, index=False)

    print("\n===== Evaluation Summary =====")
    print(f"Episodes: {episodes}")
    print(f"Mean reward: {df['reward'].mean():.2f}")
    print(f"Std reward: {df['reward'].std():.2f}")
    print(f"Mean steps: {df['steps'].mean():.2f}")
    print(f"Success rate: {df['success'].mean() * 100:.2f}%")
    print(f"Collision rate: {df['collision'].mean() * 100:.2f}%")
    print(f"Results saved to: {output_path}")

    env.close()


if __name__ == "__main__":
    evaluate_model(MODEL_PATH, episodes=20)