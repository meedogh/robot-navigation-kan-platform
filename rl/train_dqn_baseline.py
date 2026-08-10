from stable_baselines3 import DQN
from simulation.envs.robot_navigation_env import RobotNavigationEnv


def main():
    env = RobotNavigationEnv()

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.0005,
        buffer_size=50_000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1
    )

    model.learn(total_timesteps=100_000)

    model.save("experiments/checkpoints/dqn_mlp_baseline")

    print("Training finished and model saved.")


if __name__ == "__main__":
    main()