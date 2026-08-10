from envs.robot_navigation_env import RobotNavigationEnv


def main():
    env = RobotNavigationEnv()

    total_episodes = 5

    for episode in range(total_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward

            if terminated or truncated:
                done = True

        print(f"Episode: {episode + 1}, Reward: {episode_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()