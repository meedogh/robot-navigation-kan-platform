from pathlib import Path

import numpy as np
import torch

# from simulation.envs.robot_navigation_env import RobotNavigationEnv
from simulation.envs.robot_navigation_env_v2 import RobotNavigationEnv
from rl.policies.mlp_network import MLPQNetwork
from rl.policies.kan_network import KANQNetwork


BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = BASE_DIR / "experiments" / "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _create_model(model_type: str, obs_dim: int, action_dim: int):
    if model_type == "mlp":
        return MLPQNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=64)
    if model_type == "kan":
        return KANQNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=32)
    raise ValueError("model_type must be 'mlp' or 'kan'")


class LiveSimulator:
    """Loads a trained model and streams one navigation episode frame by frame."""

    def __init__(self, model_type: str = "kan"):
        self.env = RobotNavigationEnv()
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.model_type = model_type

        self.model = _create_model(model_type, self.obs_dim, self.action_dim)

        best = CHECKPOINT_DIR / f"custom_dqn_{model_type}_best.pt"
        final = CHECKPOINT_DIR / f"custom_dqn_{model_type}.pt"
        path = best if best.exists() else final
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint for model '{model_type}'")

        state = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(state)
        self.model.to(DEVICE)
        self.model.eval()

        self.obs, _ = self.env.reset()
        self.episode_reward = 0.0
        self.episode_step = 0

    def reset(self):
        self.obs, _ = self.env.reset()
        self.episode_reward = 0.0
        self.episode_step = 0

    @torch.no_grad()
    def _act(self, obs):
        tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        q = self.model(tensor)
        return int(q.argmax(dim=1).item())

    def step(self):
        action = self._act(self.obs)
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs = next_obs
        self.episode_reward += float(reward)
        self.episode_step += 1
        done = bool(terminated or truncated)

        # V2 env stores multiple obstacles as a list of (position, radius).
        obstacles = [
            {
                "x": float(pos[0]),
                "y": float(pos[1]),
                "radius": float(radius),
            }
            for pos, radius in self.env.obstacles
        ]

        frame = {
            "model": self.model_type,
            "robot_x": float(self.env.robot_pos[0]),
            "robot_y": float(self.env.robot_pos[1]),
            "robot_angle": float(self.env.robot_angle),
            "target_x": float(self.env.target_pos[0]),
            "target_y": float(self.env.target_pos[1]),
            "obstacles": obstacles,
            "action": action,
            "reward": float(reward),
            "episode_reward": self.episode_reward,
            "step": self.episode_step,
            "reached_target": bool(info.get("reached_target", False)),
            "collision": bool(info.get("collision", False)),
            "done": done,
        }
        return frame