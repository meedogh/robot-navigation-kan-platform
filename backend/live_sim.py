from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from simulation.env_factory import create_env
from rl.config_io import env_config_from_checkpoint_dir
from rl.model_factory import create_qnetwork_from_arch, load_arch


BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = BASE_DIR / "experiments" / "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _create_model(model_type: str, obs_dim: int, action_dim: int):
    arch = load_arch(CHECKPOINT_DIR, model_type)
    return create_qnetwork_from_arch(model_type, obs_dim, action_dim, arch)


class LiveSimulator:
    """Loads a trained model and streams one navigation episode frame by frame."""

    def __init__(
        self,
        model_type: str = "kan",
        env_config: Optional[Dict[str, Any]] = None,
    ):
        # Without an explicit env config, re-create the environment the
        # checkpoint was trained in (from custom_dqn_{model}_config.json).
        # That may be the builtin env or an external Unity / Gazebo adapter.
        if env_config is None:
            env_config = env_config_from_checkpoint_dir(CHECKPOINT_DIR, model_type)
        self.env_config = env_config
        self.env = create_env(env_config)
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

        # Visualization attributes: external environments may not expose them,
        # in which case the frame falls back to neutral values (the Live page
        # renders whatever is present).
        robot_pos = getattr(self.env, "robot_pos", None)
        target_pos = getattr(self.env, "target_pos", None)
        obstacles = [
            {
                "x": float(pos[0]),
                "y": float(pos[1]),
                "radius": float(radius),
            }
            for pos, radius in (getattr(self.env, "obstacles", None) or [])
        ]

        frame = {
            "model": self.model_type,
            "world_size": float(getattr(self.env, "world_size", 20.0)),
            "robot_x": float(robot_pos[0]) if robot_pos is not None else 0.0,
            "robot_y": float(robot_pos[1]) if robot_pos is not None else 0.0,
            "robot_angle": float(getattr(self.env, "robot_angle", 0.0)),
            "target_x": float(target_pos[0]) if target_pos is not None else 0.0,
            "target_y": float(target_pos[1]) if target_pos is not None else 0.0,
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