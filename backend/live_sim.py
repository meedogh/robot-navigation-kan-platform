from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from simulation.env_factory import create_env
from rl.config_io import env_config_from_checkpoint_dir
from rl.frames import serialize_obstacles
from rl.model_factory import create_qnetwork_from_arch, load_arch


BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = BASE_DIR / "experiments" / "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _create_model(model_type: str, obs_dim: int, action_dim: int):
    arch = load_arch(CHECKPOINT_DIR, model_type)
    return create_qnetwork_from_arch(model_type, obs_dim, action_dim, arch)


class LiveSimulator:
    """Streams one navigation episode frame by frame.

    Two modes:
      - auto=True (default): the trained model picks the action every step.
      - auto=False (manual play): the caller supplies the action (keyboard /
        on-screen buttons in the dashboard); the model - when a checkpoint
        exists - still computes Q-values so the UI can show what the agent
        *would* do in the same state ("ghost advice").
    """

    def __init__(
        self,
        model_type: str = "kan",
        env_config: Optional[Dict[str, Any]] = None,
        auto: bool = True,
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
        self.auto = auto

        # Human-readable env identity, shown on the Live page so it is always
        # obvious which environment is being simulated.
        if isinstance(env_config, dict):
            src = env_config.get("source", "builtin")
            variant = env_config.get("variant")
            module = env_config.get("module")
            if src == "module" and module:
                self.env_label = f"external: {module}"
            elif variant:
                names = {"v1": "v1 (single obstacle)", "v2": "v2 (random obstacles)", "custom": "custom map (v3)"}
                self.env_label = f"builtin {names.get(variant, variant)}"
            else:
                self.env_label = "builtin"
        else:
            self.env_label = "builtin"

        # The model is optional in manual mode: a user can play the custom map
        # before any training run exists.
        self.model = None
        if auto:
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

    def initial_frame(self) -> Dict[str, Any]:
        """Frame for the current state WITHOUT stepping the environment
        (used as the first manual-play frame so the canvas is not blank)."""
        frame = {
            "model": self.model_type,
            "env_label": self.env_label,
            "world_size": float(getattr(self.env, "world_size", 20.0)),
            "robot_x": float(self.env.robot_pos[0]) if hasattr(self.env, "robot_pos") else 0.0,
            "robot_y": float(self.env.robot_pos[1]) if hasattr(self.env, "robot_pos") else 0.0,
            "robot_angle": float(getattr(self.env, "robot_angle", 0.0)),
            "target_x": float(self.env.target_pos[0]) if hasattr(self.env, "target_pos") else 0.0,
            "target_y": float(self.env.target_pos[1]) if hasattr(self.env, "target_pos") else 0.0,
            "obstacles": serialize_obstacles(getattr(self.env, "obstacles", None)),
            "action": -1,
            "suggested_action": None,
            "q_values": self._q_values(self.obs) if self.model is not None else None,
            "sensors": (
                [float(self.obs[7]), float(self.obs[8]), float(self.obs[9])]
                if len(self.obs) >= 10
                else None
            ),
            "sensor_range": float(getattr(self.env, "sensor_range", 0.0) or 0.0),
            "manual": not self.auto,
            "reward": 0.0,
            "episode_reward": 0.0,
            "step": 0,
            "reached_target": False,
            "collision": False,
            "done": False,
        }
        return frame

    @torch.no_grad()
    def _q_values(self, obs):
        tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        return self.model(tensor).squeeze(0).cpu().tolist()

    def step(self, action: Optional[int] = None):
        q_values = self._q_values(self.obs) if self.model is not None else None
        suggested = int(np.argmax(q_values)) if q_values is not None else None

        if self.auto:
            action = suggested
        else:
            action = int(action) if action is not None else suggested
            if action is None or not (0 <= action < self.action_dim):
                action = self.env.action_space.sample()  # no model: random

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
        obstacles = serialize_obstacles(getattr(self.env, "obstacles", None))

        # Normalized sensor readings live at obs[7:10] in the platform's
        # observation contract.
        sensors = (
            [float(self.obs[7]), float(self.obs[8]), float(self.obs[9])]
            if self.obs is not None and len(self.obs) >= 10
            else None
        )

        frame = {
            "model": self.model_type,
            "env_label": self.env_label,
            "world_size": float(getattr(self.env, "world_size", 20.0)),
            "robot_x": float(robot_pos[0]) if robot_pos is not None else 0.0,
            "robot_y": float(robot_pos[1]) if robot_pos is not None else 0.0,
            "robot_angle": float(getattr(self.env, "robot_angle", 0.0)),
            "target_x": float(target_pos[0]) if target_pos is not None else 0.0,
            "target_y": float(target_pos[1]) if target_pos is not None else 0.0,
            "obstacles": obstacles,
            "action": int(action),
            "suggested_action": suggested,
            "q_values": q_values,
            "sensors": sensors,
            "sensor_range": float(getattr(self.env, "sensor_range", 0.0) or 0.0),
            "manual": not self.auto,
            "reward": float(reward),
            "episode_reward": self.episode_reward,
            "step": self.episode_step,
            "reached_target": bool(info.get("reached_target", False)),
            "collision": bool(info.get("collision", False)),
            "done": done,
        }
        return frame