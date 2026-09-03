"""Shared frame builder for live visualization streams.

Frames produced during training / evaluation have exactly the same shape as the
frames emitted by ``backend.live_sim.LiveSimulator``, so the Live page's canvas
renders saved-model runs and active-job runs identically.
"""

from typing import Any, Dict, Optional


def build_frame(
    env,
    model_type: str,
    action: int,
    reward: float,
    episode_reward: float,
    step: int,
    info: dict,
    done: bool,
    phase: str = "training",
    source: Optional[str] = None,
    training_step: Optional[int] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a renderable frame from the current environment and step results.

    Visualization attributes (robot_pos, target_pos, obstacles, world_size) are
    optional: external environments (Unity / Gazebo / custom) that do not expose
    them simply render with neutral fallback values.
    """
    robot_pos = getattr(env, "robot_pos", None)
    target_pos = getattr(env, "target_pos", None)
    obstacles = [
        {
            "x": float(pos[0]),
            "y": float(pos[1]),
            "radius": float(radius),
        }
        for pos, radius in (getattr(env, "obstacles", None) or [])
    ]

    frame: Dict[str, Any] = {
        "model": model_type,
        "model_name": model_name,
        "source": source or phase,
        "phase": phase,
        "world_size": float(getattr(env, "world_size", 20.0) or 20.0),
        "robot_x": float(robot_pos[0]) if robot_pos is not None else 0.0,
        "robot_y": float(robot_pos[1]) if robot_pos is not None else 0.0,
        "robot_angle": float(getattr(env, "robot_angle", 0.0) or 0.0),
        "target_x": float(target_pos[0]) if target_pos is not None else 0.0,
        "target_y": float(target_pos[1]) if target_pos is not None else 0.0,
        "obstacles": obstacles,
        "action": int(action),
        "reward": float(reward),
        "episode_reward": float(episode_reward),
        "step": int(step),
        "reached_target": bool(info.get("reached_target", False)),
        "collision": bool(info.get("collision", False)),
        "done": bool(done),
    }

    if training_step is not None:
        frame["training_step"] = int(training_step)

    return frame