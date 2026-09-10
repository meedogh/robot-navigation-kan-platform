"""Shared frame builder for live visualization streams.

Frames produced during training / evaluation have exactly the same shape as the
frames emitted by ``backend.live_sim.LiveSimulator``, so the Live page's canvas
renders saved-model runs and active-job runs identically.
"""

import math
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Obstacle serialization (shape-aware)
# ---------------------------------------------------------------------------
def serialize_obstacles(raw) -> list:
    """Convert an environment's ``obstacles`` attribute to JSON-safe dicts.

    Supports the shape-dict format used by the builtin v2 environment
    (``{"shape": "rect", "pos": ..., "width": ..., ...}`` / circle dicts) and
    the legacy ``(pos, radius)`` tuple format used by older environments and
    external adapters.  Every entry carries ``shape``, ``x``, ``y`` and a
    ``radius`` (bounding circle) so renderers that only know circles keep
    working; rectangles additionally carry ``width`` / ``height`` / ``angle``.
    """
    obstacles = []

    for obstacle in raw or []:
        if isinstance(obstacle, dict):
            pos = obstacle.get("pos", (0.0, 0.0))
            entry = {
                "shape": obstacle.get("shape", "circle"),
                "x": float(pos[0]),
                "y": float(pos[1]),
            }
            if entry["shape"] == "rect":
                entry["width"] = float(obstacle.get("width", 1.0))
                entry["height"] = float(obstacle.get("height", 1.0))
                entry["angle"] = float(obstacle.get("angle", 0.0))
                # Bounding circle keeps legacy renderers sane
                entry["radius"] = 0.5 * math.hypot(
                    entry["width"], entry["height"]
                )
            else:
                entry["radius"] = float(obstacle.get("radius", 0.5))
        else:
            pos, radius = obstacle
            obstacles.append({
                "shape": "circle",
                "x": float(pos[0]),
                "y": float(pos[1]),
                "radius": float(radius),
            })
            continue

        obstacles.append(entry)

    return obstacles


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
    env_label: Optional[str] = None,
    q_values: Optional[list] = None,
    sensors: Optional[list] = None,
) -> Dict[str, Any]:
    """Build a renderable frame from the current environment and step results.

    Visualization attributes (robot_pos, target_pos, obstacles, world_size) are
    optional: external environments (Unity / Gazebo / custom) that do not expose
    them simply render with neutral fallback values.
    """
    robot_pos = getattr(env, "robot_pos", None)
    target_pos = getattr(env, "target_pos", None)
    obstacles = serialize_obstacles(getattr(env, "obstacles", None))

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

    # Explainability extras (all optional - external envs / legacy callers
    # simply omit them and the Live page hides the corresponding panels).
    if env_label:
        frame["env_label"] = str(env_label)
    if q_values is not None:
        frame["q_values"] = [float(q) for q in q_values]
    if sensors is not None:
        frame["sensors"] = [float(s) for s in sensors]
        frame["sensor_range"] = float(getattr(env, "sensor_range", 0.0) or 0.0)

    return frame