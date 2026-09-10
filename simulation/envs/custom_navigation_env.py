"""Custom (v3) navigation environment: user-defined fixed obstacle layout.

Unlike v2, which *samples* obstacles randomly every episode (domain
randomization), v3 lets a run config pin an explicit obstacle layout — the
output of the dashboard's **Custom Environment** editor (Setup page).  This is
the "design your own map" path:

    env_source:  builtin
    env_variant: custom
    env_layout:  '[{"shape": "rect", "x": 0, "y": -3, "width": 8,
                    "height": 0.8, "angle": 0.0}, ...]'

Layout format (JSON list, coordinates in world units, origin at center):

    circle: {"shape": "circle", "x": ..., "y": ..., "radius": ...}
    rect:   {"shape": "rect", "x": ..., "y": ..., "width": ...,
             "height": ..., "angle": radians}

Robot / target spawn points are still randomized per episode (kept clear of
the layout), so the task is not degenerate, but the map itself is fixed -
making it ideal for controlled comparisons (same map for MLP and KAN) and for
demonstrating a hand-designed level in the live view.

All geometry (collision, raycast sensors, BFS solvability check) is inherited
from the shape-aware v2 implementation.
"""

import json
import math

import numpy as np

from simulation.envs.robot_navigation_env_v2 import RobotNavigationEnv


class CustomNavigationEnv(RobotNavigationEnv):
    """v2 navigation env with a fixed, user-defined obstacle layout."""

    def __init__(
        self,
        world_size: float = 20.0,
        max_steps: int = 300,
        frame_skip: int = 3,
        sensor_range: float = 12.0,
        robot_radius: float = 0.35,
        target_radius: float = 0.8,
        max_speed: float = 0.35,
        turn_angle_deg: float = 30.0,
        layout=None,
        **_ignored,  # min/max_obstacles, rect ratio etc. are irrelevant here
    ):
        super().__init__(
            world_size=world_size,
            max_steps=max_steps,
            frame_skip=frame_skip,
            min_obstacles=0,
            max_obstacles=0,
            sensor_range=sensor_range,
            robot_radius=robot_radius,
            target_radius=target_radius,
            max_speed=max_speed,
            turn_angle_deg=turn_angle_deg,
            rect_obstacle_ratio=0.0,
            rect_rotation=False,
        )

        self.layout = self._parse_layout(layout)

    # ------------------------------------------------------------------
    # Layout handling
    # ------------------------------------------------------------------
    def _parse_layout(self, layout):
        """Accept a JSON string, a list of dicts, or None (-> empty map)."""
        if layout is None or layout == "":
            return []

        if isinstance(layout, str):
            try:
                layout = json.loads(layout)
            except json.JSONDecodeError as exc:
                raise ValueError(f"env_layout is not valid JSON: {exc}") from exc

        if not isinstance(layout, list):
            raise ValueError("env_layout must be a JSON list of obstacle objects")

        parsed = []
        half = self.half
        for i, item in enumerate(layout):
            if not isinstance(item, dict):
                raise ValueError(f"env_layout[{i}] must be an object, got {item!r}")

            shape = item.get("shape", "circle")
            x = float(item.get("x", 0.0))
            y = float(item.get("y", 0.0))

            if not (-half <= x <= half and -half <= y <= half):
                raise ValueError(
                    f"env_layout[{i}] center ({x}, {y}) is outside the "
                    f"[-{half}, {half}] world"
                )

            if shape == "circle":
                radius = float(item.get("radius", 1.0))
                if radius <= 0:
                    raise ValueError(f"env_layout[{i}] radius must be > 0")
                parsed.append({
                    "shape": "circle",
                    "pos": np.array([x, y], dtype=np.float32),
                    "radius": radius,
                })
            elif shape == "rect":
                width = float(item.get("width", 1.0))
                height = float(item.get("height", 1.0))
                angle = float(item.get("angle", 0.0))
                if width <= 0 or height <= 0:
                    raise ValueError(f"env_layout[{i}] width/height must be > 0")
                parsed.append({
                    "shape": "rect",
                    "pos": np.array([x, y], dtype=np.float32),
                    "width": width,
                    "height": height,
                    "angle": angle,
                })
            else:
                raise ValueError(
                    f"env_layout[{i}] shape must be 'circle' or 'rect', got {shape!r}"
                )

        return parsed

    # ------------------------------------------------------------------
    # Reset: fixed map, randomized spawn points
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        if not self.layout:
            # Empty layout -> degenerate; fall back to v2 random sampling so
            # the env stays usable.
            return super().reset(seed=seed, options=options)

        super().reset(seed=seed, options=options)  # reseeds np_random, resets counters

        margin = 1.0
        min_target_distance = max(4.0, self.world_size * 0.25)

        for _ in range(200):
            robot = self.np_random.uniform(
                low=-self.half + margin,
                high=self.half - margin,
                size=(2,),
            ).astype(np.float32)
            target = self.np_random.uniform(
                low=-self.half + margin,
                high=self.half - margin,
                size=(2,),
            ).astype(np.float32)

            if self._distance(robot, target) < min_target_distance:
                continue
            if any(self._robot_hits_obstacle(robot, o) for o in self.layout):
                continue
            if any(self._robot_hits_obstacle(target, o) for o in self.layout):
                continue

            self.robot_pos = robot
            self.target_pos = target
            self.robot_angle = float(self.np_random.uniform(-np.pi, np.pi))
            self.obstacles = list(self.layout)

            if not self._path_exists(robot, target, self.obstacles):
                # Designed map may legitimately block some spawn pairs; keep
                # sampling for a reachable pair.
                continue

            return self._get_obs(), self._get_info()

        # Fallback: use a collision-free pair even if the (inflated) BFS grid
        # stays pessimistic about reachability.
        self.obstacles = list(self.layout)
        print(
            "[CustomNavigationEnv] warning: no BFS-reachable spawn pair found "
            "in 200 tries - using a collision-free pair anyway."
        )
        return self._get_obs(), self._get_info()

    def _get_info(self):
        info = super()._get_info()
        info["num_obstacles"] = len(self.obstacles)
        return info


if __name__ == "__main__":
    layout = json.dumps([
        {"shape": "rect", "x": 0.0, "y": -3.0, "width": 8.0, "height": 0.8, "angle": 0.0},
        {"shape": "circle", "x": -5.0, "y": 4.0, "radius": 1.2},
        {"shape": "rect", "x": 5.0, "y": 4.0, "width": 0.8, "height": 5.0, "angle": math.pi / 6},
    ])

    env = CustomNavigationEnv(layout=layout)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)

    for _ in range(300):
        obs, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            obs, info = env.reset()

    print("CustomNavigationEnv smoke test OK")
    print(f"  layout obstacles: {len(env.obstacles)}")
    print(f"  obs shape: {env.observation_space.shape}, actions: {env.action_space.n}")


