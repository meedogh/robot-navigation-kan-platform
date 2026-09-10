"""Example external environment: fixed map with rectangular walls.

A minimal, dependency-free example of a **custom environment source** that the
platform can train against through the Environment Source -> "External module"
option (or ``env_source: "module"`` in a run config):

    Module path:  my_adapters.custom_rect_env:RectNavEnv

The world is a fixed 16x16 room with rectangular wall segments (a simple
corridor layout).  It follows the same Gymnasium contract as the builtin v2
environment (see ``simulation/env_factory.py::builtin_env_spec``):

- 6 discrete actions (forward / forward-left / forward-right / turn-left /
  turn-right / stop)
- 10-D normalized observation (robot pose, target pose, distance/angle to
  target, 3 raycast sector sensors)
- ``obstacles`` attribute with shape dicts ({"shape": "rect", ...}) so the
  live dashboard renders the walls as rectangles
- info keys: distance_to_target, step, reached_target, collision

Smoke test it standalone::

    python -m my_adapters.custom_rect_env
"""

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces


WORLD_SIZE = 16.0

# Fixed corridor layout: axis-aligned and rotated wall segments.
# (center_x, center_y, width, height, angle_radians)
WALLS = [
    (-4.0, 2.0, 8.0, 0.8, 0.0),          # horizontal wall, upper-left
    (4.0, -2.0, 0.8, 7.0, 0.0),          # vertical wall, lower-right
    (0.0, -6.0, 6.0, 0.8, math.pi / 4),  # rotated diagonal wall, lower-center
    (-6.0, -5.0, 0.8, 5.0, 0.0),         # vertical wall, lower-left
]


class RectNavEnv(gym.Env):
    """Fixed-map navigation env with rectangular obstacles."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        world_size: float = WORLD_SIZE,
        max_steps: int = 300,
        frame_skip: int = 3,
        sensor_range: float = 10.0,
        robot_radius: float = 0.35,
        target_radius: float = 0.8,
        max_speed: float = 0.35,
        turn_angle_deg: float = 30.0,
        **_ignored,  # extra run-config params are accepted and ignored
    ):
        super().__init__()

        self.world_size = float(world_size)
        self.half = self.world_size / 2.0
        self.max_steps = int(max_steps)
        self.frame_skip = int(frame_skip)
        self.sensor_range = float(sensor_range)
        self.robot_radius = float(robot_radius)
        self.target_radius = float(target_radius)
        self.max_speed = float(max_speed)
        self.turn_angle = math.radians(float(turn_angle_deg))

        # Scale the fixed layout to the requested world size
        scale = self.world_size / WORLD_SIZE
        self.obstacles = [
            {
                "shape": "rect",
                "pos": np.array([cx * scale, cy * scale], dtype=np.float32),
                "width": w * scale,
                "height": h * scale,
                "angle": angle,
            }
            for cx, cy, w, h, angle in WALLS
        ]

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )

        self.robot_pos = np.zeros(2, dtype=np.float32)
        self.robot_angle = 0.0
        self.target_pos = np.zeros(2, dtype=np.float32)
        self.current_step = 0
        self.reached_target = False
        self.collision = False

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.reached_target = False
        self.collision = False

        margin = 1.5
        min_distance = self.world_size * 0.4

        for _ in range(200):
            robot = self.np_random.uniform(
                low=-self.half + margin, high=self.half - margin, size=(2,)
            ).astype(np.float32)
            target = self.np_random.uniform(
                low=-self.half + margin, high=self.half - margin, size=(2,)
            ).astype(np.float32)

            if self._distance(robot, target) < min_distance:
                continue
            if any(self._robot_hits(wall, robot) for wall in self.obstacles):
                continue
            if any(self._robot_hits(wall, target) for wall in self.obstacles):
                continue
            break

        self.robot_pos = robot
        self.target_pos = target
        self.robot_angle = float(self.np_random.uniform(-np.pi, np.pi))

        return self._get_obs(), self._get_info()

    def step(self, action):
        action = int(action)
        self.current_step += 1

        prev_distance = self._distance(self.robot_pos, self.target_pos)

        self.reached_target = False
        self.collision = False

        if action == 1:
            self.robot_angle += self.turn_angle / 2.0
        elif action == 2:
            self.robot_angle -= self.turn_angle / 2.0
        elif action == 3:
            self.robot_angle += self.turn_angle
        elif action == 4:
            self.robot_angle -= self.turn_angle

        self.robot_angle = self._normalize_angle(self.robot_angle)

        if action in (0, 1, 2):
            for _ in range(self.frame_skip):
                direction = np.array(
                    [math.cos(self.robot_angle), math.sin(self.robot_angle)],
                    dtype=np.float32,
                )
                self.robot_pos = self.robot_pos + direction * self.max_speed
                self.robot_pos = np.clip(
                    self.robot_pos, -self.half, self.half
                ).astype(np.float32)
                if self._check_collision():
                    self.collision = True
                    break

        current_distance = self._distance(self.robot_pos, self.target_pos)
        reward = (prev_distance - current_distance) * 2.0 - 0.03
        self.reached_target = bool(current_distance < self.target_radius)

        if self.reached_target:
            reward += 100.0
        if self.collision:
            reward -= 100.0

        terminated = bool(self.reached_target or self.collision)
        truncated = bool(self.current_step >= self.max_steps and not terminated)

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def render(self):
        pass

    # ------------------------------------------------------------------
    # Geometry (shape-aware, same math as the builtin v2 env)
    # ------------------------------------------------------------------
    def _distance(self, a, b):
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

    def _to_local(self, point, wall):
        c = math.cos(-wall["angle"])
        s = math.sin(-wall["angle"])
        d = np.asarray(point, dtype=np.float64) - np.asarray(
            wall["pos"], dtype=np.float64
        )
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])

    def _robot_hits(self, wall, point=None):
        point = self.robot_pos if point is None else point
        local = self._to_local(point, wall)
        half_w = 0.5 * wall["width"] + self.robot_radius
        half_h = 0.5 * wall["height"] + self.robot_radius
        return abs(local[0]) < half_w and abs(local[1]) < half_h

    def _check_collision(self):
        return any(self._robot_hits(wall) for wall in self.obstacles)

    def _sector_distance(self, center_relative_angle, spread):
        distances = []
        for angle in np.linspace(
            center_relative_angle - spread / 2.0,
            center_relative_angle + spread / 2.0,
            5,
        ):
            distances.append(self._ray_distance(float(angle)))
        return float(np.clip(min(distances) / self.sensor_range, 0.0, 1.0))

    def _ray_distance(self, relative_angle):
        angle = self.robot_angle + relative_angle
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)

        half = self.half
        eps = 1e-8

        tx = float("inf")
        ty = float("inf")

        if direction[0] > eps:
            tx = (half - self.robot_pos[0]) / direction[0]
        elif direction[0] < -eps:
            tx = (-half - self.robot_pos[0]) / direction[0]

        if direction[1] > eps:
            ty = (half - self.robot_pos[1]) / direction[1]
        elif direction[1] < -eps:
            ty = (-half - self.robot_pos[1]) / direction[1]

        if tx < 0:
            tx = float("inf")
        if ty < 0:
            ty = float("inf")

        t = min(self.sensor_range, tx, ty)

        for wall in self.obstacles:
            t = min(t, self._rect_ray_distance(direction, wall))

        return float(np.clip(t, 0.0, self.sensor_range))

    def _rect_ray_distance(self, direction, wall):
        local_origin = self._to_local(self.robot_pos, wall)
        c = math.cos(-wall["angle"])
        s = math.sin(-wall["angle"])
        local_dir = np.array(
            [c * direction[0] - s * direction[1], s * direction[0] + c * direction[1]]
        )

        half_w = 0.5 * wall["width"] + self.robot_radius
        half_h = 0.5 * wall["height"] + self.robot_radius

        t_min = -float("inf")
        t_max = float("inf")

        for axis in range(2):
            half = half_w if axis == 0 else half_h
            o = local_origin[axis]
            d = local_dir[axis]

            if abs(d) < 1e-8:
                if abs(o) > half:
                    return self.sensor_range
                continue

            t1 = (-half - o) / d
            t2 = (half - o) / d
            if t1 > t2:
                t1, t2 = t2, t1

            t_min = max(t_min, t1)
            t_max = min(t_max, t2)

            if t_min > t_max:
                return self.sensor_range

        if t_min <= 0.0:
            return 0.0

        return float(min(t_min, self.sensor_range))

    # ------------------------------------------------------------------
    # Observation / info (v2 layout contract)
    # ------------------------------------------------------------------
    def _get_obs(self):
        distance_to_target = self._distance(self.robot_pos, self.target_pos)
        max_distance = math.sqrt(2.0) * self.world_size
        sector_width = 2.0 * math.pi / 3.0

        obs = np.array(
            [
                self.robot_pos[0] / self.half,
                self.robot_pos[1] / self.half,
                self.robot_angle / np.pi,
                self.target_pos[0] / self.half,
                self.target_pos[1] / self.half,
                distance_to_target / max_distance,
                self._angle_to_target() / np.pi,
                self._sector_distance(0.0, sector_width),
                self._sector_distance(2.0 * math.pi / 3.0, sector_width),
                self._sector_distance(-2.0 * math.pi / 3.0, sector_width),
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0)

    def _angle_to_target(self):
        vector = self.target_pos - self.robot_pos
        target_angle = math.atan2(vector[1], vector[0])
        return self._normalize_angle(target_angle - self.robot_angle)

    def _normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

    def _get_info(self):
        return {
            "robot_pos": self.robot_pos.copy(),
            "target_pos": self.target_pos.copy(),
            "num_obstacles": len(self.obstacles),
            "distance_to_target": self._distance(self.robot_pos, self.target_pos),
            "step": self.current_step,
            "reached_target": self.reached_target,
            "collision": self.collision,
        }


if __name__ == "__main__":
    env = RectNavEnv()

    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs), obs
    assert all(o["shape"] == "rect" for o in env.obstacles)

    for _ in range(300):
        obs, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            obs, info = env.reset()

    print("RectNavEnv smoke test OK")
    print(f"  obstacles: {len(env.obstacles)} rectangular walls")
    print(f"  obs shape: {env.observation_space.shape}, actions: {env.action_space.n}")
    print(f"  sample front sensor: {float(env._get_obs()[7]):.3f}")



