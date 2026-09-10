import math
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RobotNavigationEnv(gym.Env):
    """
    Robust robot navigation environment (V2).

    Features:
        - Larger map
        - Multiple random obstacles (Domain Randomization)
        - Random obstacle sizes
        - Raycast-style obstacle sensors (front / left / right)
        - BFS path existence check (guarantees a solvable map)
        - Anti-stuck detection
        - Stop penalty (punishes idling when far from target)
        - Energy usage tracking
        - 6 discrete actions
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        world_size: float = 20.0,
        max_steps: int = 300,
        frame_skip: int = 3,
        min_obstacles: int = 3,
        max_obstacles: int = 6,
        sensor_range: float = 12.0,
        robot_radius: float = 0.35,
        target_radius: float = 0.8,
        max_speed: float = 0.35,
        turn_angle_deg: float = 30.0,
        rect_obstacle_ratio: float = 0.5,
        rect_rotation: bool = True,
    ):
        super().__init__()

        self.world_size = world_size
        self.half = world_size / 2.0

        self.max_steps = max_steps
        self.frame_skip = frame_skip

        self.min_obstacles = min_obstacles
        self.max_obstacles = max_obstacles
        self.sensor_range = sensor_range

        # Obstacle shape variation: with probability `rect_obstacle_ratio` a
        # sampled obstacle is an axis-aligned (or rotated) rectangle instead of
        # a circle.  0.0 = all circles, 1.0 = all rectangles.
        self.rect_obstacle_ratio = float(np.clip(rect_obstacle_ratio, 0.0, 1.0))
        self.rect_rotation = bool(rect_rotation)

        # Physics constants (previously hardcoded; now configurable so a run
        # config can fully describe the environment, e.g. for re-implementing
        # it in Unity / Gazebo).
        self.robot_radius = float(robot_radius)
        self.target_radius = float(target_radius)
        self.max_speed = float(max_speed)
        self.turn_angle = math.radians(float(turn_angle_deg))

        # 0 forward
        # 1 forward-left
        # 2 forward-right
        # 3 turn left
        # 4 turn right
        # 5 stop
        self.action_space = spaces.Discrete(6)

        # Observation:
        # robot_x, robot_y, robot_angle,
        # target_x, target_y,
        # distance_to_target, angle_to_target,
        # front_sensor, left_sensor, right_sensor
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(10,),
            dtype=np.float32,
        )

        self.robot_pos = np.zeros(2, dtype=np.float32)
        self.robot_angle = 0.0
        self.target_pos = np.zeros(2, dtype=np.float32)
        self.obstacles = []

        self.current_step = 0
        self.energy_used = 0.0
        self.stuck_counter = 0
        self.stop_counter = 0

        self.reached_target = False
        self.collision = False
        self.stuck = False

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.energy_used = 0.0
        self.stuck_counter = 0
        self.stop_counter = 0

        self.reached_target = False
        self.collision = False
        self.stuck = False

        margin = 1.0
        min_target_distance = max(4.0, self.world_size * 0.25)

        # Try to generate a valid, reachable map
        for _ in range(100):
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

            while self._distance(robot, target) < min_target_distance:
                target = self.np_random.uniform(
                    low=-self.half + margin,
                    high=self.half - margin,
                    size=(2,),
                ).astype(np.float32)

            obstacles = self._sample_obstacles(robot, target)
            if obstacles is None:
                continue

            if self._path_exists(robot, target, obstacles):
                self.robot_pos = robot
                self.target_pos = target
                self.robot_angle = float(self.np_random.uniform(-np.pi, np.pi))
                self.obstacles = obstacles
                return self._get_obs(), self._get_info()

        # Fallback: simple map with no obstacles if generation keeps failing
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

        self.robot_pos = robot
        self.target_pos = target
        self.robot_angle = float(self.np_random.uniform(-np.pi, np.pi))
        self.obstacles = []

        return self._get_obs(), self._get_info()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action):
        action = int(action)
        self.current_step += 1

        prev_pos = self.robot_pos.copy()
        prev_distance = self._distance(self.robot_pos, self.target_pos)

        self.reached_target = False
        self.collision = False
        self.stuck = False

        energy_costs = {
            0: 1.0,    # forward
            1: 0.85,   # forward-left
            2: 0.85,   # forward-right
            3: 0.25,   # turn left
            4: 0.25,   # turn right
            5: 0.05,   # stop
        }
        self.energy_used += energy_costs[action]

        # Turning actions
        if action == 1:
            self.robot_angle += self.turn_angle / 2.0
        elif action == 2:
            self.robot_angle -= self.turn_angle / 2.0
        elif action == 3:
            self.robot_angle += self.turn_angle
        elif action == 4:
            self.robot_angle -= self.turn_angle

        self.robot_angle = self._normalize_angle(self.robot_angle)

        # Moving actions
        if action in [0, 1, 2]:
            for _ in range(self.frame_skip):
                self._move_forward()
                if self._check_collision():
                    self.collision = True
                    break

        current_distance = self._distance(self.robot_pos, self.target_pos)
        displacement = self._distance(self.robot_pos, prev_pos)

        progress = prev_distance - current_distance
        near_target = current_distance < self.target_radius * 1.5

        reward = 0.0

        # Reward actual progress toward target
        reward += progress * 2.0

        # Reward actual movement
        reward += displacement * 1.5

        # General time penalty
        reward -= 0.03

        # Punish stopping when not near target
        if action == 5 and not near_target:
            self.stop_counter += 1
            # Base stop penalty
            reward -= 0.25
            # Increasing penalty for continuous stopping
            reward -= 0.02 * min(self.stop_counter, 20)
        else:
            self.stop_counter = 0

        # If the robot keeps stopping, treat it as stuck
        if self.stop_counter >= 25 and not near_target:
            self.stuck = True
            reward -= 30.0

        # Target reached
        if current_distance < self.target_radius:
            self.reached_target = True
            reward += 100.0

        # Collision
        if self.collision:
            reward -= 100.0

        # Anti-stuck detection based on displacement
        if displacement < 0.08 and current_distance > self.target_radius * 1.5:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        if self.stuck_counter >= 35:
            self.stuck = True
            reward -= 25.0

        # Timeout penalty
        if (
            self.current_step >= self.max_steps
            and not self.reached_target
            and not self.collision
        ):
            reward -= 15.0
            reward -= current_distance * 0.5

        terminated = bool(self.reached_target or self.collision or self.stuck)
        truncated = bool(self.current_step >= self.max_steps and not terminated)

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        pass

    # ------------------------------------------------------------------
    # Physics helpers
    # ------------------------------------------------------------------
    def _move_forward(self):
        direction = np.array(
            [math.cos(self.robot_angle), math.sin(self.robot_angle)],
            dtype=np.float32,
        )
        self.robot_pos = self.robot_pos + direction * self.max_speed
        self.robot_pos = np.clip(self.robot_pos, -self.half, self.half).astype(
            np.float32
        )

    def _check_collision(self):
        for obstacle in self.obstacles:
            if self._robot_hits_obstacle(self.robot_pos, obstacle):
                return True
        return False

    # ------------------------------------------------------------------
    # Obstacle / map generation
    # ------------------------------------------------------------------
    def _sample_obstacles(self, robot, target):
        count = int(
            self.np_random.integers(self.min_obstacles, self.max_obstacles + 1)
        )
        obstacles = []
        margin = 1.0

        for _ in range(count):
            placed = False
            for _ in range(100):
                pos = self.np_random.uniform(
                    low=-self.half + margin,
                    high=self.half - margin,
                    size=(2,),
                ).astype(np.float32)

                if self.np_random.random() < self.rect_obstacle_ratio:
                    # Rectangle obstacle (axis-aligned or rotated).  Half
                    # extents sampled from the same size range as circle
                    # radii so both shapes occupy a comparable footprint.
                    half_w = float(self.np_random.uniform(0.5, 1.3))
                    half_h = float(self.np_random.uniform(0.5, 1.3))
                    angle = (
                        float(self.np_random.uniform(-np.pi / 2.0, np.pi / 2.0))
                        if self.rect_rotation
                        else 0.0
                    )
                    obstacle = {
                        "shape": "rect",
                        "pos": pos,
                        "width": 2.0 * half_w,
                        "height": 2.0 * half_h,
                        "angle": angle,
                    }
                else:
                    obstacle = {
                        "shape": "circle",
                        "pos": pos,
                        "radius": float(self.np_random.uniform(0.5, 1.3)),
                    }

                # Keep obstacles away from robot start and target
                bound = self._obstacle_bounding_radius(obstacle)
                if self._distance(pos, robot) < bound + self.robot_radius + 1.5:
                    continue
                if self._distance(pos, target) < bound + self.target_radius + 1.5:
                    continue

                # Avoid overlap with existing obstacles (conservative: use
                # bounding circles for the separation test)
                valid = True
                for existing in obstacles:
                    existing_bound = self._obstacle_bounding_radius(existing)
                    if (
                        self._distance(pos, existing["pos"])
                        < bound + existing_bound + 0.5
                    ):
                        valid = False
                        break

                if valid:
                    obstacles.append(obstacle)
                    placed = True
                    break

            if not placed:
                return None

        return obstacles

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------
    def _obstacle_bounding_radius(self, obstacle):
        """Radius of the circle that fully contains an obstacle."""
        if obstacle.get("shape") == "rect":
            return 0.5 * math.hypot(obstacle["width"], obstacle["height"])
        return float(obstacle["radius"])

    def _to_local(self, point, obstacle):
        """Transform a world point into a rectangle obstacle's local frame."""
        c = math.cos(-obstacle["angle"])
        s = math.sin(-obstacle["angle"])
        d = np.asarray(point, dtype=np.float64) - np.asarray(
            obstacle["pos"], dtype=np.float64
        )
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])

    def _robot_hits_obstacle(self, point, obstacle):
        """Whether a disc of radius robot_radius at `point` overlaps an obstacle."""
        if obstacle.get("shape") == "rect":
            local = self._to_local(point, obstacle)
            half_w = 0.5 * obstacle["width"] + self.robot_radius
            half_h = 0.5 * obstacle["height"] + self.robot_radius
            return abs(local[0]) < half_w and abs(local[1]) < half_h

        distance = self._distance(point, obstacle["pos"])
        return distance < self.robot_radius + obstacle["radius"]

    def _rect_ray_distance(self, origin, direction, obstacle):
        """Distance along a ray (origin, direction) to an inflated rectangle.

        The rectangle is inflated by the robot radius (Minkowski expansion in
        the rect's local frame); returns sensor_range if the ray misses.
        """
        local_origin = self._to_local(origin, obstacle)
        c = math.cos(-obstacle["angle"])
        s = math.sin(-obstacle["angle"])
        local_dir = np.array(
            [
                c * direction[0] - s * direction[1],
                s * direction[0] + c * direction[1],
            ]
        )

        half_w = 0.5 * obstacle["width"] + self.robot_radius
        half_h = 0.5 * obstacle["height"] + self.robot_radius

        eps = 1e-8

        t_min = -float("inf")
        t_max = float("inf")

        for axis in range(2):
            half = half_w if axis == 0 else half_h
            o = local_origin[axis]
            d = local_dir[axis]

            if abs(d) < eps:
                # Ray parallel to this slab pair: must start inside it
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

        # Starting inside the inflated rectangle -> touching
        if t_min <= 0.0:
            return 0.0

        return float(min(t_min, self.sensor_range))

    def _path_exists(self, robot, target, obstacles):
        resolution = 1.0
        n = int(self.world_size / resolution)
        if n <= 2:
            return True

        grid = np.zeros((n, n), dtype=bool)

        # Mark occupied cells (shape-aware, inflated by the robot radius)
        for obstacle in obstacles:
            inflated = self.robot_radius + 0.1

            bound = self._obstacle_bounding_radius(obstacle) + inflated
            pos = obstacle["pos"]

            x_min = max(0, int((pos[0] - bound + self.half) / resolution))
            x_max = min(n - 1, int((pos[0] + bound + self.half) / resolution))
            y_min = max(0, int((pos[1] - bound + self.half) / resolution))
            y_max = min(n - 1, int((pos[1] + bound + self.half) / resolution))

            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    cell_x = x * resolution - self.half + resolution / 2.0
                    cell_y = y * resolution - self.half + resolution / 2.0

                    if self._robot_hits_obstacle(
                        np.array([cell_x, cell_y]),
                        obstacle,
                    ):
                        grid[y, x] = True

        def to_grid(point):
            x = int((point[0] + self.half) / resolution)
            y = int((point[1] + self.half) / resolution)
            x = min(max(x, 0), n - 1)
            y = min(max(y, 0), n - 1)
            return x, y

        start_x, start_y = to_grid(robot)
        goal_x, goal_y = to_grid(target)

        if grid[start_y, start_x] or grid[goal_y, goal_x]:
            return False

        visited = np.zeros((n, n), dtype=bool)
        queue = deque([(start_x, start_y)])
        visited[start_y, start_x] = True

        while queue:
            x, y = queue.popleft()
            if x == goal_x and y == goal_y:
                return True

            neighbors = [
                (x + 1, y),
                (x - 1, y),
                (x, y + 1),
                (x, y - 1),
                (x + 1, y + 1),
                (x + 1, y - 1),
                (x - 1, y + 1),
                (x - 1, y - 1),
            ]

            for nx, ny in neighbors:
                if 0 <= nx < n and 0 <= ny < n:
                    if not grid[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((nx, ny))

        return False

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_obs(self):
        distance_to_target = self._distance(self.robot_pos, self.target_pos)
        angle_to_target = self._angle_to_target()

        max_distance = math.sqrt(2.0) * self.world_size

        sector_width = 2.0 * np.pi / 3.0  # 120 degrees

        front_sensor = self._sector_distance(0.0, sector_width)
        left_sensor = self._sector_distance(2.0 * np.pi / 3.0, sector_width)
        right_sensor = self._sector_distance(-2.0 * np.pi / 3.0, sector_width)

        obs = np.array(
            [
                self.robot_pos[0] / self.half,
                self.robot_pos[1] / self.half,
                self.robot_angle / np.pi,
                self.target_pos[0] / self.half,
                self.target_pos[1] / self.half,
                distance_to_target / max_distance,
                angle_to_target / np.pi,
                front_sensor,
                left_sensor,
                right_sensor,
            ],
            dtype=np.float32,
        )

        return np.clip(obs, -1.0, 1.0)

    def _get_info(self):
        return {
            "robot_pos": self.robot_pos.copy(),
            "target_pos": self.target_pos.copy(),
            "num_obstacles": len(self.obstacles),
            "distance_to_target": self._distance(self.robot_pos, self.target_pos),
            "step": self.current_step,
            "reached_target": self.reached_target,
            "collision": self.collision,
            "stuck": self.stuck,
            "energy_used": self.energy_used,
        }

    # ------------------------------------------------------------------
    # Sensors (raycast-style)
    # ------------------------------------------------------------------
    def _sector_distance(self, center_relative_angle, spread):
        angles = np.linspace(
            center_relative_angle - spread / 2.0,
            center_relative_angle + spread / 2.0,
            5,
        )

        distances = []
        for angle in angles:
            distances.append(self._ray_distance(float(angle)))

        min_distance = min(distances)
        return float(np.clip(min_distance / self.sensor_range, 0.0, 1.0))

    def _ray_distance(self, relative_angle):
        angle = self.robot_angle + relative_angle
        direction = np.array(
            [math.cos(angle), math.sin(angle)],
            dtype=np.float32,
        )

        t = self.sensor_range
        eps = 1e-8

        # Distance to walls
        tx = float("inf")
        ty = float("inf")

        if direction[0] > eps:
            tx = (self.half - self.robot_pos[0]) / direction[0]
        elif direction[0] < -eps:
            tx = (-self.half - self.robot_pos[0]) / direction[0]

        if direction[1] > eps:
            ty = (self.half - self.robot_pos[1]) / direction[1]
        elif direction[1] < -eps:
            ty = (-self.half - self.robot_pos[1]) / direction[1]

        if tx < 0:
            tx = float("inf")
        if ty < 0:
            ty = float("inf")

        t = min(t, tx, ty)

        # Distance to obstacles
        for obstacle in self.obstacles:
            if obstacle.get("shape") == "rect":
                t = min(
                    t,
                    self._rect_ray_distance(self.robot_pos, direction, obstacle),
                )
                continue

            inflated_radius = obstacle["radius"] + self.robot_radius
            oc = self.robot_pos - obstacle["pos"]

            c = float(np.dot(oc, oc) - inflated_radius * inflated_radius)

            # Already inside the inflated obstacle radius
            if c <= 0.0:
                return 0.0

            b = 2.0 * float(np.dot(oc, direction))
            discriminant = b * b - 4.0 * c

            if discriminant >= 0.0:
                sqrt_discriminant = math.sqrt(discriminant)
                t1 = (-b - sqrt_discriminant) / 2.0

                if t1 > 0.0:
                    t = min(t, t1)
                else:
                    t2 = (-b + sqrt_discriminant) / 2.0
                    if t2 > 0.0:
                        t = min(t, t2)

        return float(np.clip(t, 0.0, self.sensor_range))

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------
    def _distance(self, a, b):
        return float(np.linalg.norm(a - b))

    def _angle_to_target(self):
        target_vector = self.target_pos - self.robot_pos
        target_angle = np.arctan2(target_vector[1], target_vector[0])
        angle_difference = target_angle - self.robot_angle
        return self._normalize_angle(angle_difference)

    def _normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle