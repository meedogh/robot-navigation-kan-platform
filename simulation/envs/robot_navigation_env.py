import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RobotNavigationEnv(gym.Env):
    """
    Simple robot navigation environment (V1).

    NOTE: as of the sensor fix, front/left/right observations are true
    raycast-style sector distances in [0, 1] (like the v2 environment).
    Checkpoints trained on the *old* v1 implementation are NOT compatible
    with this observation layout - retrain any v1 models.

    Actions:
        0 = move forward
        1 = turn left
        2 = turn right
        3 = stop
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=200, sensor_range=3.0):
        super().__init__()

        self.max_steps = max_steps
        self.current_step = 0

        # Maximum distance the raycast sensors can see (normalized to [0, 1])
        self.sensor_range = float(sensor_range)

        # forward, left, right, stop
        self.action_space = spaces.Discrete(4)
        
        self.reached_target = False
        self.collision = False

        # robot_x, robot_y, robot_angle,
        # target_x, target_y,
        # distance_to_target, angle_to_target,
        # front_obstacle, left_obstacle, right_obstacle
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32
        )

        # World boundaries
        self.world_size = 10.0

        # state
        self.robot_pos = np.zeros(2, dtype=np.float32)
        self.robot_angle = 0.0

        # target state
        self.target_pos = np.zeros(2, dtype=np.float32)

        self.obstacle_pos = np.zeros(2, dtype=np.float32)

        self.target_radius = 0.5
        self.robot_radius = 0.3
        self.obstacle_radius = 0.5

        self.max_speed = 0.5
        self.turn_angle = np.pi / 12  # 15 degrees

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
    
        self.current_step = 0
        
        self.reached_target = False
        
        self.collision = False
        
        self.robot_pos = self.np_random.uniform(
            low=-self.world_size / 2,
            high=self.world_size / 2,
            size=(2,)
        ).astype(np.float32)

        self.robot_angle = self.np_random.uniform(
            low=-np.pi,
            high=np.pi
        )

        self.target_pos = self.np_random.uniform(
            low=-self.world_size / 2,
            high=self.world_size / 2,
            size=(2,)
        ).astype(np.float32)

        while self._distance(self.robot_pos, self.target_pos) < 2.0:
            self.target_pos = self.np_random.uniform(
                low=-self.world_size / 2,
                high=self.world_size / 2,
                size=(2,)
            ).astype(np.float32)

        self.obstacle_pos = self._sample_obstacle()

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1

        previous_distance = self._distance(self.robot_pos, self.target_pos)

        if action == 0:
            direction = np.array([
                np.cos(self.robot_angle),
                np.sin(self.robot_angle)
            ], dtype=np.float32)

            self.robot_pos += self.max_speed * direction

        elif action == 1: # LEFT
            self.robot_angle += self.turn_angle

        elif action == 2: # RIGHT
            self.robot_angle -= self.turn_angle

        elif action == 3:
            pass

        self.robot_angle = self._normalize_angle(self.robot_angle)

        self.robot_pos = np.clip(
            self.robot_pos,
            -self.world_size / 2,
            self.world_size / 2
        )

        current_distance = self._distance(self.robot_pos, self.target_pos)

        reward = previous_distance - current_distance
        reward -= 0.01

        self.reached_target = bool(current_distance < self.target_radius)
        self.collision = bool(self._check_collision())
        timeout = self.current_step >= self.max_steps

        if self.reached_target:
            reward += 100.0

        if self.collision:
            reward -= 100.0

        terminated = self.reached_target or self.collision
        truncated = timeout

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        # VISUALIZATION
        pass

    def _get_obs(self):
        distance_to_target = self._distance(self.robot_pos, self.target_pos)

        angle_to_target = self._angle_to_target()

        # Raycast-style sector sensors (front / left / right), each the minimum
        # ray distance over a small angular sector, normalized to [0, 1] where
        # 0 = touching and 1 = nothing within sensor_range.
        front_sensor = self._sector_distance(0.0)
        left_sensor = self._sector_distance(np.pi / 2.0)
        right_sensor = self._sector_distance(-np.pi / 2.0)

        obs = np.array([
            self.robot_pos[0],
            self.robot_pos[1],
            self.robot_angle,
            self.target_pos[0],
            self.target_pos[1],
            distance_to_target,
            angle_to_target,
            front_sensor,
            left_sensor,
            right_sensor
        ], dtype=np.float32)

        return obs

    def _get_info(self):
        return {
            "robot_pos": self.robot_pos.copy(),
            "target_pos": self.target_pos.copy(),
            "obstacle_pos": self.obstacle_pos.copy(),
            "distance_to_target": self._distance(self.robot_pos, self.target_pos),
            "step": self.current_step,
            "reached_target": self.reached_target,
            "collision": self.collision
        }

    def _distance(self, a, b):
        return float(np.linalg.norm(a - b))

    def _angle_to_target(self):
        target_vector = self.target_pos - self.robot_pos
        target_angle = np.arctan2(target_vector[1], target_vector[0])
        angle_difference = target_angle - self.robot_angle
        return self._normalize_angle(angle_difference)

    def _normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


    def _sample_obstacle(self):
        min_robot_obstacle_distance = 1.0
        min_target_obstacle_distance = 1.0

        while True:
            obstacle_pos = self.np_random.uniform(
                low=-self.world_size / 2,
                high=self.world_size / 2,
                size=(2,)
            ).astype(np.float32)

            robot_obstacle_distance = self._distance(self.robot_pos, obstacle_pos)
            target_obstacle_distance = self._distance(self.target_pos, obstacle_pos)

            if (
                robot_obstacle_distance > min_robot_obstacle_distance
                and target_obstacle_distance > min_target_obstacle_distance
            ):
                return obstacle_pos
        

    def _check_collision(self):
        distance_to_obstacle = self._distance(self.robot_pos, self.obstacle_pos)
        return distance_to_obstacle < (self.robot_radius + self.obstacle_radius)

    # ------------------------------------------------------------------
    # Sensors (raycast-style, same approach as the v2 environment)
    # ------------------------------------------------------------------
    def _sector_distance(self, center_relative_angle, spread=np.pi / 3.0):
        angles = np.linspace(
            center_relative_angle - spread / 2.0,
            center_relative_angle + spread / 2.0,
            5,
        )

        distances = [self._ray_distance(float(angle)) for angle in angles]

        min_distance = min(distances)
        return float(np.clip(min_distance / self.sensor_range, 0.0, 1.0))

    def _ray_distance(self, relative_angle):
        angle = self.robot_angle + relative_angle
        direction = np.array(
            [np.cos(angle), np.sin(angle)],
            dtype=np.float32,
        )

        half = self.world_size / 2.0
        eps = 1e-8

        # Distance to walls
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

        # Distance to the obstacle
        inflated_radius = self.obstacle_radius + self.robot_radius
        oc = self.robot_pos - self.obstacle_pos

        c = float(np.dot(oc, oc) - inflated_radius * inflated_radius)

        # Already inside the inflated obstacle radius
        if c <= 0.0:
            return 0.0

        b = 2.0 * float(np.dot(oc, direction))
        discriminant = b * b - 4.0 * c

        if discriminant >= 0.0:
            sqrt_discriminant = np.sqrt(discriminant)
            t1 = (-b - sqrt_discriminant) / 2.0

            if t1 > 0.0:
                t = min(t, t1)
            else:
                t2 = (-b + sqrt_discriminant) / 2.0
                if t2 > 0.0:
                    t = min(t, t2)

        return float(np.clip(t, 0.0, self.sensor_range))