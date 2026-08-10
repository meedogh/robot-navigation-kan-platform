import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RobotNavigationEnv(gym.Env):
    """
    Simple robot navigation environment.

    Actions:
        0 = move forward
        1 = turn left
        2 = turn right
        3 = stop
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=200):
        super().__init__()

        self.max_steps = max_steps
        self.current_step = 0

        # Actions: forward, left, right, stop
        self.action_space = spaces.Discrete(4)

        # Observation:
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

        # Robot state
        self.robot_pos = np.zeros(2, dtype=np.float32)
        self.robot_angle = 0.0

        # Target state
        self.target_pos = np.zeros(2, dtype=np.float32)

        # Simple obstacle representation
        # For now, we use one obstacle distance placeholder.
        self.obstacle_pos = np.zeros(2, dtype=np.float32)

        self.target_radius = 0.5
        self.robot_radius = 0.3
        self.obstacle_radius = 0.5

        self.max_speed = 0.5
        self.turn_angle = np.pi / 12  # 15 degrees

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        # Random robot position
        self.robot_pos = self.np_random.uniform(
            low=-self.world_size / 2,
            high=self.world_size / 2,
            size=(2,)
        ).astype(np.float32)

        # Random robot angle
        self.robot_angle = self.np_random.uniform(
            low=-np.pi,
            high=np.pi
        )

        # Random target position
        self.target_pos = self.np_random.uniform(
            low=-self.world_size / 2,
            high=self.world_size / 2,
            size=(2,)
        ).astype(np.float32)

        # Make sure target is not too close to robot
        while self._distance(self.robot_pos, self.target_pos) < 2.0:
            self.target_pos = self.np_random.uniform(
                low=-self.world_size / 2,
                high=self.world_size / 2,
                size=(2,)
            ).astype(np.float32)

        # Random obstacle position
        self.obstacle_pos = self.np_random.uniform(
            low=-self.world_size / 2,
            high=self.world_size / 2,
            size=(2,)
        ).astype(np.float32)

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1

        previous_distance = self._distance(self.robot_pos, self.target_pos)

        # Simple action dynamics
        if action == 0:
            # Move forward
            direction = np.array([
                np.cos(self.robot_angle),
                np.sin(self.robot_angle)
            ], dtype=np.float32)

            self.robot_pos += self.max_speed * direction

        elif action == 1:
            # Turn left
            self.robot_angle += self.turn_angle

        elif action == 2:
            # Turn right
            self.robot_angle -= self.turn_angle

        elif action == 3:
            # Stop
            pass

        # Normalize angle
        self.robot_angle = self._normalize_angle(self.robot_angle)

        # Keep robot inside world
        self.robot_pos = np.clip(
            self.robot_pos,
            -self.world_size / 2,
            self.world_size / 2
        )

        current_distance = self._distance(self.robot_pos, self.target_pos)

        # Basic reward
        reward = previous_distance - current_distance
        reward -= 0.01

        reached_target = current_distance < self.target_radius
        collision = self._check_collision()
        timeout = self.current_step >= self.max_steps

        if reached_target:
            reward += 100.0

        if collision:
            reward -= 100.0

        terminated = reached_target or collision
        truncated = timeout

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        # Later we can add pygame visualization
        pass

    def _get_obs(self):
        distance_to_target = self._distance(self.robot_pos, self.target_pos)

        angle_to_target = self._angle_to_target()

        # Very simple obstacle information
        obstacle_distance = self._distance(self.robot_pos, self.obstacle_pos)

        # Placeholder sensor values
        front_obstacle = obstacle_distance
        left_obstacle = obstacle_distance
        right_obstacle = obstacle_distance

        obs = np.array([
            self.robot_pos[0],
            self.robot_pos[1],
            self.robot_angle,
            self.target_pos[0],
            self.target_pos[1],
            distance_to_target,
            angle_to_target,
            front_obstacle,
            left_obstacle,
            right_obstacle
        ], dtype=np.float32)

        return obs

    def _get_info(self):
        return {
            "robot_pos": self.robot_pos.copy(),
            "target_pos": self.target_pos.copy(),
            "obstacle_pos": self.obstacle_pos.copy(),
            "distance_to_target": self._distance(self.robot_pos, self.target_pos),
            "step": self.current_step
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

    def _check_collision(self):
        distance_to_obstacle = self._distance(self.robot_pos, self.obstacle_pos)
        return distance_to_obstacle < (self.robot_radius + self.obstacle_radius)