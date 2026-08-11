import copy

import numpy as np
import torch

from rl.dqn.replay_buffer import ReplayBuffer
from rl.policies.mlp_network import MLPQNetwork
from rl.policies.kan_network import KANQNetwork


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        network_type: str = "mlp",
        lr: float = 0.0005,
        gamma: float = 0.99,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        target_update_interval: int = 500,
        device: str = "cpu"
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        self.target_update_interval = target_update_interval

        self.device = torch.device(device)

        if network_type == "mlp":
            self.policy = MLPQNetwork(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=64
            )
        elif network_type == "kan":
            self.policy = KANQNetwork(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=32,
                grid_size=12,
                grid_range=2.0
            )
        else:
            raise ValueError("network_type must be 'mlp' or 'kan'")

        self.policy.to(self.device)

        # Target network
        self.target_network = copy.deepcopy(self.policy)
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=lr
        )

        self.buffer = ReplayBuffer(capacity=buffer_size)

        self.timesteps = 0
        self.update_count = 0

    def current_epsilon(self) -> float:
        fraction = min(1.0, self.timesteps / self.epsilon_decay_steps)
        epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)
        return epsilon

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training:
            self.timesteps += 1
            epsilon = self.current_epsilon()

            if np.random.random() < epsilon:
                return np.random.randint(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32
            ).unsqueeze(0).to(self.device)

            q_values = self.policy(state_tensor)
            action = int(q_values.argmax(dim=1).item())

        return action

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values = next_q_values.max(dim=1).values

            target_q_values = rewards + (1.0 - dones) * self.gamma * max_next_q_values

        current_q_values = self.policy(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        loss = torch.nn.functional.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)

        self.optimizer.step()

        self.update_count += 1

        if self.update_count % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.policy.state_dict())

        return float(loss.item())