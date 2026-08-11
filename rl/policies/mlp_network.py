import torch
import torch.nn as nn


class MLPQNetwork(nn.Module):
    """
    Simple DQN MLP network.

    Input:
        robot observation vector

    Output:
        Q-values for each discrete action
    """

    def __init__(
        self,
        obs_dim: int = 10,
        action_dim: int = 4,
        hidden_dim: int = 64
    ):
        super().__init__()

        self.register_buffer(
            "obs_scale",
            torch.tensor(
                [
                    5.0,   # robot_x
                    5.0,   # robot_y
                    3.1416,  # robot_angle
                    5.0,   # target_x
                    5.0,   # target_y
                    10.0,  # distance_to_target
                    3.1416,  # angle_to_target
                    10.0,  # front_obstacle
                    10.0,  # left_obstacle
                    10.0,  # right_obstacle
                ],
                dtype=torch.float32
            )
        )

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(obs / self.obs_scale)
        q_values = self.network(x)
        return q_values