import torch
import torch.nn as nn

from rl.policies.kan_layer import KANLayer


class KANQNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int = 10,
        action_dim: int = 4,
        hidden_dim: int = 32,
        grid_size: int = 12,
        grid_range: float = 2.0
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

        self.kan_input = KANLayer(
            in_features=obs_dim,
            out_features=hidden_dim,
            grid_size=grid_size,
            grid_range=grid_range
        )

        self.norm1 = nn.LayerNorm(hidden_dim)

        self.kan_output = KANLayer(
            in_features=hidden_dim,
            out_features=action_dim,
            grid_size=grid_size,
            grid_range=grid_range
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(obs / self.obs_scale)

        x = self.kan_input(x)
        x = self.norm1(x)

        q_values = self.kan_output(x)

        return q_values