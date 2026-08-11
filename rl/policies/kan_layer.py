import torch
import torch.nn as nn


class KANLayer(nn.Module):
    """
    For each input feature i and output feature o, it learns a function
    f_{i,o}(x_i) represented by piecewise-linear basis functions.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 12,
        grid_range: float = 2.0
    ):
        super().__init__()

        if grid_size < 2:
            grid_size = 2

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.grid_range = grid_range

        self.register_buffer(
            "grid",
            torch.linspace(-grid_range, grid_range, grid_size)
        )

        # Distance between grid points
        self.step = (2.0 * grid_range) / float(grid_size - 1)

        # Learnable coefficients:
        # shape = (input_feature, output_feature, basis_function)
        self.coefficients = nn.Parameter(
            torch.randn(in_features, out_features, grid_size) * 0.02
        )

        # Output bias
        self.bias = nn.Parameter(torch.zeros(out_features))

    def basis_functions(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, in_features)
        returns: (batch_size, in_features, grid_size)
        """

        # Keep inputs inside KAN grid
        x = torch.clamp(x, -self.grid_range, self.grid_range)

        # Shape: (batch, in_features, grid_size)
        diff = (x.unsqueeze(-1) - self.grid) / self.step

        # Piecewise-linear "hat" basis functions
        basis = torch.relu(1.0 - torch.abs(diff))

        return basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, in_features)
        returns: (batch_size, out_features)
        """

        basis = self.basis_functions(x)

        # Combine learned edge functions
        # basis:       batch, in_features, grid_size
        # coefficients: in_features, out_features, grid_size
        output = torch.einsum("big,iog->bo", basis, self.coefficients)

        return output + self.bias