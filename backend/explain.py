from pathlib import Path

import numpy as np
import torch

from rl.policies.kan_network import KANQNetwork


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OBSERVATION_LABELS = [
    "robot_x",
    "robot_y",
    "robot_angle",
    "target_x",
    "target_y",
    "distance_to_target",
    "angle_to_target",
    "front_obstacle",
    "left_obstacle",
    "right_obstacle",
]


def build_kan_explanation(model_path: Path):
    """
    Extract explainability data from the input KAN layer:
      - grid points of the basis functions
      - per-input-feature importance (L1 norm of learned edge coefficients)
      - learned univariate function curves for the most important features
    """
    model = KANQNetwork(obs_dim=10, action_dim=4, hidden_dim=32)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    kan_input = model.kan_input
    grid = kan_input.grid.detach().cpu().numpy().tolist()
    coeffs = kan_input.coefficients.detach().cpu().numpy()  # (in, out, basis)

    # Feature importance: total magnitude of all edge functions for each input feature
    importance = np.abs(coeffs).sum(axis=(1, 2))  # (in,)
    importance = importance / (importance.sum() + 1e-9)

    # Evaluate learned functions over a fine grid for the top features
    xs = np.linspace(-2.0, 2.0, 80)
    feature_curves = []
    top_indices = np.argsort(importance)[::-1][:4]

    with torch.no_grad():
        for idx in top_indices:
            # basis values for a sweep of x on this single feature
            x_tensor = torch.tensor(xs, dtype=torch.float32).unsqueeze(1).to(DEVICE)
            # Build a zero input batch, vary only feature idx
            batch = torch.zeros(len(xs), 10, device=DEVICE)
            batch[:, idx] = torch.tensor(xs, dtype=torch.float32).to(DEVICE)
            out = kan_input(batch)  # (samples, hidden)
            # Sum across hidden units to get overall influence of this feature
            curve = out.abs().sum(dim=1).cpu().numpy().tolist()
            feature_curves.append(
                {
                    "feature": OBSERVATION_LABELS[idx],
                    "index": int(idx),
                    "importance": float(importance[idx]),
                    "xs": xs.tolist(),
                    "ys": curve,
                }
            )

    return {
        "model_path": str(model_path),
        "grid": grid,
        "labels": OBSERVATION_LABELS,
        "importance": [float(v) for v in importance],
        "top_features": feature_curves,
    }