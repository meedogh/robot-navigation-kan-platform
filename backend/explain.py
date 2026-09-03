from pathlib import Path

import numpy as np
import torch

from simulation.env_factory import create_env
from rl.config_io import env_config_from_checkpoint_dir
from rl.model_factory import create_qnetwork_from_arch, load_arch


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
    # Derive obs/action dims from the environment the checkpoint was trained in
    # (recorded in custom_dqn_kan_config.json; falls back to the builtin env)
    # so the network architecture always matches the checkpoint - including
    # when training ran against an external Unity / Gazebo / custom env.
    env_config = env_config_from_checkpoint_dir(Path(model_path).parent, "kan")
    env = create_env(env_config)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()
    arch = load_arch(Path(model_path).parent, "kan")
    model = create_qnetwork_from_arch(
        "kan",
        obs_dim=obs_dim,
        action_dim=action_dim,
        arch=arch
    )
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
            batch = torch.zeros(len(xs), obs_dim, device=DEVICE)
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