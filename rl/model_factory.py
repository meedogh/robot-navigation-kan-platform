"""Central factory for policy network architectures.

All consumers (training, saved-model evaluation, live simulation, explainability)
build networks through this module so the architecture always matches the
checkpoint - including when training is started from the dashboard with custom
hyperparameters.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_MLP_ARCH: Dict[str, Any] = {"hidden_dim": 64}
DEFAULT_KAN_ARCH: Dict[str, Any] = {"hidden_dim": 32, "grid_size": 12, "grid_range": 2.0}


def network_arch(model_type: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the network kwargs for a model type based on a (partial) config dict."""
    cfg = config or {}

    if model_type == "mlp":
        return {
            "hidden_dim": int(cfg.get("mlp_hidden_dim", DEFAULT_MLP_ARCH["hidden_dim"])),
        }

    if model_type == "kan":
        return {
            "hidden_dim": int(cfg.get("kan_hidden_dim", DEFAULT_KAN_ARCH["hidden_dim"])),
            "grid_size": int(cfg.get("kan_grid_size", DEFAULT_KAN_ARCH["grid_size"])),
            "grid_range": float(cfg.get("kan_grid_range", DEFAULT_KAN_ARCH["grid_range"])),
        }

    raise ValueError(f"model_type must be 'mlp' or 'kan', got {model_type!r}")


def create_qnetwork(model_type: str, obs_dim: int, action_dim: int, config: Optional[Dict[str, Any]] = None):
    """Create a Q-network for the given model type, honoring any custom architecture."""
    arch = network_arch(model_type, config)
    return create_qnetwork_from_arch(model_type, obs_dim, action_dim, arch)


def create_qnetwork_from_arch(model_type: str, obs_dim: int, action_dim: int, arch: Optional[Dict[str, Any]] = None):
    """Create a Q-network from an explicit architecture dict (used by checkpoint loaders)."""
    arch = arch or network_arch(model_type, None)

    if model_type == "mlp":
        from rl.policies.mlp_network import MLPQNetwork

        return MLPQNetwork(obs_dim=obs_dim, action_dim=action_dim, **arch)

    if model_type == "kan":
        from rl.policies.kan_network import KANQNetwork

        return KANQNetwork(obs_dim=obs_dim, action_dim=action_dim, **arch)

    raise ValueError(f"model_type must be 'mlp' or 'kan', got {model_type!r}")


def arch_override_path(checkpoint_dir: Path, model_type: str) -> Path:
    return Path(checkpoint_dir) / f"custom_dqn_{model_type}_arch.json"


def save_arch(checkpoint_dir: Path, model_type: str, config: Dict[str, Any]) -> None:
    """Persist the exact architecture used to train a checkpoint next to it."""
    path = arch_override_path(checkpoint_dir, model_type)
    path.write_text(json.dumps(network_arch(model_type, config), indent=2) + "\n")


def load_arch(checkpoint_dir: Path, model_type: str) -> Dict[str, Any]:
    """Load the architecture a checkpoint was trained with, falling back to defaults."""
    path = arch_override_path(checkpoint_dir, model_type)
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return network_arch(model_type, None)