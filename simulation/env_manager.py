"""Custom environment save / load manager.

Provides persistence for user-designed obstacle layouts (custom environments)
created via the dashboard Custom Environment editor (Setup page). Saved
environments can be reloaded for training, evaluation, or live visualization.

Storage
-------
Custom environments are stored under experiments/envs/<name>/ as:

    experiments/envs/<name>/
    ├── env_config.json      # Full environment section
    └── metadata.json         # name, description, created_at, etc.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from simulation import env_factory

BASE_DIR = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = BASE_DIR / "experiments"
ENVS_DIR = EXPERIMENTS_DIR / "envs"
ENVS_DIR.mkdir(parents=True, exist_ok=True)

ENV_STORAGE_DIR = ENVS_DIR
ENV_CONFIG_FILENAME = "env_config.json"
METADATA_FILENAME = "metadata.json"


def _env_dir(name: str) -> Path:
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_")
    if not safe_name:
        raise ValueError(f"Invalid environment name: {name}")
    return ENVS_DIR / safe_name


def _env_config_path(name: str) -> Path:
    return _env_dir(name) / ENV_CONFIG_FILENAME


def _metadata_path(name: str) -> Path:
    return _env_dir(name) / METADATA_FILENAME


def _layout_summary(layout: Any) -> Dict[str, Any]:
    if layout is None or layout == "":
        return {"obstacle_count": 0, "shapes": {}}
    if isinstance(layout, str):
        try:
            layout = json.loads(layout)
        except json.JSONDecodeError:
            return {"obstacle_count": -1, "shapes": {}, "parse_error": True}
    if not isinstance(layout, list):
        return {"obstacle_count": -1, "shapes": {}, "invalid_type": type(layout).__name__}
    shapes: Dict[str, int] = {}
    for item in layout:
        if isinstance(item, dict):
            shape = item.get("shape", "unknown")
            shapes[shape] = shapes.get(shape, 0) + 1
    return {"obstacle_count": len(layout), "shapes": shapes}


def save_custom_environment(
    name: str,
    layout: Any = None,
    world_size: float = 20.0,
    max_steps: int = 300,
    frame_skip: int = 3,
    sensor_range: float = 12.0,
    robot_radius: float = 0.35,
    target_radius: float = 0.8,
    max_speed: float = 0.35,
    turn_angle_deg: float = 30.0,
    description: str = "",
    overwrite: bool = False,
) -> Dict[str, Any]:
    env_dir = _env_dir(name)
    if env_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Environment '{name}' already exists.")
        shutil.rmtree(env_dir)
    env_dir.mkdir(parents=True, exist_ok=True)

    env_section = env_factory.environment_section(
        source="builtin", variant="custom",
        world_size=world_size, max_steps=max_steps, frame_skip=frame_skip,
        sensor_range=sensor_range, robot_radius=robot_radius,
        target_radius=target_radius, max_speed=max_speed,
        turn_angle_deg=turn_angle_deg, layout=layout,
    )

    _env_config_path(name).write_text(json.dumps(env_section, indent=2))

    try:
        probe_env = env_factory.create_env({"environment": env_section})
        probe_env.reset(seed=0)
        probe_env.close()
    except Exception as exc:
        shutil.rmtree(env_dir)
        raise ValueError(f"Failed to validate environment layout: {exc}") from exc

    metadata = {
        "name": name,
        "description": description,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "world_size": world_size,
        "max_steps": max_steps,
        "sensor_range": sensor_range,
        "obstacle_count": 0,
        "layout_summary": _layout_summary(layout),
    }
    _metadata_path(name).write_text(json.dumps(metadata, indent=2))
    return metadata


def load_custom_environment(name: str) -> Dict[str, Any]:
    env_config_path = _env_config_path(name)
    if not env_config_path.exists():
        available = [e["name"] for e in list_saved_environments()]
        raise FileNotFoundError(f"Environment '{name}' not found. Available: {available}")
    return json.loads(env_config_path.read_text())


def load_environment_for_training(name: str) -> Dict[str, Any]:
    env_section = load_custom_environment(name)
    params = env_section.get("params", {})
    flat = {
        "env_source": env_section.get("source", "builtin"),
        "env_variant": env_section.get("variant", "custom"),
    }
    for param_name in env_factory.ENV_PARAM_NAMES:
        if param_name in params:
            flat[f"env_{param_name}"] = params[param_name]
    return flat


def get_environment_spec(name: str) -> Dict[str, Any]:
    """Return public metadata + env section for a saved environment."""
    metadata_path = _metadata_path(name)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Environment '{name}' not found")
    metadata = json.loads(metadata_path.read_text())
    env_section = load_custom_environment(name)
    return {
        "name": metadata["name"],
        "description": metadata.get("description", ""),
        "created_at": metadata.get("created_at"),
        "world_size": metadata.get("world_size"),
        "obstacle_count": metadata.get("obstacle_count"),
        "layout_summary": metadata.get("layout_summary"),
        "env_section": env_section,
    }


def list_saved_environments() -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if not ENVS_DIR.exists():
        return results
    for child in sorted(ENVS_DIR.iterdir()):
        if not child.is_dir():
            continue
        meta_path = child / METADATA_FILENAME
        if not meta_path.exists():
            continue
        try:
            metadata = json.loads(meta_path.read_text())
            results.append(metadata)
        except (json.JSONDecodeError, OSError):
            continue
    return results


def delete_custom_environment(name: str, missing_ok: bool = False) -> None:
    env_dir = _env_dir(name)
    if not env_dir.exists():
        if not missing_ok:
            raise FileNotFoundError(f"Environment '{name}' not found")
        return
    shutil.rmtree(env_dir)
