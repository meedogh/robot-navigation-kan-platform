"""Portable run-config import / export.

A *run config* is a single JSON file that fully describes a training run:
the training / DQN / network hyperparameters **and** the environment the run
should execute in (world geometry, physics, or a pointer to an external
simulator environment such as Unity ML-Agents or Gazebo/ROS).

File format (``robotnav-run-config`` v1)::

    {
      "format": "robotnav-run-config",
      "schema_version": 1,
      "name": "my-run",
      "description": "what this run is for",
      "created_at": "2026-09-04T12:00:00+00:00",
      "training":   { ...training / DQN / network keys... },
      "environment": {
        "source": "builtin" | "module",
        "variant": "v2",                      # when source == builtin
        "module": "my_pkg.adapters:MyEnv",    # when source == module
        "params": { "world_size": 20.0, ... },
        "spec":   { "observation": {...}, "action": {...}, ... }
      }
    }

The ``environment.spec`` block documents the observation layout, action
semantics and info keys the training stack expects - it is the contract an
environment built in Unity, Gazebo or any other simulator must satisfy for
training / inference to run against it unchanged.

Usage::

    python -m rl.config_io export --out run.json [--from-model kan]
    python -m rl.config_io validate run.json
    python -m rl.train_custom_dqn --config run.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from rl.train_custom_dqn import DEFAULT_TRAINING_CONFIG, validate_training_config
from simulation import env_factory


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "experiments" / "results"

FORMAT_NAME = "robotnav-run-config"
SCHEMA_VERSION = 1

# Keys of the run-config file, for strict validation (typo protection).
_TOP_LEVEL_KEYS = {
    "format",
    "schema_version",
    "name",
    "description",
    "created_at",
    "training",
    "environment",
}
_ENVIRONMENT_KEYS = {"source", "variant", "module", "params", "spec"}

# Flat-config keys that belong to the environment section, not `training`.
_ENV_FLAT_KEYS = {f"env_{name}" for name in env_factory.ENV_PARAM_NAMES} | set(
    env_factory.ENV_SOURCE_KEYS
)


def _reject_unknown_keys(keys, allowed, where: str) -> None:
    unknown = sorted(set(keys) - set(allowed))
    if unknown:
        known = sorted(set(allowed))
        raise ValueError(
            f"Unknown keys in {where}: {unknown}. Known keys: {known}"
        )


def validate_run_config(data: Any) -> Dict[str, Any]:
    """Validate a run config dict (as parsed from JSON) and return it.

    Raises ``ValueError`` with a precise message when the file is malformed,
    uses an unsupported schema version, or contains unknown / mistyped keys.
    """
    if not isinstance(data, dict):
        raise ValueError("Run config must be a JSON object")

    if data.get("format") != FORMAT_NAME:
        raise ValueError(
            f"'format' must be {FORMAT_NAME!r}, got {data.get('format')!r}. "
            "Is this a RobotNav run config file?"
        )

    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version {version!r} (this build supports "
            f"{SCHEMA_VERSION})"
        )

    _reject_unknown_keys(data.keys(), _TOP_LEVEL_KEYS, "run config")

    name = data.get("name")
    description = data.get("description")
    if name is not None and not isinstance(name, str):
        raise ValueError("'name' must be a string")
    if description is not None and not isinstance(description, str):
        raise ValueError("'description' must be a string")

    training = data.get("training")
    if training is not None:
        if not isinstance(training, dict):
            raise ValueError("'training' must be a JSON object")
        _reject_unknown_keys(
            training.keys(),
            {k for k in DEFAULT_TRAINING_CONFIG if k not in _ENV_FLAT_KEYS},
            "training",
        )

    environment = data.get("environment")
    if environment is not None:
        if not isinstance(environment, dict):
            raise ValueError("'environment' must be a JSON object")
        _reject_unknown_keys(environment.keys(), _ENVIRONMENT_KEYS, "environment")

        source = environment.get("source") or "builtin"
        if source not in ("builtin", "module"):
            raise ValueError(
                f"environment.source must be 'builtin' or 'module', got {source!r}"
            )

        module = environment.get("module")
        if source == "module":
            if not isinstance(module, str) or ":" not in module:
                raise ValueError(
                    "environment.module must look like 'package.module:ClassName' "
                    "when environment.source is 'module'"
                )
        if module is not None and not isinstance(module, str):
            raise ValueError("environment.module must be a string")

        variant = environment.get("variant")
        if variant is not None:
            if not isinstance(variant, str):
                raise ValueError("environment.variant must be a string")
            if source == "builtin" and variant not in env_factory.BUILTIN_ENVS:
                raise ValueError(
                    f"Unknown builtin environment variant {variant!r} "
                    f"(available: {', '.join(sorted(env_factory.BUILTIN_ENVS))})"
                )

        params = environment.get("params")
        if params is not None:
            if not isinstance(params, dict):
                raise ValueError("environment.params must be a JSON object")
            _reject_unknown_keys(
                params.keys(), env_factory.ENV_PARAM_NAMES, "environment.params"
            )
            for key, value in params.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(
                        f"environment.params.{key} must be a number, got {value!r}"
                    )

        spec = environment.get("spec")
        if spec is not None and not isinstance(spec, dict):
            raise ValueError("environment.spec must be a JSON object")

    return data


def run_config_from_flat(
    flat: Optional[Dict[str, Any]],
    name: str = "unnamed-run",
    description: str = "",
) -> Dict[str, Any]:
    """Build a portable run config from a flat training config.

    The flat config is the dict used by ``rl.train_custom_dqn.train`` and the
    dashboard's Setup page (``model_type``, ``learning_rate``, ``env_world_size``,
    ``env_module``, ...).  It is validated and merged over the defaults first,
    then split into the ``training`` and ``environment`` sections.
    """
    flat = validate_training_config(flat)

    env_params = {
        param: flat[f"env_{param}"] for param in env_factory.ENV_PARAM_NAMES
    }
    training_section = {
        key: value for key, value in flat.items() if not key.startswith("env_")
    }

    source = flat.get("env_source") or "builtin"
    spec = env_factory.builtin_env_spec() if source == "builtin" else {}

    return {
        "format": FORMAT_NAME,
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "description": description,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training": training_section,
        "environment": {
            "source": source,
            "variant": flat.get("env_variant") or env_factory.DEFAULT_VARIANT,
            "module": flat.get("env_module") or None,
            "params": env_params,
            "spec": spec,
        },
    }


def flat_config_from_run(run: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a validated run config into the training config dict used by
    ``rl.train_custom_dqn.train`` (merged over the defaults and validated)."""
    run = validate_run_config(run)

    training = dict(run.get("training") or {})
    _reject_unknown_keys(
        training.keys(),
        {k for k in DEFAULT_TRAINING_CONFIG if k not in _ENV_FLAT_KEYS},
        "training",
    )

    environment = run.get("environment") or {}
    params = environment.get("params") or {}

    flat: Dict[str, Any] = dict(training)
    for param, value in params.items():
        flat[f"env_{param}"] = value
    flat["env_source"] = environment.get("source") or "builtin"
    flat["env_variant"] = environment.get("variant") or env_factory.DEFAULT_VARIANT
    flat["env_module"] = environment.get("module") or None

    return validate_training_config(flat)


def export_run_config(
    path,
    config: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    description: str = "",
) -> Dict[str, Any]:
    """Write a portable run config JSON for a flat training config to ``path``."""
    run = run_config_from_flat(
        config,
        name=name or Path(path).stem or "unnamed-run",
        description=description,
    )
    Path(path).write_text(json.dumps(run, indent=2) + "\n")
    return run


def load_run_config(path) -> Dict[str, Any]:
    """Read and validate a run config JSON file."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Run config file not found: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    return validate_run_config(data)


def load_run_config_flat(path) -> Dict[str, Any]:
    """Read a run config JSON file and return the flattened training config."""
    return flat_config_from_run(load_run_config(path))


def env_config_from_checkpoint_dir(
    checkpoint_dir, model_type: str
) -> Optional[Dict[str, Any]]:
    """Environment section for a saved checkpoint, from the config JSON that
    ``train()`` persisted next to it.

    Used by evaluation / live simulation / explainability so a model is always
    run in the same environment it was trained in - including external
    (Unity / Gazebo / custom module) environments.  Returns ``None`` when no
    config file exists (legacy checkpoints), meaning "default builtin env".
    """
    path = Path(checkpoint_dir) / f"custom_dqn_{model_type}_config.json"
    if not path.exists():
        return None
    try:
        flat = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(flat, dict):
        return None
    return env_factory.env_config_from_flat(flat)


def _last_run_flat(model_type: str) -> Optional[Dict[str, Any]]:
    """Flat config of the most recent run of ``model_type`` (or None)."""
    if model_type not in ("mlp", "kan"):
        raise ValueError(f"model_type must be 'mlp' or 'kan', got {model_type!r}")
    path = RESULTS_DIR / f"custom_dqn_{model_type}_config.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rl.config_io",
        description="Export / validate portable RobotNav run config files.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_export = sub.add_parser(
        "export", help="Write a run config JSON from the defaults or a past run"
    )
    p_export.add_argument("--out", required=True, help="Output JSON path")
    p_export.add_argument(
        "--from-model",
        choices=["mlp", "kan"],
        default=None,
        help="Start from the config of the most recent run of this model "
        "instead of the platform defaults",
    )
    p_export.add_argument("--name", default=None, help="Run name stored in the file")
    p_export.add_argument(
        "--description", default="", help="Free-text description stored in the file"
    )

    p_validate = sub.add_parser(
        "validate", help="Validate a run config and print the resolved flat config"
    )
    p_validate.add_argument("path", help="Run config JSON path")

    args = parser.parse_args(argv)

    if args.command == "export":
        base_config: Optional[Dict[str, Any]] = None
        if args.from_model is not None:
            base_config = _last_run_flat(args.from_model)
            if base_config is None:
                print(f"No previous run config found for '{args.from_model}'.")
                return 1
        run = export_run_config(
            args.out,
            config=base_config,
            name=args.name,
            description=args.description,
        )
        print(f"Run config written to: {args.out}")
        print(
            f"  environment: source={run['environment']['source']!r}, "
            f"variant={run['environment']['variant']!r}, "
            f"module={run['environment']['module']!r}"
        )
        print(f"  training keys: {len(run['training'])}")
        return 0

    if args.command == "validate":
        run = load_run_config(args.path)
        flat = flat_config_from_run(run)
        env_section = run.get("environment") or {}
        source = env_section.get("source") or "builtin"
        print(f"OK: {args.path} is a valid {FORMAT_NAME} v{SCHEMA_VERSION} file.")
        print(f"  name: {run.get('name')!r}")
        print(f"  environment source: {source!r}")
        if source == "module":
            print(f"  environment module: {env_section.get('module')!r}")
        print("  resolved flat training config:")
        for key in sorted(flat):
            print(f"    {key}: {flat[key]!r}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())




