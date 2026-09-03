"""Environment factory: build Gymnasium environments from portable run configs.

The ``environment`` section of a run config (see ``rl/config_io.py``) describes
how to construct the simulation:

    {"source": "builtin", "variant": "v2", "params": {...}}
    {"source": "module",  "module": "my_pkg.adapters:MyUnityEnv", "params": {...}}

Any external environment works as long as it follows the Gymnasium API
(``reset`` / ``step`` / ``observation_space`` / ``action_space``) with the same
observation layout and action semantics as the builtin environment.  That is
the contract that lets a world built in Unity (ML-Agents), Gazebo/ROS, or any
other simulator drive training, inference and the live dashboard here.

Environment classes are resolved lazily so an external simulator package is
only imported when it is actually used.
"""

import importlib
import inspect
from typing import Any, Dict, List, Optional, Tuple


# Builtin environments shipped with the platform (registry key -> class path).
BUILTIN_ENVS: Dict[str, str] = {
    "v1": "simulation.envs.robot_navigation_env:RobotNavigationEnv",
    "v2": "simulation.envs.robot_navigation_env_v2:RobotNavigationEnv",
}
DEFAULT_VARIANT = "v2"

# All constructor parameters the run-config format knows about.  The builtin v2
# env accepts every one of them; other env classes (including external ones)
# simply receive the subset their __init__ supports.
ENV_PARAM_NAMES: Tuple[str, ...] = (
    "world_size",
    "max_steps",
    "frame_skip",
    "min_obstacles",
    "max_obstacles",
    "sensor_range",
    "robot_radius",
    "target_radius",
    "max_speed",
    "turn_angle_deg",
)

# Parameters that must stay integers when passed to an env constructor.
ENV_INT_PARAMS = {"max_steps", "frame_skip", "min_obstacles", "max_obstacles"}

ENV_SOURCE_KEYS: Tuple[str, ...] = ("env_source", "env_variant", "env_module")


def _import_class(class_path: str):
    """Import 'package.module:ClassName' and return the class object."""
    module_path, _, class_name = class_path.partition(":")
    if not module_path or not class_name:
        raise ValueError(
            f"Invalid environment class path {class_path!r} - expected "
            f"'package.module:ClassName'"
        )
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ValueError(
            f"Could not import environment module {module_path!r}: {exc}"
        ) from exc

    env_class = getattr(module, class_name, None)
    if env_class is None:
        raise ValueError(
            f"Module {module_path!r} has no attribute {class_name!r}"
        )
    if not callable(env_class):
        raise ValueError(
            f"Environment target {class_path!r} is not a callable class"
        )
    return env_class


def resolve_env_class(
    source: str = "builtin",
    variant: Optional[str] = None,
    module: Optional[str] = None,
):
    """Return the environment class described by an environment section."""
    source = (source or "builtin").strip().lower()

    if source == "builtin":
        key = (variant or DEFAULT_VARIANT).strip()
        if key not in BUILTIN_ENVS:
            raise ValueError(
                f"Unknown builtin environment variant {key!r} "
                f"(available: {', '.join(sorted(BUILTIN_ENVS))})"
            )
        return _import_class(BUILTIN_ENVS[key])

    if source == "module":
        if not module or ":" not in module:
            raise ValueError(
                "environment.module must look like 'package.module:ClassName' "
                "when environment.source is 'module'"
            )
        return _import_class(module.strip())

    raise ValueError(
        f"environment.source must be 'builtin' or 'module', got {source!r}"
    )


def _supported_kwargs(env_class, params: Optional[Dict[str, Any]]):
    """Filter params down to the keyword arguments the env class accepts."""
    params = params or {}
    try:
        signature = inspect.signature(env_class.__init__)
    except (TypeError, ValueError):
        return dict(params), sorted(params)

    accepted = set(signature.parameters) - {"self"}
    kwargs: Dict[str, Any] = {}
    skipped: List[str] = []
    for name, value in params.items():
        if name in accepted:
            if name in ENV_INT_PARAMS:
                kwargs[name] = int(value)
            else:
                kwargs[name] = float(value)
        else:
            skipped.append(name)
    return kwargs, sorted(skipped)


def env_config_from_flat(flat: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract the run-config ``environment`` section from a flat training config.

    Flat configs (the dict passed to ``rl.train_custom_dqn.train``) keep env
    knobs under ``env_*`` keys plus the source keys ``env_source``,
    ``env_variant`` and ``env_module``.
    """
    flat = flat or {}
    params: Dict[str, Any] = {}
    for name in ENV_PARAM_NAMES:
        key = f"env_{name}"
        if key in flat and flat[key] is not None:
            params[name] = flat[key]

    return {
        "source": flat.get("env_source") or "builtin",
        "variant": flat.get("env_variant") or DEFAULT_VARIANT,
        "module": flat.get("env_module") or None,
        "params": params,
    }


def _is_env_section(env_config: Any) -> bool:
    """A run-config environment section vs. a flat training config."""
    if not isinstance(env_config, dict):
        return False
    return any(key in env_config for key in ("source", "variant", "module", "params"))


def _check_spaces(env, spec: Dict[str, Any]) -> None:
    """Warn when an env does not match the observation/action spec it declares."""
    if not isinstance(spec, dict):
        return

    obs_spec = spec.get("observation") or {}
    shape = obs_spec.get("shape")
    if shape is not None:
        actual = list(getattr(env.observation_space, "shape", ()) or ())
        if actual != list(shape):
            print(
                f"[env_factory] WARNING: env observation shape {actual} differs "
                f"from the spec {list(shape)}. Checkpoints trained on the "
                f"original env will not be compatible with this env."
            )

    act_spec = spec.get("action") or {}
    n_actions = act_spec.get("n")
    if n_actions is not None:
        actual_n = getattr(env.action_space, "n", None)
        if actual_n is not None and actual_n != n_actions:
            print(
                f"[env_factory] WARNING: env action count {actual_n} differs "
                f"from the spec {n_actions}. Checkpoints trained on the "
                f"original env will not be compatible with this env."
            )


def create_env(env_config: Optional[Dict[str, Any]] = None):
    """Instantiate the environment described by a run-config ``environment``
    section (or a flat training config, which is converted automatically).

    With no argument this returns the default builtin environment, exactly as
    before the factory existed.
    """
    if env_config is None:
        section: Dict[str, Any] = {}
    elif _is_env_section(env_config):
        section = env_config
    else:
        section = env_config_from_flat(env_config)

    env_class = resolve_env_class(
        source=section.get("source", "builtin"),
        variant=section.get("variant"),
        module=section.get("module"),
    )

    kwargs, skipped = _supported_kwargs(env_class, section.get("params"))
    if skipped:
        print(
            f"[env_factory] note: environment {env_class.__name__} ignores "
            f"unsupported params: {', '.join(skipped)}"
        )

    env = env_class(**kwargs)
    _check_spaces(env, section.get("spec") or {})
    return env


def builtin_env_spec() -> Dict[str, Any]:
    """Interface contract of the builtin (v2) environment.

    This is embedded in exported run configs so an external implementation -
    e.g. a Unity ML-Agents scene or a Gazebo/ROS world - can be built to match
    observation layout, action semantics and physics exactly.
    """
    return {
        "observation": {
            "type": "box",
            "shape": [10],
            "low": -1.0,
            "high": 1.0,
            "features": [
                "robot_x",
                "robot_y",
                "robot_angle",
                "target_x",
                "target_y",
                "distance_to_target",
                "angle_to_target",
                "front_sensor",
                "left_sensor",
                "right_sensor",
            ],
        },
        "action": {
            "type": "discrete",
            "n": 6,
            "labels": [
                "forward",
                "forward_left",
                "forward_right",
                "turn_left",
                "turn_right",
                "stop",
            ],
        },
        "episode_end": ["target_reached", "collision", "stuck", "max_steps"],
        "info_keys": [
            "distance_to_target",
            "step",
            "reached_target",
            "collision",
            "stuck",
            "energy_used",
        ],
        "visualization_attrs": [
            "world_size",
            "robot_pos",
            "robot_angle",
            "target_pos",
            "obstacles",
        ],
    }

