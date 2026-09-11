#!/usr/bin/env python3
"""CLI tool for managing saved custom environments.

Usage:
    python -m rl.env_manager_cli save --name my-env --layout '[{"shape": "circle", "x": 0, "y": 0, "radius": 1}]'
    python -m rl.env_manager_cli load --name my-env
    python -m rl.env_manager_cli list
    python -m rl.env_manager_cli delete --name my-env
"""
import argparse
import json
import sys
from pathlib import Path

from simulation.env_manager import (
    save_custom_environment,
    load_custom_environment,
    load_environment_for_training,
    list_saved_environments,
    delete_custom_environment,
    get_environment_spec,
)


def cmd_save(args):
    """Save a custom environment."""
    # Parse layout from JSON string or file
    layout = None
    if args.layout:
        try:
            layout = json.loads(args.layout)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON for layout: {e}")
            return 1
    elif args.layout_file:
        try:
            layout = json.loads(Path(args.layout_file).read_text())
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error: Could not read layout file: {e}")
            return 1

    try:
        metadata = save_custom_environment(
            name=args.name,
            layout=layout,
            world_size=args.world_size,
            max_steps=args.max_steps,
            sensor_range=args.sensor_range,
            description=args.description or "",
            overwrite=args.overwrite,
        )
        print(f"Environment '{args.name}' saved successfully.")
        print(f"  World size: {metadata['world_size']}")
        print(f"  Max steps: {metadata['max_steps']}")
        print(f"  Obstacles: {metadata['obstacle_count']}")
        print(f"  Location: {Path(metadata.get('location', 'experiments/envs/' + args.name))}")
        return 0
    except FileExistsError:
        print(f"Error: Environment '{args.name}' already exists. Use --overwrite to replace.")
        return 1
    except ValueError as e:
        print(f"Error: {e}")
        return 1


def cmd_load(args):
    """Load and display a saved environment."""
    try:
        if args.format == "section":
            env = load_custom_environment(args.name)
        elif args.format == "training":
            env = load_environment_for_training(args.name)
        else:
            env = get_environment_spec(args.name)

        print(json.dumps(env, indent=2))
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1


def cmd_list(args):
    """List all saved environments."""
    envs = list_saved_environments()
    if not envs:
        print("No saved environments found.")
        return 0

    print(f"Saved environments ({len(envs)}):")
    for env in envs:
        print(f"  - {env['name']}")
        if env.get("description"):
            print(f"    Description: {env['description']}")
        print(f"    World size: {env.get('world_size', 'N/A')}")
        print(f"    Created: {env.get('created_at', 'N/A')}")
        summary = env.get("layout_summary", {})
        print(f"    Obstacles: {summary.get('obstacle_count', 'N/A')}")
    return 0


def cmd_delete(args):
    """Delete a saved environment."""
    try:
        delete_custom_environment(args.name, missing_ok=args.missing_ok)
        print(f"Environment '{args.name}' deleted.")
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m rl.env_manager_cli",
        description="Manage saved custom environments for RobotNav training"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Save command
    p_save = sub.add_parser("save", help="Save a custom environment")
    p_save.add_argument("--name", required=True, help="Environment name")
    p_save.add_argument("--layout", help="JSON string of obstacle layout")
    p_save.add_argument("--layout-file", help="Path to JSON file with layout")
    p_save.add_argument("--world-size", type=float, default=20.0)
    p_save.add_argument("--max-steps", type=int, default=300)
    p_save.add_argument("--sensor-range", type=float, default=12.0)
    p_save.add_argument("--description", default="")
    p_save.add_argument("--overwrite", action="store_true")
    p_save.set_defaults(func=cmd_save)

    # Load command
    p_load = sub.add_parser("load", help="Load a saved environment")
    p_load.add_argument("--name", required=True, help="Environment name")
    p_load.add_argument(
        "--format",
        choices=["section", "training", "spec"],
        default="spec",
        help="Output format",
    )
    p_load.set_defaults(func=cmd_load)

    # List command
    p_list = sub.add_parser("list", help="List all saved environments")
    p_list.set_defaults(func=cmd_list)

    # Delete command
    p_del = sub.add_parser("delete", help="Delete a saved environment")
    p_del.add_argument("--name", required=True, help="Environment name")
    p_del.add_argument("--missing-ok", action="store_true")
    p_del.set_defaults(func=cmd_delete)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
