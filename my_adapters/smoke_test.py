"""End-to-end smoke test for the RobotNav bridge (no Unity required).

Starts the reference bridge server in a background thread, connects the
``UnityNavEnv`` adapter to it, and exercises the full Gymnasium contract:
spaces, seeded reset, a random-policy episode, info keys and the Live-page
visualization attributes.

Run from the repository root::

    python -m my_adapters.smoke_test
"""

import random
import sys
import time
import traceback

import numpy as np

from my_adapters.bridge_server import BridgeServer
from my_adapters.unity_env import UnityNavEnv

HOST = "127.0.0.1"
PORT = 5598  # test port; keep clear of the default 5577


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    server = BridgeServer(host=HOST, port=PORT, variant="v2")
    server.serve_background()
    time.sleep(0.5)  # give the listener a moment to bind

    env = UnityNavEnv(host=HOST, port=PORT, timeout=10.0)
    try:
        check(
            env.observation_space.shape == (10,),
            f"obs space shape {env.observation_space.shape} != (10,)",
        )
        check(
            getattr(env.action_space, "n", None) == 6,
            f"action space {env.action_space} != Discrete(6)",
        )
        check(abs(env.world_size - 20.0) < 1e-6, f"world_size {env.world_size}")

        obs, info = env.reset(seed=7)
        check(obs.shape == (10,), f"reset obs shape {obs.shape}")
        check(
            float(obs.min()) >= -1.0 and float(obs.max()) <= 1.0,
            "reset obs outside [-1, 1]",
        )
        for key in (
            "distance_to_target",
            "step",
            "reached_target",
            "collision",
            "stuck",
            "energy_used",
        ):
            check(key in info, f"info missing key {key!r}")
        check(
            len(env.obstacles) >= 1 or env.obstacles == [],
            "obstacles attribute malformed",
        )
        check(env.robot_pos.shape == (2,), "robot_pos attribute malformed")
        check(env.target_pos.shape == (2,), "target_pos attribute malformed")

        rng = random.Random(0)
        total_reward = 0.0
        steps = 0
        done = False
        while not done and steps < 1000:
            action = rng.randrange(env.action_space.n)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            total_reward += float(reward)
            check(
                obs.shape == (10,),
                f"step obs shape {obs.shape}",
            )
            check(
                float(obs.min()) >= -1.0 and float(obs.max()) <= 1.0,
                "step obs outside [-1, 1]",
            )
            done = bool(terminated or truncated)

        check(done, "episode never ended within 1000 steps")
        check(
            info.get("reached_target") or info.get("collision") or info.get("stuck")
            or truncated,
            "episode ended without a terminal reason",
        )

        print("PASS: bridge smoke test")
        print(f"  episode steps      : {steps}")
        print(f"  episode reward     : {total_reward:.2f}")
        print(f"  reached_target     : {bool(info.get('reached_target'))}")
        print(f"  collision          : {bool(info.get('collision'))}")
        print(f"  obstacles rendered : {len(env.obstacles)}")
        return 0
    finally:
        env.close()
        server.stop()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        print("FAIL: bridge smoke test")
        sys.exit(1)
