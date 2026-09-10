"""Reference simulator-side implementation of the RobotNav bridge protocol.

Runs a TCP server (protocol v1, see ``my_adapters/bridge_protocol.py``) and
serves episodes from the builtin environment (v2 by default).  Three uses:

1. **Bridge testing without Unity** - start it, point
   ``my_adapters.unity_env.UnityNavEnv`` at the port, and the whole platform
   (training, evaluation, live view) runs against it.
2. **Parity reference** - the Unity C# script (``unity_bridge/RobotNavBridge.cs``)
   is a 1:1 port of the same builtin environment, so its messages must match
   what this server emits.
3. **Remote simulation** - the environment can live in a separate process or
   on another machine even without Unity.

Usage::

    python -m my_adapters.bridge_server --port 5577 --world-size 20
"""

import argparse
import socket
import threading
from typing import Any, Dict, Optional

from rl.frames import serialize_obstacles
from my_adapters.bridge_protocol import (
    DEFAULT_PORT,
    PROTOCOL_VERSION,
    BridgeProtocolError,
    recv_message,
    send_message,
)


def _state_payload(
    env, reward: float, terminated: bool, truncated: bool
) -> Dict[str, Any]:
    """Build a protocol ``state`` message from a builtin env after a step."""
    info = env._get_info() if hasattr(env, "_get_info") else {}
    return {
        "type": "state",
        "obs": [float(v) for v in env._get_obs()],
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": {
            "distance_to_target": float(info.get("distance_to_target", 0.0)),
            "step": int(info.get("step", 0)),
            "reached_target": bool(info.get("reached_target", False)),
            "collision": bool(info.get("collision", False)),
            "stuck": bool(info.get("stuck", False)),
            "energy_used": float(info.get("energy_used", 0.0)),
        },
        "world_size": float(env.world_size),
        "robot_pos": [float(env.robot_pos[0]), float(env.robot_pos[1])],
        "robot_angle": float(env.robot_angle),
        "target_pos": [float(env.target_pos[0]), float(env.target_pos[1])],
        # Bridge protocol v1 wire format: [[x, y, radius]] bounding circles
        # (protocol docs: my_adapters/bridge_protocol.py).  Rect obstacles use
        # their bounding-circle radius; full shape data only flows through the
        # platform's own live-view frames (rl.frames).
        "obstacles": [
            [o["x"], o["y"], o["radius"]]
            for o in serialize_obstacles(env.obstacles)
        ],
    }



class BridgeServer:
    """TCP server that speaks the RobotNav bridge protocol for a builtin env."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        variant: str = "v2",
        env_params: Optional[Dict[str, Any]] = None,
    ):
        self.host = host
        self.port = int(port)
        self.variant = variant
        self.env_params = dict(env_params or {})
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    def _build_env(self, seed: Optional[int] = None):
        from simulation.env_factory import create_env

        env = create_env(
            {
                "source": "builtin",
                "variant": self.variant,
                "params": self.env_params,
            }
        )
        obs, _info = env.reset(seed=seed)
        return env, obs

    def _serve_connection(self, conn: socket.socket) -> None:
        """Handle one client connection (reset/step/close loop)."""
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        env = None
        try:
            # Builtin v2 contract: 10 observations, 6 discrete actions.
            probe, _obs = self._build_env()
            obs_dim = int(probe.observation_space.shape[0])
            n_actions = int(probe.action_space.n)
            world_size = float(getattr(probe, "world_size", 20.0))
            probe.close()

            send_message(
                conn,
                {
                    "type": "hello",
                    "protocol": PROTOCOL_VERSION,
                    "world_size": world_size,
                    "obs_dim": obs_dim,
                    "n_actions": n_actions,
                },
            )

            while not self._stop.is_set():
                try:
                    message = recv_message(conn)
                except BridgeProtocolError:
                    break

                cmd = message.get("cmd")
                if cmd == "close":
                    break
                if cmd == "reset":
                    seed = message.get("seed")
                    env, _obs = self._build_env(
                        int(seed) if seed is not None else None
                    )
                    send_message(conn, _state_payload(env, 0.0, False, False))
                elif cmd == "step":
                    if env is None:
                        raise BridgeProtocolError(
                            "Received 'step' before 'reset'."
                        )
                    action = int(message.get("action", 0))
                    _obs, reward, terminated, truncated, _info = env.step(action)
                    send_message(
                        conn,
                        _state_payload(env, reward, terminated, truncated),
                    )
                    if terminated or truncated:
                        env.close()
                        env = None
                else:
                    raise BridgeProtocolError(f"Unknown command {cmd!r}.")
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        except Exception as exc:  # report anything else back to the client
            try:
                send_message(conn, {"type": "error", "message": str(exc)})
            except OSError:
                pass
        finally:
            if env is not None:
                env.close()
            try:
                conn.close()
            except OSError:
                pass

    # ------------------------------------------------------------------
    def serve(self) -> None:
        """Accept clients until :meth:`stop` is called (blocking).

        Each client is served on its own thread: the platform opens one
        connection per environment instance (e.g. train env + eval env).
        """
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(4)
        server.settimeout(0.5)
        print(
            f"[bridge_server] serving builtin env variant {self.variant!r} "
            f"on {self.host}:{self.port}"
        )
        threads = []
        try:
            while not self._stop.is_set():
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue
                print(f"[bridge_server] client connected: {addr[0]}:{addr[1]}")

                def _serve(conn=conn, addr=addr):
                    try:
                        self._serve_connection(conn)
                    finally:
                        print(f"[bridge_server] client disconnected: {addr[0]}:{addr[1]}")

                thread = threading.Thread(target=_serve, daemon=True)
                thread.start()
                threads.append(thread)
        finally:
            server.close()

    def serve_background(self) -> threading.Thread:
        """Run :meth:`serve` on a daemon thread and return it."""
        if self._thread is not None and self._thread.is_alive():
            return self._thread
        self._stop.clear()
        self._thread = threading.Thread(target=self.serve, daemon=True)
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


def _main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m my_adapters.bridge_server",
        description=(
            "Reference RobotNav bridge simulator: serves the builtin "
            "environment over the bridge protocol (protocol v1)."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--variant",
        default="v2",
        help="Builtin environment variant to serve (v1 or v2).",
    )
    parser.add_argument("--world-size", type=float, default=20.0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--min-obstacles", type=int, default=3)
    parser.add_argument("--max-obstacles", type=int, default=6)

    args = parser.parse_args()

    server = BridgeServer(
        host=args.host,
        port=args.port,
        variant=args.variant,
        env_params={
            "world_size": args.world_size,
            "max_steps": args.max_steps,
            "min_obstacles": args.min_obstacles,
            "max_obstacles": args.max_obstacles,
        },
    )
    try:
        server.serve()
    except KeyboardInterrupt:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


