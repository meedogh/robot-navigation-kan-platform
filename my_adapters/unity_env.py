"""Gymnasium adapter for an external RobotNav simulator (Unity / Gazebo / ...).

``UnityNavEnv`` is a client of the RobotNav bridge protocol (see
``my_adapters/bridge_protocol.py``): the simulator (e.g. the
``unity_bridge/RobotNavBridge.cs`` MonoBehaviour in a Unity scene, a ROS node,
or ``my_adapters.bridge_server`` for testing) runs a TCP server, and this
class turns it into a standard Gymnasium environment.

Because the adapter only speaks the protocol, the platform's training loop,
evaluation, KAN explainability and the live dashboard work unchanged with a
world that is physically simulated elsewhere.  Register it in a run config::

    "environment": {
        "source": "module",
        "module": "my_adapters.unity_env:UnityNavEnv",
        "params": {"host": "127.0.0.1", "port": 5577}
    }

Observation / action spaces are taken from the simulator's ``hello``
handshake, so they automatically match whatever scene is connected (the run
config's ``environment.spec`` is cross-checked against them by
``simulation.env_factory``).
"""

import socket
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from my_adapters.bridge_protocol import (
    DEFAULT_PORT,
    PROTOCOL_VERSION,
    BridgeProtocolError,
    recv_message,
    send_message,
)


class UnityNavEnv(gym.Env):
    """Gymnasium environment backed by an external simulator over TCP.

    The simulator owns the physics: it randomizes the map, executes the
    discrete actions, computes rewards and decides when episodes end.  This
    class only relays commands and normalizes what comes back into the
    Gymnasium API plus the visualization attributes the Live page renders
    (``world_size``, ``robot_pos``, ``robot_angle``, ``target_pos``,
    ``obstacles``).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        timeout: float = 10.0,
        world_size: float = 20.0,
        max_steps: int = 300,
        **ignored,
    ):
        super().__init__()

        if ignored:
            print(
                f"[UnityNavEnv] note: ignoring parameters not used by the "
                f"bridge adapter: {', '.join(sorted(ignored))}"
            )

        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self.max_steps = int(max_steps)

        self._sock: Optional[socket.socket] = None
        self._step_count = 0
        self._closed = False

        self._connect()

        # Attributes the Live page / frame builder reads (with fallbacks).
        self.world_size = float(self._hello.get("world_size") or world_size)
        self.robot_pos = np.zeros(2, dtype=np.float32)
        self.robot_angle = 0.0
        self.target_pos = np.zeros(2, dtype=np.float32)
        self.obstacles: list = []


    # ------------------------------------------------------------------
    # Connection handling
    # ------------------------------------------------------------------
    def _connect(self) -> None:
        try:
            self._sock = socket.create_connection(
                (self.host, self.port), timeout=self.timeout
            )
        except OSError as exc:
            raise RuntimeError(
                f"Could not connect to the simulator bridge at "
                f"{self.host}:{self.port} ({exc}). Start the simulator first "
                f"- e.g. open the Unity scene with RobotNavBridge attached "
                f"and press Play, or run "
                f"'python -m my_adapters.bridge_server --port {self.port}'."
            ) from exc

        try:
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._hello = self._recv_state_or_hello()
        except Exception:
            self._close_socket()
            raise

        if self._hello.get("type") != "hello":
            raise BridgeProtocolError(
                f"Expected a 'hello' handshake from the simulator, got "
                f"{self._hello.get('type')!r}."
            )

        remote_protocol = int(self._hello.get("protocol", PROTOCOL_VERSION))
        if remote_protocol != PROTOCOL_VERSION:
            raise BridgeProtocolError(
                f"Bridge protocol mismatch: simulator speaks v{remote_protocol}, "
                f"this adapter speaks v{PROTOCOL_VERSION}."
            )

        obs_dim = int(self._hello.get("obs_dim") or 10)
        n_actions = int(self._hello.get("n_actions") or 6)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(n_actions)

    def _recv_state_or_hello(self) -> Dict[str, Any]:
        payload = recv_message(self._sock)
        if payload.get("type") == "error":
            raise RuntimeError(
                f"Simulator bridge error: {payload.get('message', 'unknown')}"
            )
        return payload

    def _send(self, payload: Dict[str, Any]) -> None:
        if self._sock is None:
            raise RuntimeError("The bridge connection is closed.")
        try:
            send_message(self._sock, payload)
        except OSError as exc:
            raise RuntimeError(
                f"Lost connection to the simulator bridge ({exc})."
            ) from exc

    # ------------------------------------------------------------------
    # State ingestion
    # ------------------------------------------------------------------
    def _ingest(self, state: Dict[str, Any]):
        if state.get("type") != "state":
            raise BridgeProtocolError(
                f"Expected a 'state' message from the simulator, got "
                f"{state.get('type')!r}."
            )

        obs = np.asarray(state.get("obs") or [], dtype=np.float32)
        if obs.shape != self.observation_space.shape:
            raise BridgeProtocolError(
                f"Simulator sent obs shape {obs.shape}, expected "
                f"{self.observation_space.shape}."
            )

        reward = float(state.get("reward") or 0.0)
        terminated = bool(state.get("terminated", False))
        if "truncated" in state:
            truncated = bool(state["truncated"])
        else:
            # Simulator does not manage the step limit - apply it client-side.
            truncated = (
                self.max_steps > 0
                and self._step_count >= self.max_steps
                and not terminated
            )

        info = dict(state.get("info") or {})
        info.setdefault("reached_target", False)
        info.setdefault("collision", False)
        info.setdefault("stuck", False)
        info.setdefault("distance_to_target", 0.0)
        info.setdefault("step", self._step_count)
        info.setdefault("energy_used", 0.0)

        # Visualization attributes (optional; Live page degrades without them).
        self.world_size = float(state.get("world_size") or self.world_size)
        robot_pos = state.get("robot_pos")
        if robot_pos is not None:
            self.robot_pos = np.asarray(robot_pos, dtype=np.float32)[:2]
        self.robot_angle = float(state.get("robot_angle") or 0.0)
        target_pos = state.get("target_pos")
        if target_pos is not None:
            self.target_pos = np.asarray(target_pos, dtype=np.float32)[:2]
        obstacles = state.get("obstacles")
        if obstacles is not None:
            self.obstacles = [
                (np.asarray(o[:2], dtype=np.float32), float(o[2]))
                for o in obstacles
                if len(o) >= 3
            ]

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self._send({"cmd": "reset", "seed": seed, "options": options or {}})
        state = self._recv_state_or_hello()
        self._step_count = 0
        obs, _reward, _term, _trunc, info = self._ingest(state)
        return obs, info

    def step(self, action):
        action = int(action)
        self._step_count += 1
        self._send({"cmd": "step", "action": action})
        state = self._recv_state_or_hello()
        return self._ingest(state)

    def render(self):
        """The simulator renders itself (Unity scene / ROS visualization)."""

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._sock is not None:
            try:
                self._send({"cmd": "close"})
            except Exception:
                pass
            self._close_socket()

    def _close_socket(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


