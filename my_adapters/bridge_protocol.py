"""Wire format for the RobotNav simulator bridge (protocol version 1).

Transport: TCP.  Every message is a 4-byte big-endian unsigned integer payload
length followed by that many bytes of UTF-8 JSON.

Message flow (Python adapter = client, simulator = server):

    on connect            simulator -> {"type": "hello", "protocol": 1,
                                       "world_size": float, "obs_dim": int,
                                       "n_actions": int}
    reset(seed, options)  client    -> {"cmd": "reset", "seed": int | null}
                          simulator -> state message (see below)
    step(action)          client    -> {"cmd": "step", "action": int}
                          simulator -> state message
    close()               client    -> {"cmd": "close"}   (connection closes)

A ``state`` message looks like::

    {
      "type": "state",
      "obs": [10 floats, normalized to [-1, 1] like the builtin env],
      "reward": float,
      "terminated": bool,          # target reached / collision / stuck
      "truncated": bool,           # max steps reached
      "info": {                    # keys used by evaluation & the dashboard
        "distance_to_target": float,
        "step": int,
        "reached_target": bool,
        "collision": bool,
        "stuck": bool,
        "energy_used": float
      },
      "world_size": float,             # optional, for the Live page canvas
      "robot_pos": [x, y],             # optional
      "robot_angle": float,            # optional (radians)
      "target_pos": [x, y],            # optional
      "obstacles": [[x, y, radius]]    # optional
    }

Any problem on the simulator side is reported as
``{"type": "error", "message": "..."}``.

This module is intentionally dependency-free (stdlib only) so it can be
reimplemented verbatim in Unity C# / ROS nodes / any other runtime.
"""

import json
import socket
import struct

PROTOCOL_VERSION = 1

# Default TCP port for the bridge.  Avoids the well-known service range and
# ML-Agents' default ports.
DEFAULT_PORT = 5577

# Maximum accepted payload (16 MiB) - guards against corrupted framing.
MAX_MESSAGE_BYTES = 16 * 1024 * 1024


class BridgeProtocolError(RuntimeError):
    """Raised when the simulator sends malformed or unexpected messages."""


def send_message(sock: socket.socket, payload: dict) -> None:
    """Serialize ``payload`` as JSON and send it with the length prefix."""
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    header = struct.pack(">I", len(data))
    sock.sendall(header + data)


def recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
    """Read exactly ``num_bytes`` bytes from ``sock`` or raise on EOF."""
    chunks = []
    remaining = num_bytes
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise BridgeProtocolError(
                "Simulator closed the connection unexpectedly."
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(sock: socket.socket) -> dict:
    """Receive one length-prefixed JSON message and return it as a dict."""
    (length,) = struct.unpack(">I", recv_exact(sock, 4))
    if length <= 0 or length > MAX_MESSAGE_BYTES:
        raise BridgeProtocolError(
            f"Invalid message length {length} from simulator "
            f"(expected 1..{MAX_MESSAGE_BYTES})."
        )
    data = recv_exact(sock, length)
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BridgeProtocolError(f"Malformed JSON from simulator: {exc}") from exc
    if not isinstance(payload, dict):
        raise BridgeProtocolError(
            f"Expected a JSON object from simulator, got {type(payload).__name__}."
        )
    return payload
