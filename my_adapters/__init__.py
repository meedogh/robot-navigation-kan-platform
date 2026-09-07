"""External simulator adapters for the RobotNav platform.

``my_adapters.unity_env.UnityNavEnv`` is a Gymnasium environment that connects
to an external simulator (Unity, Gazebo, or any program implementing the
RobotNav bridge protocol) over TCP.  The simulator runs the physics and sends
observations / rewards / episode flags; training, evaluation and the live
dashboard here consume it exactly like the builtin environment.

``my_adapters.bridge_server`` is a reference implementation of the simulator
side of the protocol (wrapping the builtin v2 environment).  It is useful to
test the bridge without Unity and as the parity reference for the Unity C#
script in ``unity_bridge/RobotNavBridge.cs``.

Protocol (version 1): TCP, 4-byte big-endian payload length followed by a
UTF-8 JSON payload.  See ``bridge_protocol.py`` and the README section
"External simulators".
"""
