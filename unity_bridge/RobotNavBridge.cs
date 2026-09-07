// RobotNavBridge.cs - simulator-side server for the RobotNav bridge protocol.
//
// Drop this script on any GameObject in a Unity scene, press Play, and the
// RobotNav platform (training, evaluation, live dashboard) connects to this
// scene as if it were the builtin environment.
//
// The physics is a 1:1 C# port of simulation/envs/robot_navigation_env_v2.py:
//   - domain-randomized map generation with a BFS solvability check
//   - 6 discrete actions (forward, forward-left, forward-right,
//     turn-left, turn-right, stop) with identical kinematics + frame skip
//   - identical reward function (progress, displacement, stop/stuck penalties,
//     target bonus, collision penalty, timeout penalty, energy tracking)
//   - analytic sector sensors (3 x 120 deg, 5 rays each, ray-circle + wall
//     intersection - no Unity Physics, so results match Python exactly)
//   - identical observation normalization ([-1, 1])
//
// World coordinates are the environment's XY plane; they map to Unity's
// (x, z) plane (robot_pos[1] <-> transform.position.z).
//
// Wire format: 4-byte big-endian payload length + UTF-8 JSON (see
// my_adapters/bridge_protocol.py). Each client (the platform opens one per
// environment instance - e.g. train env + eval env) is served on its own
// thread with its own SimState, so training speed does not depend on the
// render frame rate and concurrent connections stay independent.

using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class RobotNavBridge : MonoBehaviour
{
    [Header("Bridge")]
    public int port = 5577;
    public bool autoStart = true;

    [Header("World (defaults match builtin v2 env)")]
    public float worldSize = 20f;
    public int maxSteps = 300;
    public int frameSkip = 3;
    public int minObstacles = 3;
    public int maxObstacles = 6;
    public float sensorRange = 12f;
    public float robotRadius = 0.35f;
    public float targetRadius = 0.8f;
    public float maxSpeed = 0.35f;
    public float turnAngleDeg = 30f;

    [Header("Visualization (auto-created primitives)")]
    public GameObject robotVisual;
    public GameObject targetVisual;
    public Material robotMaterial;
    public Material targetMaterial;
    public Material obstacleMaterial;

    // Shared visualization snapshot (latest session state, applied on main thread)
    private readonly object syncLock = new object();
    private bool needRebuildVisuals;
    private Vector2 visRobotPos;
    private float visRobotAngle;
    private Vector2 visTargetPos;
    private List<Vector3> visObstacles = new List<Vector3>();

    private TcpListener listener;
    private Thread listenerThread;
    private volatile bool quitting;

    // ------------------------------------------------------------------
    // Unity lifecycle
    // ------------------------------------------------------------------
    private void Awake()
    {
        // Keep stepping while the window is unfocused (headless-ish training).
        Application.runInBackground = true;
    }

    private void Start()
    {
        if (autoStart)
        {
            listenerThread = new Thread(ListenLoop);
            listenerThread.IsBackground = true;
            listenerThread.Start();
        }
    }

    private void Update()
    {
        ApplyVisuals();
    }

    private void OnDestroy()
    {
        quitting = true;
        if (listener != null)
        {
            try { listener.Stop(); } catch (Exception) { }
        }
        if (listenerThread != null && listenerThread.IsAlive)
        {
            listenerThread.Join(1000);
        }
    }

    private void OnApplicationQuit()
    {
        quitting = true;
    }

    // ------------------------------------------------------------------
    // TCP server (listener thread; one worker thread per client)
    // ------------------------------------------------------------------
    private void ListenLoop()
    {
        try
        {
            listener = new TcpListener(IPAddress.Loopback, port);
            listener.Server.SetSocketOption(
                SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);
            listener.Start(4);
            Debug.Log($"[RobotNavBridge] listening on 127.0.0.1:{port}");
        }
        catch (Exception exc)
        {
            Debug.LogError($"[RobotNavBridge] could not bind port {port}: {exc.Message}");
            return;
        }

        while (!quitting)
        {
            TcpClient client;
            try
            {
                if (!listener.Pending()) { Thread.Sleep(100); continue; }
                client = listener.AcceptTcpClient();
            }
            catch (Exception)
            {
                break; // listener stopped
            }

            Debug.Log("[RobotNavBridge] client connected");

            Thread clientThread = new Thread(() =>
            {
                try
                {
                    HandleClient(client);
                }
                catch (Exception exc)
                {
                    Debug.LogWarning($"[RobotNavBridge] client error: {exc.Message}");
                }
                finally
                {
                    try { client.Close(); } catch (Exception) { }
                    Debug.Log("[RobotNavBridge] client disconnected");
                }
            });
            clientThread.IsBackground = true;
            clientThread.Start();
        }
    }

    private void HandleClient(TcpClient client)
    {
        NetworkStream stream = client.GetStream();
        client.NoDelay = true;

        SimState sim = new SimState(this);

        SendJson(stream, BuildHello());
        sim.ResetSim(-1);
        PublishVisualSnapshot(sim, rebuild: true);
        SendJson(stream, sim.BuildState(0f, false, false));

        while (!quitting)
        {
            RequestMsg request = ReadRequest(stream);
            if (request == null) return; // client closed

            if (request.cmd == "close")
            {
                return;
            }
            if (request.cmd == "reset")
            {
                sim.ResetSim(request.seed);
                PublishVisualSnapshot(sim, rebuild: true);
                SendJson(stream, sim.BuildState(0f, false, false));
            }
            else if (request.cmd == "step")
            {
                float reward;
                bool terminated;
                bool truncated = sim.StepSim(request.action, out reward, out terminated);
                PublishVisualSnapshot(sim, rebuild: false);
                SendJson(stream, sim.BuildState(reward, terminated, truncated));
            }
            else
            {
                Debug.LogWarning($"[RobotNavBridge] unknown cmd '{request.cmd}'");
            }
        }
    }

    // ------------------------------------------------------------------
    // Protocol plumbing (JsonUtility DTOs + length-prefixed framing)
    // ------------------------------------------------------------------
    [Serializable] private class HelloDto
    {
        public string type;
        public int protocol;
        public float world_size;
        public int obs_dim;
        public int n_actions;
    }

    [Serializable] private class ObstacleDto { public float x; public float y; public float radius; }

    [Serializable] private class InfoDto
    {
        public float distance_to_target;
        public int step;
        public bool reached_target;
        public bool collision;
        public bool stuck;
        public float energy_used;
    }

    [Serializable] private class StateMsg
    {
        public string type;
        public float[] obs;
        public float reward;
        public bool terminated;
        public bool truncated;
        public InfoDto info;
        public float world_size;
        public float[] robot_pos;
        public float robot_angle;
        public float[] target_pos;
        public ObstacleDto[] obstacles;
    }

    [Serializable] private class RequestMsg { public string cmd; public int action; public long seed; }

    private HelloDto BuildHello()
    {
        return new HelloDto
        {
            type = "hello",
            protocol = 1,
            world_size = worldSize,
            obs_dim = 10,
            n_actions = 6,
        };
    }

    private static byte[] ReadExact(NetworkStream stream, int count)
    {
        byte[] buffer = new byte[count];
        int read = 0;
        while (read < count)
        {
            int n = stream.Read(buffer, read, count - read);
            if (n <= 0) return null; // EOF
            read += n;
        }
        return buffer;
    }

    private RequestMsg ReadRequest(NetworkStream stream)
    {
        byte[] header = ReadExact(stream, 4);
        if (header == null) return null;

        int length = (header[0] << 24) | (header[1] << 16) | (header[2] << 8) | header[3];
        if (length <= 0 || length > (16 * 1024 * 1024))
        {
            throw new IOException($"Invalid message length {length}");
        }

        byte[] payload = ReadExact(stream, length);
        if (payload == null) return null;

        return JsonUtility.FromJson<RequestMsg>(Encoding.UTF8.GetString(payload));
    }

    private static void SendJson(NetworkStream stream, object message)
    {
        byte[] payload = Encoding.UTF8.GetBytes(JsonUtility.ToJson(message));
        int length = payload.Length;
        byte[] header =
        {
            (byte)((length >> 24) & 0xFF),
            (byte)((length >> 16) & 0xFF),
            (byte)((length >> 8) & 0xFF),
            (byte)(length & 0xFF),
        };
        stream.Write(header, 0, 4);
        stream.Write(payload, 0, payload.Length);
        stream.Flush();
    }

    // ------------------------------------------------------------------
    // Visualization (worker threads -> main thread snapshot)
    // ------------------------------------------------------------------
    private void PublishVisualSnapshot(SimState sim, bool rebuild)
    {
        lock (syncLock)
        {
            visRobotPos = sim.robotPos;
            visRobotAngle = sim.robotAngle;
            visTargetPos = sim.targetPos;
            visObstacles = new List<Vector3>(sim.obstacles);
            needRebuildVisuals = needRebuildVisuals || rebuild;
        }
    }

    private void ApplyVisuals()
    {
        List<Vector3> obstaclesCopy;
        bool rebuild;
        Vector2 robot;
        float angle;
        Vector2 target;

        lock (syncLock)
        {
            obstaclesCopy = visObstacles;
            rebuild = needRebuildVisuals;
            needRebuildVisuals = false;
            robot = visRobotPos;
            angle = visRobotAngle;
            target = visTargetPos;
        }

        if (rebuild) RebuildWorldVisuals(obstaclesCopy);

        if (robotVisual != null)
        {
            robotVisual.transform.position = new Vector3(robot.x, robotRadius, robot.y);
            // World CCW angle -> Unity yaw (rotation around +Y is CW from above).
            robotVisual.transform.rotation = Quaternion.Euler(0f, -angle * Mathf.Rad2Deg, 0f);
        }

        if (targetVisual != null)
        {
            targetVisual.transform.position = new Vector3(target.x, targetRadius, target.y);
        }
    }

    private void RebuildWorldVisuals(List<Vector3> obstacleList)
    {
        // Ground + boundary walls + robot + target are created once; obstacles
        // are recreated per reset (domain randomization).
        Transform world = transform.Find("RobotNavWorld");
        if (world == null)
        {
            world = new GameObject("RobotNavWorld").transform;
            world.SetParent(transform, false);

            GameObject ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
            ground.name = "Ground";
            ground.transform.SetParent(world, false);
            ground.transform.localScale = new Vector3(worldSize / 10f, 1f, worldSize / 10f);
            ground.transform.position = Vector3.zero;

            CreateWall(world, "WallNorth", new Vector3(0f, 0.5f, WorldHalf),
                new Vector3(worldSize + 1f, 1f, 0.5f));
            CreateWall(world, "WallSouth", new Vector3(0f, 0.5f, -WorldHalf),
                new Vector3(worldSize + 1f, 1f, 0.5f));
            CreateWall(world, "WallEast", new Vector3(WorldHalf, 0.5f, 0f),
                new Vector3(0.5f, 1f, worldSize + 1f));
            CreateWall(world, "WallWest", new Vector3(-WorldHalf, 0.5f, 0f),
                new Vector3(0.5f, 1f, worldSize + 1f));

            if (robotVisual == null)
            {
                robotVisual = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                robotVisual.name = "Robot";
                robotVisual.transform.SetParent(world, false);
                robotVisual.transform.localScale = new Vector3(
                    robotRadius * 2f, robotRadius * 2f, robotRadius * 2f);
                if (robotMaterial == null)
                {
                    robotMaterial = new Material(Shader.Find("Standard"));
                    robotMaterial.color = new Color(0.2f, 0.5f, 1f);
                }
                robotVisual.GetComponent<Renderer>().material = robotMaterial;
            }

            if (targetVisual == null)
            {
                targetVisual = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                targetVisual.name = "Target";
                targetVisual.transform.SetParent(world, false);
                targetVisual.transform.localScale = new Vector3(
                    targetRadius * 2f, 0.1f, targetRadius * 2f);
                if (targetMaterial == null)
                {
                    targetMaterial = new Material(Shader.Find("Standard"));
                    targetMaterial.color = new Color(0.2f, 0.85f, 0.3f);
                }
                targetVisual.GetComponent<Renderer>().material = targetMaterial;
            }
        }

        // Recreate the randomized obstacles.
        Transform obstacleRoot = world.Find("Obstacles");
        if (obstacleRoot != null) Destroy(obstacleRoot.gameObject);
        obstacleRoot = new GameObject("Obstacles").transform;
        obstacleRoot.SetParent(world, false);

        if (obstacleMaterial == null)
        {
            obstacleMaterial = new Material(Shader.Find("Standard"));
            obstacleMaterial.color = new Color(0.45f, 0.45f, 0.5f);
        }

        foreach (Vector3 obstacle in obstacleList)
        {
            GameObject obstacleGo = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            obstacleGo.name = "Obstacle";
            obstacleGo.transform.SetParent(obstacleRoot, false);
            // Unity cylinder: diameter along X/Z, height along Y.
            obstacleGo.transform.localScale = new Vector3(
                obstacle.z * 2f, 1f, obstacle.z * 2f);
            obstacleGo.transform.position = new Vector3(
                obstacle.x, 0.5f, obstacle.y);
            obstacleGo.GetComponent<Renderer>().material = obstacleMaterial;
        }
    }

    private static void CreateWall(Transform parent, string name, Vector3 position, Vector3 scale)
    {
        GameObject wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
        wall.name = name;
        wall.transform.SetParent(parent, false);
        wall.transform.position = position;
        wall.transform.localScale = scale;
        wall.GetComponent<Renderer>().material.color = new Color(0.6f, 0.6f, 0.65f);
    }

    // ==================================================================
    // Per-connection simulation session (1:1 port of robot_navigation_env_v2)
    // ==================================================================
    private class SimState
    {
        // World config (copied from the bridge component).
        private readonly float worldSize;
        private readonly int maxSteps;
        private readonly int frameSkip;
        private readonly int minObstacles;
        private readonly int maxObstacles;
        private readonly float sensorRange;
        private readonly float robotRadius;
        private readonly float targetRadius;
        private readonly float maxSpeed;
        private readonly float turnAngleDeg;

        // Episode state.
        private System.Random rng;
        internal Vector2 robotPos;
        internal float robotAngle;
        internal Vector2 targetPos;
        internal List<Vector3> obstacles = new List<Vector3>(); // x, y, radius
        private int currentStep;
        private float energyUsed;
        private int stuckCounter;
        private int stopCounter;
        private bool reachedTarget;
        private bool collision;
        private bool stuck;

        internal float WorldHalf { get { return worldSize / 2f; } }
        private float TurnAngleRad { get { return turnAngleDeg * Mathf.Deg2Rad; } }

        internal SimState(RobotNavBridge bridge)
        {
            worldSize = bridge.worldSize;
            maxSteps = bridge.maxSteps;
            frameSkip = bridge.frameSkip;
            minObstacles = bridge.minObstacles;
            maxObstacles = bridge.maxObstacles;
            sensorRange = bridge.sensorRange;
            robotRadius = bridge.robotRadius;
            targetRadius = bridge.targetRadius;
            maxSpeed = bridge.maxSpeed;
            turnAngleDeg = bridge.turnAngleDeg;
        }

        internal void ResetSim(long seed)
        {
            rng = seed >= 0 ? new System.Random((int)seed) : new System.Random();

            currentStep = 0;
            energyUsed = 0f;
            stuckCounter = 0;
            stopCounter = 0;
            reachedTarget = false;
            collision = false;
            stuck = false;

            const float margin = 1f;
            float minTargetDistance = Mathf.Max(4f, worldSize * 0.25f);

            bool placed = false;
            for (int attempt = 0; attempt < 100 && !placed; attempt++)
            {
                Vector2 robot = SamplePoint(margin);
                Vector2 target = SamplePoint(margin);
                while (Vector2.Distance(robot, target) < minTargetDistance)
                {
                    target = SamplePoint(margin);
                }

                List<Vector3> candidate;
                if (!SampleObstacles(robot, target, out candidate)) continue;
                if (!PathExists(robot, target, candidate)) continue;

                robotPos = robot;
                targetPos = target;
                robotAngle = (float)(rng.NextDouble() * 2.0 * Math.PI - Math.PI);
                obstacles = candidate;
                placed = true;
            }

            if (!placed)
            {
                // Fallback: simple map with no obstacles (matches the Python env).
                robotPos = SamplePoint(margin);
                targetPos = SamplePoint(margin);
                robotAngle = (float)(rng.NextDouble() * 2.0 * Math.PI - Math.PI);
                obstacles = new List<Vector3>();
            }
        }

        internal bool StepSim(int action, out float reward, out bool terminated)
        {
            action = Mathf.Clamp(action, 0, 5);
            currentStep += 1;

            Vector2 prevPos = robotPos;
            float prevDistance = Vector2.Distance(robotPos, targetPos);

            reachedTarget = false;
            collision = false;
            stuck = false;

            // Energy costs (v2: 1.0 / 0.85 / 0.85 / 0.25 / 0.25 / 0.05).
            float[] energyCosts = { 1f, 0.85f, 0.85f, 0.25f, 0.25f, 0.05f };
            energyUsed += energyCosts[action];

            // Turning actions (1/2 = half turn, 3/4 = full turn).
            if (action == 1) robotAngle += TurnAngleRad / 2f;
            else if (action == 2) robotAngle -= TurnAngleRad / 2f;
            else if (action == 3) robotAngle += TurnAngleRad;
            else if (action == 4) robotAngle -= TurnAngleRad;
            robotAngle = NormalizeAngle(robotAngle);

            // Moving actions with frame skip; stop at the first collision.
            if (action == 0 || action == 1 || action == 2)
            {
                for (int i = 0; i < frameSkip; i++)
                {
                    MoveForward();
                    if (CheckCollision())
                    {
                        collision = true;
                        break;
                    }
                }
            }

            float currentDistance = Vector2.Distance(robotPos, targetPos);
            float displacement = Vector2.Distance(robotPos, prevPos);

            float progress = prevDistance - currentDistance;
            bool nearTarget = currentDistance < targetRadius * 1.5f;

            reward = 0f;
            reward += progress * 2f;        // progress toward the target
            reward += displacement * 1.5f;  // actual movement
            reward -= 0.03f;                // time penalty

            // Punish stopping when not near the target.
            if (action == 5 && !nearTarget)
            {
                stopCounter += 1;
                reward -= 0.25f;
                reward -= 0.02f * Mathf.Min(stopCounter, 20);
            }
            else
            {
                stopCounter = 0;
            }

            // Persistent stopping counts as stuck.
            if (stopCounter >= 25 && !nearTarget)
            {
                stuck = true;
                reward -= 30f;
            }

            // Target reached.
            if (currentDistance < targetRadius)
            {
                reachedTarget = true;
                reward += 100f;
            }

            // Collision.
            if (collision) reward -= 100f;

            // Anti-stuck detection based on displacement.
            if (displacement < 0.08f && currentDistance > targetRadius * 1.5f)
            {
                stuckCounter += 1;
            }
            else
            {
                stuckCounter = 0;
            }

            if (stuckCounter >= 35)
            {
                stuck = true;
                reward -= 25f;
            }

            // Timeout penalty.
            if (currentStep >= maxSteps && !reachedTarget && !collision)
            {
                reward -= 15f;
                reward -= currentDistance * 0.5f;
            }

            terminated = reachedTarget || collision || stuck;
            bool truncated = currentStep >= maxSteps && !terminated;
            return truncated;
        }

        private void MoveForward()
        {
            float x = robotPos.x + Mathf.Cos(robotAngle) * maxSpeed;
            float y = robotPos.y + Mathf.Sin(robotAngle) * maxSpeed;
            robotPos = new Vector2(
                Mathf.Clamp(x, -WorldHalf, WorldHalf),
                Mathf.Clamp(y, -WorldHalf, WorldHalf));
        }

        private bool CheckCollision()
        {
            foreach (Vector3 obstacle in obstacles)
            {
                Vector2 obstaclePos = new Vector2(obstacle.x, obstacle.y);
                if (Vector2.Distance(robotPos, obstaclePos)
                    < robotRadius + obstacle.z)
                {
                    return true;
                }
            }
            return false;
        }

        private static float NormalizeAngle(float angle)
        {
            while (angle > Mathf.PI) angle -= 2f * Mathf.PI;
            while (angle < -Mathf.PI) angle += 2f * Mathf.PI;
            return angle;
        }

        private Vector2 SamplePoint(float margin)
        {
            float low = -WorldHalf + margin;
            float high = WorldHalf - margin;
            float x = (float)(rng.NextDouble() * (high - low) + low);
            float y = (float)(rng.NextDouble() * (high - low) + low);
            return new Vector2(x, y);
        }

        private bool SampleObstacles(Vector2 robot, Vector2 target, out List<Vector3> result)
        {
            result = new List<Vector3>();
            int count = rng.Next(minObstacles, maxObstacles + 1);
            const float margin = 1f;

            for (int i = 0; i < count; i++)
            {
                bool placedObstacle = false;
                for (int attempt = 0; attempt < 100 && !placedObstacle; attempt++)
                {
                    Vector2 pos = SamplePoint(margin);
                    float radius = (float)(rng.NextDouble() * 0.8 + 0.5); // 0.5 .. 1.3

                    // Keep obstacles away from robot start and target.
                    if (Vector2.Distance(pos, robot) < radius + robotRadius + 1.5f) continue;
                    if (Vector2.Distance(pos, target) < radius + targetRadius + 1.5f) continue;

                    // Avoid overlap with existing obstacles.
                    bool valid = true;
                    foreach (Vector3 existing in result)
                    {
                        if (Vector2.Distance(pos, new Vector2(existing.x, existing.y))
                            < radius + existing.z + 0.5f)
                        {
                            valid = false;
                            break;
                        }
                    }

                    if (valid)
                    {
                        result.Add(new Vector3(pos.x, pos.y, radius));
                        placedObstacle = true;
                    }
                }

                if (!placedObstacle) return false; // whole map rejected, like v2
            }

            return true;
        }

        private bool PathExists(Vector2 robot, Vector2 target, List<Vector3> map)
        {
            const float resolution = 1f;
            int n = Mathf.Max(1, (int)worldSize);
            bool[,] grid = new bool[n, n];

            foreach (Vector3 obstacle in map)
            {
                float inflated = obstacle.z + robotRadius;
                int xMin = Mathf.Max(0, (int)((obstacle.x - inflated + WorldHalf) / resolution));
                int xMax = Mathf.Min(n - 1, (int)((obstacle.x + inflated + WorldHalf) / resolution));
                int yMin = Mathf.Max(0, (int)((obstacle.y - inflated + WorldHalf) / resolution));
                int yMax = Mathf.Min(n - 1, (int)((obstacle.y + inflated + WorldHalf) / resolution));

                for (int y = yMin; y <= yMax; y++)
                {
                    for (int x = xMin; x <= xMax; x++)
                    {
                        float cellX = x * resolution - WorldHalf + resolution / 2f;
                        float cellY = y * resolution - WorldHalf + resolution / 2f;
                        float dx = cellX - obstacle.x;
                        float dy = cellY - obstacle.y;
                        if (dx * dx + dy * dy <= inflated * inflated) grid[y, x] = true;
                    }
                }
            }

            Vector2Int start = ToGrid(robot, n);
            Vector2Int goal = ToGrid(target, n);
            if (grid[start.y, start.x] || grid[goal.y, goal.x]) return false;

            bool[,] visited = new bool[n, n];
            Queue<Vector2Int> queue = new Queue<Vector2Int>();
            queue.Enqueue(start);
            visited[start.y, start.x] = true;

            int[] dxs = { 1, -1, 0, 0, 1, 1, -1, -1 };
            int[] dys = { 0, 0, 1, -1, 1, -1, 1, -1 };

            while (queue.Count > 0)
            {
                Vector2Int cell = queue.Dequeue();
                if (cell.x == goal.x && cell.y == goal.y) return true;

                for (int i = 0; i < 8; i++)
                {
                    int nx = cell.x + dxs[i];
                    int ny = cell.y + dys[i];
                    if (nx < 0 || nx >= n || ny < 0 || ny >= n) continue;
                    if (grid[ny, nx] || visited[ny, nx]) continue;
                    visited[ny, nx] = true;
                    queue.Enqueue(new Vector2Int(nx, ny));
                }
            }

            return false;
        }

        private Vector2Int ToGrid(Vector2 point, int n)
        {
            // Mirrors v2's to_grid(): floor((coord + half) / resolution), clamped.
            int x = Mathf.Clamp((int)((point.x + WorldHalf) / 1f), 0, n - 1);
            int y = Mathf.Clamp((int)((point.y + WorldHalf) / 1f), 0, n - 1);
            return new Vector2Int(x, y);
        }

        // Sector sensors - analytic port of v2 (no Unity Physics).
        private float SectorDistance(float centerRelativeAngle, float spread)
        {
            float minDistance = float.MaxValue;
            for (int i = 0; i < 5; i++)
            {
                float t = i / 4f; // 0..1 inclusive, like np.linspace
                float angle = (centerRelativeAngle - spread / 2f)
                              + t * spread;
                minDistance = Mathf.Min(minDistance, RayDistance(angle));
            }
            return Mathf.Clamp(minDistance / sensorRange, 0f, 1f);
        }

        private float RayDistance(float relativeAngle)
        {
            float angle = robotAngle + relativeAngle;
            Vector2 direction = new Vector2(Mathf.Cos(angle), Mathf.Sin(angle));

            float t = sensorRange;
            const float eps = 1e-8f;

            // Distance to walls.
            float tx = float.MaxValue;
            float ty = float.MaxValue;

            if (direction.x > eps) tx = (WorldHalf - robotPos.x) / direction.x;
            else if (direction.x < -eps) tx = (-WorldHalf - robotPos.x) / direction.x;

            if (direction.y > eps) ty = (WorldHalf - robotPos.y) / direction.y;
            else if (direction.y < -eps) ty = (-WorldHalf - robotPos.y) / direction.y;

            if (tx < 0f) tx = float.MaxValue;
            if (ty < 0f) ty = float.MaxValue;

            t = Mathf.Min(t, Mathf.Min(tx, ty));

            // Distance to obstacles (ray vs inflated circle).
            foreach (Vector3 obstacle in obstacles)
            {
                float inflatedRadius = obstacle.z + robotRadius;
                Vector2 oc = robotPos - new Vector2(obstacle.x, obstacle.y);

                float c = Vector2.Dot(oc, oc) - inflatedRadius * inflatedRadius;
                if (c <= 0f) return 0f; // inside the inflated radius

                float b = 2f * Vector2.Dot(oc, direction);
                float discriminant = b * b - 4f * c;

                if (discriminant >= 0f)
                {
                    float sqrtDiscriminant = Mathf.Sqrt(discriminant);
                    float t1 = (-b - sqrtDiscriminant) / 2f;
                    if (t1 > 0f)
                    {
                        t = Mathf.Min(t, t1);
                    }
                    else
                    {
                        float t2 = (-b + sqrtDiscriminant) / 2f;
                        if (t2 > 0f) t = Mathf.Min(t, t2);
                    }
                }
            }

            return Mathf.Clamp(t, 0f, sensorRange);
        }

        // Observation - identical layout and normalization to v2.
        internal float[] GetObs()
        {
            float distanceToTarget = Vector2.Distance(robotPos, targetPos);
            float maxDistance = Mathf.Sqrt(2f) * worldSize;
            float sectorWidth = 2f * Mathf.PI / 3f; // 120 degrees

            float angleToTarget = NormalizeAngle(
                Mathf.Atan2(targetPos.y - robotPos.y, targetPos.x - robotPos.x)
                - robotAngle);

            float front = SectorDistance(0f, sectorWidth);
            float left = SectorDistance(2f * Mathf.PI / 3f, sectorWidth);
            float right = SectorDistance(-2f * Mathf.PI / 3f, sectorWidth);

            float[] obs =
            {
                Clamp(robotPos.x / WorldHalf),
                Clamp(robotPos.y / WorldHalf),
                Clamp(robotAngle / Mathf.PI),
                Clamp(targetPos.x / WorldHalf),
                Clamp(targetPos.y / WorldHalf),
                Clamp(distanceToTarget / maxDistance),
                Clamp(angleToTarget / Mathf.PI),
                Clamp(front),
                Clamp(left),
                Clamp(right),
            };
            return obs;
        }

        private static float Clamp(float v)
        {
            return Mathf.Clamp(v, -1f, 1f);
        }

        internal StateMsg BuildState(float reward, bool terminated, bool truncated)
        {
            ObstacleDto[] obstacleDtos = new ObstacleDto[obstacles.Count];
            for (int i = 0; i < obstacles.Count; i++)
            {
                obstacleDtos[i] = new ObstacleDto
                {
                    x = obstacles[i].x,
                    y = obstacles[i].y,
                    radius = obstacles[i].z,
                };
            }

            return new StateMsg
            {
                type = "state",
                obs = GetObs(),
                reward = reward,
                terminated = terminated,
                truncated = truncated,
                info = new InfoDto
                {
                    distance_to_target = Vector2.Distance(robotPos, targetPos),
                    step = currentStep,
                    reached_target = reachedTarget,
                    collision = collision,
                    stuck = stuck,
                    energy_used = energyUsed,
                },
                world_size = worldSize,
                robot_pos = new[] { robotPos.x, robotPos.y },
                robot_angle = robotAngle,
                target_pos = new[] { targetPos.x, targetPos.y },
                obstacles = obstacleDtos,
            };
        }
    }
}

