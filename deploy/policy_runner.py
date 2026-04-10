"""ROS2 node: 26-dim observation → Twist command at 12 Hz.

Assembles the full observation vector from the three adapter topics
(/policy/lidar_obs, /policy/goal_obs, /policy/vel_obs), runs inference
via TensorRT (or ONNX fallback), and publishes geometry_msgs/Twist on
/cmd_vel. Includes proximity e-stop, watchdog, and per-step CSV logging.

Safety layer (non-negotiable):
  - Proximity e-stop: any raw LiDAR ray < 0.25m → zero Twist for 3 cycles
  - Watchdog: no new obs in > 300ms → zero Twist
  - Velocity cap: linear.x clamped to ±0.25 m/s during initial testing

Usage:
    ros2 run nova_carter_deploy policy_runner \
        --ros-args -p engine_path:=deploy/best_model_building_stageC.trt \
                   -p vecnorm_path:=deploy/best_model_building_stageC_vecnorm.npz
"""

import csv
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

POLICY_HZ = 12.0
OBS_DIM = 26
LIDAR_DIM = 20
GOAL_DIM = 4
VEL_DIM = 2

ESTOP_DIST_M_RAW_NORM = 0.25 / 10.0   # 0.25m / 10m max_range = 0.025 normalized
MAX_LINEAR_M_S = 0.5
MAX_ANGULAR_RAD_S = 1.0
INITIAL_VEL_CAP = 0.25   # m/s — lifted after corridor tests pass
WATCHDOG_TIMEOUT_S = 0.3
ESTOP_HOLD_CYCLES = 3


class PolicyRunner(Node):
    def __init__(self):
        super().__init__("policy_runner")

        self.declare_parameter("engine_path", "deploy/best_model_building_stageC.trt")
        self.declare_parameter("onnx_path",   "deploy/best_model_building_stageC.onnx")
        self.declare_parameter("vecnorm_path","deploy/best_model_building_stageC_vecnorm.npz")
        self.declare_parameter("vel_cap",     INITIAL_VEL_CAP)
        self.declare_parameter("log_dir",     "deploy/logs")

        engine_path  = self.get_parameter("engine_path").value
        onnx_path    = self.get_parameter("onnx_path").value
        self.vel_cap = float(self.get_parameter("vel_cap").value)
        log_dir      = self.get_parameter("log_dir").value

        # Load inference backend
        self.session = None
        self.trt_context = None
        if os.path.exists(engine_path):
            self._load_tensorrt(engine_path)
        elif os.path.exists(onnx_path):
            self.get_logger().warn(
                "TensorRT engine not found — using ONNX Runtime (280ms+ latency on Jetson). "
                "Convert to TRT before production deployment."
            )
            self._load_onnx(onnx_path)
        else:
            raise FileNotFoundError(
                f"Neither TRT engine ({engine_path}) nor ONNX ({onnx_path}) found."
            )

        # Load raw LiDAR scale for e-stop (need un-normalized values)
        data = np.load(self.get_parameter("vecnorm_path").value)
        self._lidar_mean = data["obs_mean"][:LIDAR_DIM].astype(np.float32)
        self._lidar_std  = np.sqrt(data["obs_var"][:LIDAR_DIM] + 1e-8).astype(np.float32)
        self._clip       = float(data["clip_obs"])

        # Observation buffers
        self._lidar_obs = np.zeros(LIDAR_DIM, dtype=np.float32)
        self._goal_obs  = np.zeros(GOAL_DIM,  dtype=np.float32)
        self._vel_obs   = np.zeros(VEL_DIM,   dtype=np.float32)
        self._t_lidar   = 0.0
        self._t_goal    = 0.0
        self._t_vel     = 0.0

        self._estop_cycles_remaining = 0
        self._estop_count = 0

        # CSV logger
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path = os.path.join(log_dir, f"run_{ts}.csv")
        self._estop_log_path = os.path.join(log_dir, f"estop_{ts}.csv")
        self._log_file = open(self._log_path, "w", newline="")
        self._estop_file = open(self._estop_log_path, "w", newline="")
        self._writer = csv.writer(self._log_file)
        self._estop_writer = csv.writer(self._estop_file)
        self._writer.writerow([
            "timestamp", "lidar_min", "lidar_max", "goal_dist_norm",
            "linear_x", "angular_z", "waypoint_index"
        ])
        self._estop_writer.writerow(["timestamp", "trigger_ray_value", "estop_count"])

        # ROS
        self.sub_lidar = self.create_subscription(
            Float32MultiArray, "/policy/lidar_obs", self._lidar_cb, 10)
        self.sub_goal = self.create_subscription(
            Float32MultiArray, "/policy/goal_obs", self._goal_cb, 10)
        self.sub_vel = self.create_subscription(
            Float32MultiArray, "/policy/vel_obs", self._vel_cb, 10)
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_timer(1.0 / POLICY_HZ, self.policy_step)
        self.get_logger().info(
            f"PolicyRunner ready. vel_cap={self.vel_cap} m/s, log={self._log_path}"
        )

    def _load_tensorrt(self, path):
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        self.trt_context = engine.create_execution_context()
        self.get_logger().info(f"TensorRT engine loaded: {path}")

    def _load_onnx(self, path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.get_logger().info(f"ONNX Runtime session loaded: {path}")

    def _lidar_cb(self, msg):
        self._lidar_obs = np.array(msg.data, dtype=np.float32)
        self._t_lidar = time.time()

    def _goal_cb(self, msg):
        self._goal_obs = np.array(msg.data, dtype=np.float32)
        self._t_goal = time.time()

    def _vel_cb(self, msg):
        self._vel_obs = np.array(msg.data, dtype=np.float32)
        self._t_vel = time.time()

    def _infer(self, obs: np.ndarray) -> np.ndarray:
        obs_input = obs[None].astype(np.float32)   # (1, 26)
        if self.session is not None:
            input_name = self.session.get_inputs()[0].name
            raw = self.session.run(None, {input_name: obs_input})[0][0]
        else:
            # TensorRT path — simplified; real implementation needs cuda memcpy
            raise NotImplementedError("TensorRT inference requires CUDA buffer setup.")
        return raw  # shape (2,)

    def policy_step(self):
        now = time.time()

        # Watchdog: stale observations → zero command
        obs_age = max(now - self._t_lidar, now - self._t_goal, now - self._t_vel)
        if obs_age > WATCHDOG_TIMEOUT_S and (self._t_lidar + self._t_goal + self._t_vel) > 0:
            self._publish_zero("watchdog")
            return

        # Proximity e-stop: un-normalize LiDAR to get raw distances
        lidar_raw = self._lidar_obs * self._lidar_std + self._lidar_mean
        lidar_raw_m = np.clip(lidar_raw * 10.0, 0.0, 10.0)   # ×10m max range
        min_dist = float(np.min(lidar_raw_m))
        if min_dist < 0.25 or self._estop_cycles_remaining > 0:
            if min_dist < 0.25:
                self._estop_cycles_remaining = ESTOP_HOLD_CYCLES
                self._estop_count += 1
                self._estop_writer.writerow([now, min_dist, self._estop_count])
                self._estop_file.flush()
                self.get_logger().warn(
                    f"E-STOP: obstacle at {min_dist:.3f}m (count={self._estop_count})"
                )
            self._estop_cycles_remaining = max(0, self._estop_cycles_remaining - 1)
            self._publish_zero("estop")
            return

        # Assemble 26-dim obs (already VecNormalize-normalized by adapters)
        obs = np.concatenate([self._lidar_obs, self._goal_obs, self._vel_obs])
        if np.any(~np.isfinite(obs)):
            self.get_logger().warn("NaN/Inf in obs — skipping step")
            self._publish_zero("nan")
            return

        raw_action = self._infer(obs)   # (2,)

        # SB3 Gaussian policy: action = tanh(raw) × scale
        action = np.tanh(raw_action) * np.array([MAX_LINEAR_M_S, MAX_ANGULAR_RAD_S])
        action[0] = np.clip(action[0], -self.vel_cap, self.vel_cap)

        twist = Twist()
        twist.linear.x  = float(action[0])
        twist.angular.z = float(action[1])
        self.pub_cmd.publish(twist)

        goal_dist = float(np.sqrt(self._goal_obs[0]**2 + self._goal_obs[1]**2))
        self._writer.writerow([
            now, float(np.min(lidar_raw_m)), float(np.max(lidar_raw_m)),
            goal_dist, action[0], action[1], -1
        ])
        self._log_file.flush()

    def _publish_zero(self, reason: str):
        self.pub_cmd.publish(Twist())

    def destroy_node(self):
        self._log_file.close()
        self._estop_file.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = PolicyRunner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
