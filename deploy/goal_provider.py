"""ROS2 node: building_waypoints.json → body-frame goal observation (indices 20:24).

Loads waypoints from building_waypoints.json, exposes a service to set the
active waypoint index, subscribes to robot pose, computes the 4-element
body-frame goal obs [dx_b, dy_b, cos(heading_err), sin(heading_err)], and
applies VecNormalize normalization. Publishes on /policy/goal_obs at 12 Hz.
Auto-advances to the next waypoint when within goal_reached_threshold (1.0m).

Usage:
    ros2 run nova_carter_deploy goal_provider \
        --ros-args -p vecnorm_path:=/path/to/model_vecnormalize.npz \
                   -p waypoints_path:=/path/to/building_waypoints.json \
                   -p floor:=G
"""

import json
import math

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, Int32
from std_srvs.srv import SetBool

PUBLISH_HZ = 12.0
GOAL_OBS_INDICES = slice(20, 24)   # goal obs are indices 20:24


class GoalProvider(Node):
    def __init__(self):
        super().__init__("goal_provider")

        self.declare_parameter("vecnorm_path", "deploy/best_model_building_stageC_vecnorm.npz")
        self.declare_parameter("waypoints_path", "training/nova_carter_patrol/building_waypoints.json")
        self.declare_parameter("floor", "G")
        self.declare_parameter("pose_topic", "/visual_slam/tracking/odometry")
        self.declare_parameter("goal_reached_threshold", 1.0)

        vecnorm_path     = self.get_parameter("vecnorm_path").value
        waypoints_path   = self.get_parameter("waypoints_path").value
        floor            = self.get_parameter("floor").value
        pose_topic       = self.get_parameter("pose_topic").value
        self.threshold   = self.get_parameter("goal_reached_threshold").value

        # Load VecNormalize stats
        data = np.load(vecnorm_path)
        self.obs_mean = data["obs_mean"][GOAL_OBS_INDICES].astype(np.float32)
        self.obs_var  = data["obs_var"][GOAL_OBS_INDICES].astype(np.float32)
        self.clip_obs = float(data["clip_obs"])

        # Load waypoints for selected floor
        with open(waypoints_path) as f:
            wp_data = json.load(f)
        self.waypoints = [
            (float(w["x"]), float(w["y"]), w.get("label", f"wp{i}"))
            for i, w in enumerate(wp_data.get(floor, []))
        ]
        if not self.waypoints:
            raise ValueError(f"No waypoints found for floor '{floor}' in {waypoints_path}")
        self.get_logger().info(
            f"Loaded {len(self.waypoints)} waypoints for floor {floor}"
        )

        self.wp_index = 0
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._robot_yaw = 0.0
        self._pose_received = False

        # Subscribers
        self.sub_odom = self.create_subscription(Odometry, pose_topic, self._pose_cb, 10)
        # Fallback: AMCL pose
        self.sub_amcl = self.create_subscription(
            PoseWithCovarianceStamped, "/amcl_pose", self._amcl_cb, 10
        )

        # Publishers
        self.pub_goal = self.create_publisher(Float32MultiArray, "/policy/goal_obs", 10)
        self.pub_wp_index = self.create_publisher(Int32, "/goal_provider/waypoint_index", 10)

        self.create_timer(1.0 / PUBLISH_HZ, self._publish_cb)
        self.get_logger().info(
            f"GoalProvider ready. Threshold={self.threshold}m, "
            f"starting at waypoint 0: {self.waypoints[0]}"
        )

    def _pose_cb(self, msg: Odometry):
        self._robot_x   = msg.pose.pose.position.x
        self._robot_y   = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        # Extract yaw from quaternion
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        self._pose_received = True

    def _amcl_cb(self, msg: PoseWithCovarianceStamped):
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        self._pose_received = True

    def _publish_cb(self):
        if not self._pose_received or not self.waypoints:
            return

        goal_x, goal_y, label = self.waypoints[self.wp_index]
        dx_w = goal_x - self._robot_x
        dy_w = goal_y - self._robot_y
        dist = math.hypot(dx_w, dy_w)

        # Auto-advance waypoint
        if dist < self.threshold:
            self.wp_index = (self.wp_index + 1) % len(self.waypoints)
            goal_x, goal_y, label = self.waypoints[self.wp_index]
            dx_w = goal_x - self._robot_x
            dy_w = goal_y - self._robot_y
            self.get_logger().info(
                f"Goal reached! → waypoint {self.wp_index}: {label} "
                f"({goal_x:.1f}, {goal_y:.1f})"
            )

        # Transform to body frame
        c, s = math.cos(self._robot_yaw), math.sin(self._robot_yaw)
        dx_b =  c * dx_w + s * dy_w
        dy_b = -s * dx_w + c * dy_w
        heading_err = math.atan2(dy_b, dx_b)
        goal_raw = np.array(
            [dx_b, dy_b, math.cos(heading_err), math.sin(heading_err)],
            dtype=np.float32,
        )

        goal_norm = np.clip(
            (goal_raw - self.obs_mean) / np.sqrt(self.obs_var + 1e-8),
            -self.clip_obs, self.clip_obs,
        )

        msg = Float32MultiArray()
        msg.data = goal_norm.tolist()
        self.pub_goal.publish(msg)

        idx_msg = Int32()
        idx_msg.data = self.wp_index
        self.pub_wp_index.publish(idx_msg)


def main():
    rclpy.init()
    node = GoalProvider()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
