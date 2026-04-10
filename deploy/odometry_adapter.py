"""ROS2 node: /odom → forward_velocity + yaw_rate observation (indices 24:26).

Extracts body-frame linear.x and angular.z from nav_msgs/Odometry,
applies VecNormalize normalization, and publishes on /policy/vel_obs
(Float32MultiArray, 2 elements) at 12 Hz.

Usage:
    ros2 run nova_carter_deploy odometry_adapter \
        --ros-args -p vecnorm_path:=/path/to/model_vecnormalize.npz
"""

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray

PUBLISH_HZ = 12.0
VEL_OBS_INDICES = slice(24, 26)   # forward_vel, yaw_rate are obs[24:26]


class OdometryAdapter(Node):
    def __init__(self):
        super().__init__("odometry_adapter")

        self.declare_parameter("vecnorm_path", "deploy/best_model_building_stageC_vecnorm.npz")
        self.declare_parameter("odom_topic", "/odom")

        vecnorm_path = self.get_parameter("vecnorm_path").value
        odom_topic   = self.get_parameter("odom_topic").value

        data = np.load(vecnorm_path)
        self.obs_mean = data["obs_mean"][VEL_OBS_INDICES].astype(np.float32)
        self.obs_var  = data["obs_var"][VEL_OBS_INDICES].astype(np.float32)
        self.clip_obs = float(data["clip_obs"])
        self.get_logger().info(
            f"VecNorm loaded (vel indices 24:26): mean={self.obs_mean}, clip={self.clip_obs}"
        )

        self._latest_vel = np.zeros(2, dtype=np.float32)
        self.sub = self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        self.pub = self.create_publisher(Float32MultiArray, "/policy/vel_obs", 10)
        self.create_timer(1.0 / PUBLISH_HZ, self._publish_cb)
        self.get_logger().info(f"OdometryAdapter ready. Topic={odom_topic}")

    def _odom_cb(self, msg: Odometry):
        # /odom twist is already in body frame for differential drive robots
        forward_vel = msg.twist.twist.linear.x
        yaw_rate    = msg.twist.twist.angular.z
        self._latest_vel = np.array([forward_vel, yaw_rate], dtype=np.float32)

    def _publish_cb(self):
        vel_norm = np.clip(
            (self._latest_vel - self.obs_mean) / np.sqrt(self.obs_var + 1e-8),
            -self.clip_obs, self.clip_obs,
        )
        msg = Float32MultiArray()
        msg.data = vel_norm.tolist()
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = OdometryAdapter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
