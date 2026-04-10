"""ROS2 node: Hesai XT-32 PointCloud2 → 20-ray normalized LiDAR observation.

Extracts a single horizontal ring from the XT-32 3D scan at the elevation
angle corresponding to 0.3m above ground (matching the simulated LiDAR
height), resamples to 20 rays at 18° spacing, applies Gaussian noise
(σ=0.02m, matching training DR level), and applies VecNormalize.

Publishes on /policy/lidar_obs (Float32MultiArray, 20 elements) at 12 Hz.

Usage:
    ros2 run nova_carter_deploy lidar_adapter \
        --ros-args -p vecnorm_path:=/path/to/model_vecnormalize.npz \
                   -p ring_index:=16
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray
import sensor_msgs_py.point_cloud2 as pc2

TARGET_ANGLES = np.deg2rad(np.arange(0, 360, 18))  # 20 rays at 18° spacing
MAX_RANGE_M = 10.0
NOISE_STD_M = 0.02   # slightly lower than training (0.05m) — real noise is additive to physical
PUBLISH_HZ = 12.0


class LidarAdapter(Node):
    def __init__(self):
        super().__init__("lidar_adapter")

        self.declare_parameter("vecnorm_path", "deploy/best_model_building_stageC_vecnorm.npz")
        self.declare_parameter("ring_index", 16)   # XT-32 ring at ~0° elevation; verify physically
        self.declare_parameter("lidar_topic", "/lidar_0/point_cloud")

        vecnorm_path = self.get_parameter("vecnorm_path").value
        self.ring_index = self.get_parameter("ring_index").value
        lidar_topic = self.get_parameter("lidar_topic").value

        data = np.load(vecnorm_path)
        self.obs_mean = data["obs_mean"][:20].astype(np.float32)
        self.obs_var  = data["obs_var"][:20].astype(np.float32)
        self.clip_obs = float(data["clip_obs"])
        self.get_logger().info(
            f"VecNorm loaded: mean[:3]={self.obs_mean[:3]}, clip={self.clip_obs}"
        )

        self.sub = self.create_subscription(
            PointCloud2, lidar_topic, self._cloud_cb, 10
        )
        self.pub = self.create_publisher(Float32MultiArray, "/policy/lidar_obs", 10)

        self._latest_scan = None
        self.create_timer(1.0 / PUBLISH_HZ, self._publish_cb)
        self.get_logger().info(
            f"LidarAdapter ready. Ring={self.ring_index}, topic={lidar_topic}"
        )

    def _cloud_cb(self, msg: PointCloud2):
        """Extract selected ring from XT-32 PointCloud2 and resample to 20 rays."""
        try:
            points = np.array(
                list(pc2.read_points(msg, field_names=("x", "y", "z", "ring"), skip_nans=True))
            )
        except Exception as e:
            self.get_logger().warn(f"PointCloud2 parse error: {e}")
            return

        if points.shape[0] == 0:
            return

        # Filter to the identified ring channel
        ring_mask = points[:, 3].astype(int) == self.ring_index
        ring_pts = points[ring_mask, :3]  # (N, 3) [x, y, z]

        if ring_pts.shape[0] < 5:
            self.get_logger().warn(f"Ring {self.ring_index}: only {ring_pts.shape[0]} points")
            return

        self._latest_scan = self._resample_to_20_rays(ring_pts)

    def _resample_to_20_rays(self, ring_points: np.ndarray) -> np.ndarray:
        """Resample N ring points → 20 rays at 18° spacing using nearest-angle min-range."""
        azimuths = np.arctan2(ring_points[:, 1], ring_points[:, 0])
        distances = np.hypot(ring_points[:, 0], ring_points[:, 1])

        scan_20 = np.full(20, MAX_RANGE_M, dtype=np.float32)
        for i, target in enumerate(TARGET_ANGLES):
            angle_diff = np.abs(
                np.arctan2(np.sin(azimuths - target), np.cos(azimuths - target))
            )
            candidates = distances[angle_diff < np.deg2rad(9)]
            if len(candidates) > 0:
                scan_20[i] = min(float(np.min(candidates)), MAX_RANGE_M)

        # Add noise to match training distribution (σ=0.02m)
        scan_20 += np.random.normal(0, NOISE_STD_M, 20).astype(np.float32)
        scan_20 = np.clip(scan_20, 0.0, MAX_RANGE_M)
        return scan_20

    def _publish_cb(self):
        if self._latest_scan is None:
            return

        # Apply VecNormalize
        scan_norm = np.clip(
            (self._latest_scan - self.obs_mean) / np.sqrt(self.obs_var + 1e-8),
            -self.clip_obs, self.clip_obs,
        )

        msg = Float32MultiArray()
        msg.data = scan_norm.tolist()
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = LidarAdapter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
