"""
Follow-camera utility for Isaac Sim simulations.

Wraps isaacsim.sensors.camera.Camera with automatic pose tracking:
the camera stays at a configurable offset behind/above a moving object
and always rotates to keep the target centered in frame.

Usage:
    from sim_utils.follow_camera import FollowCamera

    camera = FollowCamera(
        prim_path="/World/follow_camera",
        local_offset=[-6.0, 0.0, 2.0],   # 6m behind, 2m up in target-local frame
        resolution=(640, 480),
        fps=20,
        z_clamp=(0.5, 2.5),              # keep camera inside corridor height
    )

    my_world.reset()
    simulation_app.update()
    camera.initialize()

    while simulating:
        # Call update() BEFORE world.step(render=True)
        if world.is_playing():
            target_pos, target_quat = robot.get_world_pose()
            camera.update(target_pos, target_quat)

        world.step(render=True)

        if world.is_playing():
            recorder.capture_rgba(camera.get_rgba())
"""

import numpy as np


class FollowCamera:
    """Camera that follows a moving object, always pointing at it.

    The camera is placed at `local_offset` relative to the tracked object's
    frame (i.e. the offset is rotated by the object's orientation each frame),
    then its orientation is recomputed so it always faces the object.

    Args:
        prim_path:    USD prim path for the camera (e.g. "/World/follow_camera")
        local_offset: [x, y, z] offset in the *target's local frame*.
                      Negative x = behind the target (most common).
                      E.g. [-6, 0, 2] → 6 m behind, 2 m above.
        resolution:   (width, height) in pixels.
        fps:          Camera sensor frequency (Hz).
        z_clamp:      (z_min, z_max) world-space z limits to prevent the camera
                      from clipping into floors or ceilings.
        look_height:  How high above the target's origin to aim (metres).
                      Default 0.3 m ≈ centre of a typical ground robot.
    """

    def __init__(
        self,
        prim_path: str,
        local_offset,
        resolution=(640, 480),
        fps: int = 20,
        z_clamp=(0.1, float("inf")),
        look_height: float = 0.3,
    ):
        import isaacsim.core.utils.numpy.rotations as rot_utils
        from isaacsim.sensors.camera import Camera

        self._offset = np.array(local_offset, dtype=float)
        self._z_min, self._z_max = z_clamp
        self._look_height = look_height
        self._rot_utils = rot_utils

        # Initial orientation: roughly level, facing +X (will be overwritten on
        # first update() call before any frame is rendered)
        initial_orient = rot_utils.euler_angles_to_quats(
            np.array([0.0, 15.0, 0.0]), degrees=True
        )

        self._camera = Camera(
            prim_path=prim_path,
            position=self._offset.copy(),
            frequency=fps,
            resolution=resolution,
            orientation=initial_orient,
            annotator_device="cpu",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self):
        """Call once after world.reset() + simulation_app.update()."""
        self._camera.initialize()

    def set_local_offset(self, offset):
        """Update the camera's target-local offset (for cinematic zooms)."""
        self._offset = np.array(offset, dtype=float)

    def update(self, target_pos, target_quat):
        """Reposition and reorient the camera to follow target_pos/target_quat.

        Must be called BEFORE world.step(render=True) each frame so the render
        pipeline picks up the new pose.

        Args:
            target_pos:  np.array([x, y, z]) world position of the tracked object.
            target_quat: np.array([w, x, y, z]) world orientation quaternion.
        """
        cam_pos, cam_orient = self._compute_pose(target_pos, target_quat)
        self._camera.set_world_pose(position=cam_pos, orientation=cam_orient)

    def get_rgba(self):
        """Return latest RGBA frame as np.array (H, W, 4), or None if not ready."""
        return self._camera.get_rgba()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_pose(self, target_pos, target_quat):
        """Return (cam_pos, cam_orient) for this frame."""
        # --- Position: rotate local offset into world frame ---
        cam_pos = target_pos + self._quat_rotate(target_quat, self._offset)
        cam_pos = cam_pos.copy()
        cam_pos[2] = np.clip(cam_pos[2], self._z_min, self._z_max)

        # --- Orientation: yaw/pitch to look at target centre ---
        look_target = target_pos + np.array([0.0, 0.0, self._look_height])
        fwd = look_target - cam_pos
        norm = np.linalg.norm(fwd)
        if norm < 1e-6:
            # Degenerate case: camera is on top of target; keep current orient
            return cam_pos, self._camera.get_world_pose()[1]
        fwd /= norm

        yaw   = np.degrees(np.arctan2(fwd[1], fwd[0]))
        pitch = np.degrees(np.arctan2(-fwd[2], np.sqrt(fwd[0]**2 + fwd[1]**2)))
        orient = self._rot_utils.euler_angles_to_quats(
            np.array([0.0, pitch, yaw]), degrees=True
        )
        return cam_pos, orient

    @staticmethod
    def _quat_rotate(q, v):
        """Rotate vector v by unit quaternion q (wxyz format)."""
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ])
        return R @ v
