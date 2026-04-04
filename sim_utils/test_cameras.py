"""
Shared test camera definitions and helpers for the validation pipeline.

Provides:
  - TEST_CAMERAS: verified security-cam positions for the Innovation Center
  - create_test_camera(): create an Isaac Sim Camera at a test position
  - add_interior_lighting(): add DistantLight + DomeLight + SphereLight per camera
  - look_at_orientation(): compute quaternion from position to target
"""

import math
import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils

from pxr import Sdf, UsdLux, UsdGeom


# ---------------------------------------------------------------------------
# Test Camera Positions (Innovation Center — feet)
# ---------------------------------------------------------------------------

TEST_CAMERAS = {
    "G": {
        "pos": (-42.15, 68.44, 16.44),
        "target": (-27.42, 83.72, 14.56),
        "room": "Event Floor IC 100",
    },
    "1": {
        "pos": (-56.03, 62.23, 17.50),
        "target": (-49.19, 73.19, 14.56),
        "room": "Robotics Lab IC 101",
    },
    "2": {
        "pos": (-56.28, 60.88, 24.70),
        "target": (-49.19, 72.96, 23.50),
        "room": "Classroom 40 IC 302",
    },
    "3": {
        "pos": (-41.51, 67.62, 29.65),
        "target": (-30.23, 81.20, 27.77),
        "room": "Capstone Studio IC 405",
    },
}

# Camera hardware defaults — matches floor_camera_screenshots.py reference images
CAMERA_FOCAL_LENGTH = 12.0      # mm, wide-angle (same as reference screenshots)
CAMERA_RESOLUTION = (1920, 1080)
CAMERA_FREQUENCY = 30           # Hz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def look_at_orientation(cam_pos, target):
    """Compute a quaternion orientation looking from cam_pos toward target."""
    dx = target[0] - cam_pos[0]
    dy = target[1] - cam_pos[1]
    dz = target[2] - cam_pos[2]
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(-dz, math.sqrt(dx**2 + dy**2)))
    return rot_utils.euler_angles_to_quats(
        np.array([0, pitch, yaw]), degrees=True
    )


def create_test_camera(cam_key, cam_dict, prim_prefix="/World/TestCam", stage=None):
    """Create an Isaac Sim Camera sensor at a test camera position.

    The focal length is set directly on the USD prim (matching the
    approach in floor_camera_screenshots.py that produced the reference
    images).  If *stage* is provided, the prim attribute is written
    immediately; otherwise call ``set_focal_length_on_prim()`` after
    the camera is created.

    Args:
        cam_key: Camera key (e.g. "G", "1")
        cam_dict: Dict with "pos" and "target" tuples
        prim_prefix: Prim path prefix
        stage: USD stage (pass to set focal length on prim immediately)

    Returns:
        isaacsim.sensors.camera.Camera instance (call .initialize() after world.reset())
    """
    from isaacsim.sensors.camera import Camera

    pos = np.array(cam_dict["pos"])
    target = np.array(cam_dict["target"])
    orient = look_at_orientation(pos, target)

    prim_path = f"{prim_prefix}_{cam_key}"
    camera = Camera(
        prim_path=prim_path,
        position=pos,
        frequency=CAMERA_FREQUENCY,
        resolution=CAMERA_RESOLUTION,
        orientation=orient,
    )

    # Set focal length directly on the USD prim — same method as the
    # reference floor_camera_screenshots.py (line 620).  The Python API
    # set_focal_length() can be overridden during initialize().
    if stage is not None:
        cam_prim = stage.GetPrimAtPath(prim_path)
        UsdGeom.Camera(cam_prim).CreateFocalLengthAttr(CAMERA_FOCAL_LENGTH)

    return camera


def set_focal_length_on_prim(stage, camera):
    """Set the focal length on a camera's USD prim after creation.

    Call this after world.reset() + camera.initialize() to ensure
    the focal length matches the reference screenshots.
    """
    cam_prim = stage.GetPrimAtPath(camera.prim_path)
    UsdGeom.Camera(cam_prim).CreateFocalLengthAttr(CAMERA_FOCAL_LENGTH)


def add_interior_lighting(stage):
    """Add lighting suitable for interior building scenes.

    Enclosed rooms need strong fill light.  Uses a combination of:
      - DistantLight (sun) for natural shadows
      - DomeLight (ambient fill) for overall brightness
      - SphereLight at each camera position for local fill
      - RectLight at each camera target (floor center) for even coverage
    """
    distant = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Val_Sun"))
    distant.CreateIntensityAttr(15000)
    distant.CreateAngleAttr(1.0)
    xform = UsdGeom.Xformable(distant.GetPrim())
    xform.AddRotateXYZOp().Set((-45, 30, 0))

    UsdLux.DomeLight.Define(
        stage, Sdf.Path("/World/Val_DomeLight")
    ).CreateIntensityAttr(8000)

    for key, cam in TEST_CAMERAS.items():
        # Sphere light at camera position (overhead fill)
        light = UsdLux.SphereLight.Define(
            stage, Sdf.Path(f"/World/Val_CamLight_{key}")
        )
        light.CreateIntensityAttr(80000)
        light.CreateRadiusAttr(1.0)
        UsdGeom.Xformable(light.GetPrim()).AddTranslateOp().Set(
            tuple(cam["pos"])
        )

        # Rect light at target center (floor fill, facing down)
        rect = UsdLux.RectLight.Define(
            stage, Sdf.Path(f"/World/Val_FloorLight_{key}")
        )
        rect.CreateIntensityAttr(50000)
        rect.CreateWidthAttr(15.0)
        rect.CreateHeightAttr(15.0)
        rx = UsdGeom.Xformable(rect.GetPrim())
        tx, ty, tz = cam["target"]
        rx.AddTranslateOp().Set((tx, ty, tz + 4.0))
        rx.AddRotateXYZOp().Set((90, 0, 0))
