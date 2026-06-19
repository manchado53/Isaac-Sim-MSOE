"""
Shared test camera definitions and helpers for the validation pipeline.

Provides:
  - discover_test_cameras(stage): building-agnostic, auto-places a camera
    above each floor level of any physics USD (meters)
  - create_test_camera(): create an Isaac Sim Camera at a test position
  - add_interior_lighting(): add DistantLight + DomeLight + SphereLight per camera
  - look_at_orientation(): compute quaternion from position to target

TEST_CAMERAS (deprecated): legacy hardcoded Innovation Center security-cam
positions in feet. Kept only as a fallback so old scripts that import the
name don't crash; new code should call discover_test_cameras(stage).
"""

import math
import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils

from pxr import Sdf, Usd, UsdLux, UsdGeom


# ---------------------------------------------------------------------------
# Legacy hardcoded positions (Innovation Center raw USD, feet units).
# All current physics USDs are in meters — prefer discover_test_cameras().
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


def add_camera_headlight(stage, camera_prim_path="/OmniverseKit_Persp",
                         intensity=80000.0, radius=0.5,
                         color_temperature=5500.0,
                         fallback_dome_intensity=150.0):
    """Parent a SphereLight under a camera prim so it tracks the viewport.

    This is the "camera light" / "headlight" pattern — wherever the user
    navigates the active camera, the light follows and illuminates the
    view. Replaces the dense position-locked lighting of
    ``add_interior_lighting``, which only covered the four named test
    cameras and left the rest of the building dark.

    Also adds a dim DomeLight so geometry outside the headlight's reach
    stays readable (otherwise long corridors beyond a few meters from
    the camera turn pitch-black).

    Args:
        stage: the USD stage.
        camera_prim_path: prim path to attach the headlight under. Default
            is ``/OmniverseKit_Persp`` — the default viewport camera in
            Kit's full streaming experience. If the camera prim doesn't
            exist yet this function defines it.
        intensity: headlight intensity. 80k is tuned for interiors at
            ~3-10 m viewing distance; interior-at-distance (long
            corridors) reads cleanly without blowing out close-up walls.
        radius: headlight sphere radius in stage units.
        color_temperature: headlight color (Kelvin). 5500 ≈ neutral
            fluorescent/daylight; 4200 is warmer office lighting.
        fallback_dome_intensity: pass 0 to disable the ambient dome.
    """
    UsdGeom.Camera.Define(stage, camera_prim_path)
    headlight = UsdLux.SphereLight.Define(
        stage, Sdf.Path(f"{camera_prim_path}/Headlight"))
    headlight.CreateIntensityAttr(intensity)
    headlight.CreateRadiusAttr(radius)
    headlight.CreateColorTemperatureAttr(color_temperature)
    headlight.CreateEnableColorTemperatureAttr(True)

    if fallback_dome_intensity > 0:
        dome = UsdLux.DomeLight.Define(
            stage, Sdf.Path("/World/CameraHeadlight_FallbackDome"))
        dome.CreateIntensityAttr(fallback_dome_intensity)
        dome.CreateColorTemperatureAttr(color_temperature)
        dome.CreateEnableColorTemperatureAttr(True)


def discover_test_cameras(
    stage,
    n_cameras=4,
    building_path="/World/Building",
    offset_frac=0.3,
    z_offset=3.0,
):
    """Auto-discover camera positions from a building's floor geometry.

    Building-agnostic replacement for the hardcoded TEST_CAMERAS dict.
    Returns a dict with the same {pos, target, room} structure, keyed by
    floor name. All coordinates are in the stage's native units — works
    correctly for any physics USD (all of ours are in meters after
    setup_building_physics.py).

    Algorithm:
      1. Call discover_floor_levels(stage) to get per-floor centroids.
      2. Pick n_cameras evenly-spaced floors (bottom, 1/3, 2/3, top).
      3. Each camera: offset_frac * building_half_size away from the floor
         centroid toward one bbox corner, raised z_offset above the floor.
      4. Target = floor centroid.

    Args:
        stage: Composed USD stage (after add_reference).
        n_cameras: How many floors to sample (clamped to floors available).
        building_path: Root prim of the building.
        offset_frac: Fraction of building half-extent to shift the camera
            horizontally away from the centroid.
        z_offset: Height above the floor for the camera (stage units).

    Returns:
        dict[str, dict]: {floor_name: {"pos": (x,y,z), "target": (x,y,z),
        "room": str}}. Returns the static TEST_CAMERAS dict as a fallback
        if discovery fails (no /World/Building prim, no floors, etc).
    """
    # Import here to avoid circular import at module load time.
    from sim_utils.building_physics import discover_floor_levels

    building_prim = stage.GetPrimAtPath(building_path)
    if not building_prim or not building_prim.IsValid():
        return dict(TEST_CAMERAS)

    levels = discover_floor_levels(stage, building_path=building_path)
    if not levels:
        return dict(TEST_CAMERAS)

    # Building bbox → half extents for camera offset.
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    )
    bbox = bbox_cache.ComputeWorldBound(building_prim).ComputeAlignedRange()
    bb_min, bb_max = bbox.GetMin(), bbox.GetMax()
    half_x = (bb_max[0] - bb_min[0]) / 2.0
    half_y = (bb_max[1] - bb_min[1]) / 2.0

    # Pick n evenly-spaced floor indices.
    n = min(n_cameras, len(levels))
    if n == 1:
        indices = [0]
    else:
        indices = [round(i * (len(levels) - 1) / (n - 1)) for i in range(n)]

    cameras = {}
    # Rotate through 4 corners so cameras don't all face from the same side.
    corners = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
    for slot, idx in enumerate(indices):
        lvl = levels[idx]
        cx, cy, cz = lvl["center_x"], lvl["center_y"], lvl["z"]
        sx, sy = corners[slot % 4]
        pos = (
            cx + sx * offset_frac * half_x,
            cy + sy * offset_frac * half_y,
            cz + z_offset,
        )
        target = (cx, cy, cz)
        cameras[lvl["name"]] = {
            "pos": pos,
            "target": target,
            "room": f"{lvl['name']} (z={cz:.1f}m)",
        }
    return cameras


def add_interior_lighting(stage, cameras=None):
    """Add lighting suitable for interior building scenes.

    Enclosed rooms need strong fill light.  Uses a combination of:
      - DistantLight (sun) for natural shadows
      - DomeLight (ambient fill) for overall brightness
      - SphereLight at each camera position for local fill
      - RectLight at each camera target (floor center) for even coverage

    Args:
        stage: USD stage to add lights to.
        cameras: Camera dict (same shape as TEST_CAMERAS). If None,
            falls back to the static TEST_CAMERAS for backward compat.
    """
    if cameras is None:
        cameras = TEST_CAMERAS

    distant = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Val_Sun"))
    distant.CreateIntensityAttr(2000)
    distant.CreateAngleAttr(1.0)
    xform = UsdGeom.Xformable(distant.GetPrim())
    xform.AddRotateXYZOp().Set((-45, 30, 0))

    UsdLux.DomeLight.Define(
        stage, Sdf.Path("/World/Val_DomeLight")
    ).CreateIntensityAttr(1000)

    for key, cam in cameras.items():
        # Sphere light at camera position (overhead fill)
        light = UsdLux.SphereLight.Define(
            stage, Sdf.Path(f"/World/Val_CamLight_{key}")
        )
        light.CreateIntensityAttr(10000)
        light.CreateRadiusAttr(1.0)
        UsdGeom.Xformable(light.GetPrim()).AddTranslateOp().Set(
            tuple(cam["pos"])
        )

        # Rect light at target center (floor fill, facing down)
        rect = UsdLux.RectLight.Define(
            stage, Sdf.Path(f"/World/Val_FloorLight_{key}")
        )
        rect.CreateIntensityAttr(5000)
        rect.CreateWidthAttr(15.0)
        rect.CreateHeightAttr(15.0)
        rx = UsdGeom.Xformable(rect.GetPrim())
        tx, ty, tz = cam["target"]
        rx.AddTranslateOp().Set((tx, ty, tz + 4.0))
        rx.AddRotateXYZOp().Set((90, 0, 0))
