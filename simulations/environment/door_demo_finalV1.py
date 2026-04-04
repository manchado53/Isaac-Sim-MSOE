#!/usr/bin/env python3
"""
Single Door Opening Demo — Innovation Center (finalV1)

Loads the pre-built finalV1 physics USD, auto-discovers a single hinged
door with a working drive, positions a camera INSIDE the room facing the
door, and records the door swinging open and closed.

Run on Rosie:
    sbatch submit_isaac_job.sh simulations/environment/door_demo_finalV1.py

Output:
    videos/environment/door_demo_finalV1.mp4  (12s @ 30fps, 1920x1080)
    logs/door_demo_finalV1.txt
"""

import os
import sys
import math

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from sim_utils.video_capture import VideoRecorder
from sim_utils.building_physics import (
    set_drive_target,
    classify_door_mechanism,
    DoorMechanism,
)

_report_path = os.path.join(_project_dir, "logs", "door_demo_finalV1.txt")
os.makedirs(os.path.dirname(_report_path), exist_ok=True)
_report_file = None


def log(msg=""):
    print(msg, flush=True)
    if _report_file:
        _report_file.write(msg + "\n")
        _report_file.flush()


# ---------------------------------------------------------------------------
# Door discovery
# ---------------------------------------------------------------------------

def find_single_hinged_door(stage):
    """Auto-discover a single hinged door with a working DriveAPI.

    Traverses all prims looking for RigidBodyAPI prims whose path contains
    "Door", classifies them, and picks the first HINGED_SINGLE with a valid
    drive. Returns (door_prim, drive) or (None, None).
    """
    candidates = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        path_str = str(prim.GetPath())
        if "Door" not in path_str and "door" not in path_str:
            continue

        mechanism = classify_door_mechanism(prim)
        if mechanism != DoorMechanism.HINGED_SINGLE:
            continue

        # Look for DriveAPI on child joint
        drive = _find_drive(prim)
        if drive is None:
            continue

        candidates.append((prim, drive))

    if not candidates:
        return None, None

    # Prefer doors on lower floors (smaller Z in bbox) for visibility
    def _door_z(pair):
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
        )
        bbox = bbox_cache.ComputeWorldBound(pair[0])
        rng = bbox.ComputeAlignedRange()
        return rng.GetMin()[2]

    candidates.sort(key=_door_z)
    return candidates[0]


def _find_drive(door_prim):
    """Find DriveAPI on a child joint prim."""
    for child in door_prim.GetChildren():
        name = child.GetName()
        if "Hinge" in name or "Joint" in name:
            drive = UsdPhysics.DriveAPI(child, "angular")
            if drive.GetTargetPositionAttr().IsValid():
                return drive
    return None


def get_door_bbox(door_prim):
    """Get world-space bounding box of the door."""
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    )
    world_bbox = bbox_cache.ComputeWorldBound(door_prim)
    rng = world_bbox.ComputeAlignedRange()
    return rng.GetMin(), rng.GetMax()


# ---------------------------------------------------------------------------
# Camera — positioned INSIDE the room looking at the door
# ---------------------------------------------------------------------------

def create_interior_camera(door_center, door_min, door_max):
    """
    Position camera on the interior side of the door, looking at it.

    For a single hinged door the hinge is at one edge. The door swings
    away from the camera. We place the camera offset along the door's
    thin axis (face-normal direction) on the side the door swings INTO,
    so we see the door opening toward us.

    Building coords are in feet (~0.3048 m each). A 4-6 foot offset
    gives a good interior room view.
    """
    cx, cy, cz = door_center

    dx = door_max[0] - door_min[0]
    dy = door_max[1] - door_min[1]

    # The thin dimension is the face-normal direction.
    # Camera goes on the positive side of the thin axis (interior).
    if dy >= dx:
        # Door extends along Y (wide face). Thin axis is X.
        # Place camera on the +X side (interior), looking back at door.
        cam_pos = np.array([cx + 5.0, cy, cz + 1.0])
    else:
        # Door extends along X (wide face). Thin axis is Y.
        cam_pos = np.array([cx, cy + 5.0, cz + 1.0])

    target = np.array([cx, cy, cz])

    # Look-at orientation
    d = target - cam_pos
    yaw = math.degrees(math.atan2(d[1], d[0]))
    pitch = math.degrees(math.atan2(-d[2], math.sqrt(d[0]**2 + d[1]**2)))

    orient = rot_utils.euler_angles_to_quats(
        np.array([0, pitch, yaw]), degrees=True
    )

    return Camera(
        prim_path="/World/InteriorCamera",
        position=cam_pos,
        frequency=30,
        resolution=(1920, 1080),
        orientation=orient,
    )


# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------

def add_lighting(stage):
    """Add bright lighting visible from interior."""
    distant = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    distant.CreateIntensityAttr(2000)
    distant.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    fill = UsdLux.DomeLight.Define(stage, "/World/FillLight")
    fill.CreateIntensityAttr(600)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _report_file
    _report_file = open(_report_path, "w")

    log("=" * 60)
    log("Single Door Opening Demo — Innovation Center (finalV1)")
    log("=" * 60)

    # Load pre-built physics USD
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    physics_usd = os.path.join(
        _project_dir, "assets", "environments", "msoe_building",
        "innovation_center_finalV1_physics.usd"
    )
    if not os.path.exists(physics_usd):
        log(f"ERROR: {physics_usd} not found.")
        log("Run setup_innovation_center_physics.py first.")
        _report_file.close()
        simulation_app.close()
        sys.exit(1)

    log(f"Loading: {physics_usd}")
    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(physics_usd, Sdf.Path("/World/Building"))
    log("Building loaded.")

    # --- Auto-discover a single hinged door ---
    log("\nAuto-discovering single hinged doors...")
    door_prim, drive = find_single_hinged_door(stage)
    if door_prim is None:
        log("ERROR: No single hinged door with DriveAPI found in the building.")
        _report_file.close()
        simulation_app.close()
        sys.exit(1)

    door_path = str(door_prim.GetPath())
    mechanism = classify_door_mechanism(door_prim)
    log(f"Selected door: {door_path}")
    log(f"Mechanism: {mechanism.name}")

    # Get door bounding box for camera positioning
    bb_min, bb_max = get_door_bbox(door_prim)
    door_center = [
        (bb_min[0] + bb_max[0]) / 2.0,
        (bb_min[1] + bb_max[1]) / 2.0,
        (bb_min[2] + bb_max[2]) / 2.0,
    ]
    log(f"Door center: ({door_center[0]:.2f}, {door_center[1]:.2f}, {door_center[2]:.2f})")
    log(f"Door bbox: min=({bb_min[0]:.2f}, {bb_min[1]:.2f}, {bb_min[2]:.2f}) "
        f"max=({bb_max[0]:.2f}, {bb_max[1]:.2f}, {bb_max[2]:.2f})")

    # Freeze all OTHER dynamic bodies so only our door moves
    frozen = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim != door_prim:
            rb = UsdPhysics.RigidBodyAPI(prim)
            rb.CreateKinematicEnabledAttr(True)
            frozen += 1
    log(f"Froze {frozen} other dynamic bodies")

    # Setup camera and lighting
    add_lighting(stage)
    camera = create_interior_camera(door_center, bb_min, bb_max)
    log(f"Camera position: {camera.get_world_pose()[0]}")

    world.reset()
    camera.initialize()

    # Warm up rendering pipeline
    log("Warming up renderer (30 frames)...")
    for _ in range(30):
        world.step(render=True)

    # --- Animation timeline ---
    # 30 FPS, 12 seconds total = 360 frames
    #   0-59:    door closed, establishing shot (2s)
    #  60-179:   door swings open 0 -> 80 deg (4s, smooth ease-in-out)
    # 180-269:   hold open at 80 deg (3s)
    # 270-359:   door closes 80 -> 0 deg (3s)
    total_frames = 360
    open_start = 60
    open_end = 180
    hold_end = 270
    close_end = 360
    max_angle = 80.0

    def ease_in_out(t):
        """Smooth ease-in-out curve (cubic)."""
        if t < 0.5:
            return 4 * t * t * t
        return 1 - (-2 * t + 2) ** 3 / 2

    log(f"\nRecording {total_frames} frames (12s at 30fps)...")

    with VideoRecorder("environment/door_demo_finalV1", fps=30) as recorder:
        for i in range(total_frames):
            # Compute target angle
            if i < open_start:
                angle = 0.0
            elif i < open_end:
                t = (i - open_start) / (open_end - open_start)
                angle = ease_in_out(t) * max_angle
            elif i < hold_end:
                angle = max_angle
            else:
                t = (i - hold_end) / (close_end - hold_end)
                angle = max_angle * (1.0 - ease_in_out(t))

            set_drive_target(drive, angle)
            world.step(render=True)

            rgba = camera.get_rgba()
            if rgba is not None and i > 0:
                recorder.capture_rgba(rgba)

            if i % 60 == 0:
                log(f"  Frame {i}/{total_frames} (angle={angle:.1f} deg)")

    log(f"\nDone!")
    log(f"  Video  -> videos/environment/door_demo_finalV1.mp4")
    log(f"  Report -> {_report_path}")
    log("=" * 60)

    _report_file.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
