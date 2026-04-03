#!/usr/bin/env python3
"""
Door Opening Showcase — MSOE Innovation Center

Loads the pre-built physics USD, finds a ground-floor 2FG entrance door,
positions a cinematic camera, and records the door swinging open.

The 2FG doors are double-width curtain-wall glass doors at the building's
ground floor — the closest thing to a "front door" in the Revit export.

Run on Rosie:
    sbatch submit_isaac_job.sh simulations/environment/door_opening_showcase.py
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
from sim_utils.building_physics import set_drive_target


# ---------------------------------------------------------------------------
# Door target — ground-floor 2FG entrance door
# ---------------------------------------------------------------------------
# Doors_9494852: a Curtain Wall Type 2FG door on the ground floor.
# World-space center: (-21.32, 68.21, 14.78), extends ~1.83m along Y,
# 2.44m tall. Hinge at min-Y edge (y ≈ 67.29).
DOOR_PRIM_SUFFIX = "Doors_9494852"


def find_door_prim(stage):
    """Find the target door prim by name suffix."""
    for prim in stage.Traverse():
        if prim.GetName() == DOOR_PRIM_SUFFIX:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return prim
    return None


def find_drive_on_door(stage, door_prim):
    """Find the DriveAPI joint for this door (child HingeJoint prim)."""
    for child in door_prim.GetChildren():
        if "Hinge" in child.GetName() or "Joint" in child.GetName():
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
# Camera
# ---------------------------------------------------------------------------

def create_showcase_camera(door_center, door_min, door_max):
    """
    Position camera for a cinematic view of the door opening.

    Camera sits in front and to the side of the door, slightly above,
    looking at the door center. This gives a 3/4 view showing the
    door face and the opening arc.
    """
    cx, cy, cz = door_center

    # Door extends along Y (wide), thin in X (face normal along X).
    # Hinge is at min-Y edge. Free edge swings into negative-X.
    # Camera goes to negative-X (in front) and offset toward free edge.
    dx = door_max[0] - door_min[0]
    dy = door_max[1] - door_min[1]

    if dy >= dx:
        # Door extends along Y — camera offset in X (front), slight Y offset
        cam_pos = np.array([cx - 6.0, cy + 1.5, cz + 1.5])
    else:
        # Door extends along X — camera offset in Y, slight X offset
        cam_pos = np.array([cx + 1.5, cy - 6.0, cz + 1.5])

    target = np.array([cx, cy, cz])

    # Compute look-at orientation
    d = target - cam_pos
    yaw = math.degrees(math.atan2(d[1], d[0]))
    pitch = math.degrees(math.atan2(-d[2], math.sqrt(d[0]**2 + d[1]**2)))

    orient = rot_utils.euler_angles_to_quats(
        np.array([0, pitch, yaw]), degrees=True
    )

    return Camera(
        prim_path="/World/ShowcaseCamera",
        position=cam_pos,
        frequency=30,
        resolution=(1920, 1080),
        orientation=orient,
    )


# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------

def add_lighting(stage):
    """Add bright lighting to illuminate the door area."""
    distant = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    distant.CreateIntensityAttr(1000)
    distant.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    fill = UsdLux.DomeLight.Define(stage, "/World/FillLight")
    fill.CreateIntensityAttr(400)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60, flush=True)
    print("Door Opening Showcase — MSOE Innovation Center", flush=True)
    print("=" * 60, flush=True)

    # Load pre-built physics USD
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    physics_usd = os.path.join(
        _project_dir, "assets", "environments", "msoe_building",
        "innovation_center_physics.usd"
    )
    print(f"Loading: {physics_usd}", flush=True)

    # Load building as reference under /World/Building
    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(physics_usd, Sdf.Path("/World/Building"))
    print("Building loaded.", flush=True)

    # Find the target door
    door_prim = find_door_prim(stage)
    if door_prim is None:
        print(f"ERROR: Could not find door {DOOR_PRIM_SUFFIX}", flush=True)
        simulation_app.close()
        return

    door_path = str(door_prim.GetPath())
    print(f"Found door: {door_path}", flush=True)

    # Get door bounding box for camera positioning
    bb_min, bb_max = get_door_bbox(door_prim)
    door_center = [
        (bb_min[0] + bb_max[0]) / 2.0,
        (bb_min[1] + bb_max[1]) / 2.0,
        (bb_min[2] + bb_max[2]) / 2.0,
    ]
    print(f"Door center: ({door_center[0]:.2f}, {door_center[1]:.2f}, {door_center[2]:.2f})", flush=True)
    print(f"Door bbox: min=({bb_min[0]:.2f}, {bb_min[1]:.2f}, {bb_min[2]:.2f}) "
          f"max=({bb_max[0]:.2f}, {bb_max[1]:.2f}, {bb_max[2]:.2f})", flush=True)

    # Find drive API on the door's joint
    drive = find_drive_on_door(stage, door_prim)
    if drive is None:
        print("ERROR: No DriveAPI found on door joint", flush=True)
        simulation_app.close()
        return
    print("DriveAPI found on door joint.", flush=True)

    # Setup camera and lighting
    add_lighting(stage)
    camera = create_showcase_camera(door_center, bb_min, bb_max)

    world.reset()
    camera.initialize()

    # Warm up rendering pipeline
    print("Warming up renderer...", flush=True)
    for _ in range(30):
        world.step(render=True)

    # --- Animation timeline ---
    # 30 FPS, 12 seconds total = 360 frames
    #   0-59:    door closed, establishing shot (2s)
    #  60-179:   door swings open 0° → 80° (4s, smooth ease-in-out)
    # 180-269:   hold open at 80° (3s)
    # 270-359:   door closes 80° → 0° (3s)
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

    print(f"Recording {total_frames} frames (12s at 30fps)...", flush=True)

    with VideoRecorder("door_opening_showcase", fps=30) as recorder:
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
                print(f"  Frame {i}/{total_frames} (angle={angle:.1f}°)", flush=True)

    print("Done! Video saved to videos/door_opening_showcase.mp4", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
