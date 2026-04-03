#!/usr/bin/env python3
"""
Door Walk-Through Showcase — MSOE Innovation Center

Cinematic walk-through: camera approaches a ground-floor entrance door from
outside, the door swings open, and the camera passes through the doorway
into the interior.

Target: Doors_9494852 (2FG curtain-wall glass door, ground floor).

Run on Rosie:
    sbatch submit_isaac_job.sh simulations/environment/door_walkthrough_showcase.py
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
# Target door
# ---------------------------------------------------------------------------
DOOR_PRIM_SUFFIX = "Doors_13605973"  # Curtain Wall Type_F, single-leaf, ground floor


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
# Lighting
# ---------------------------------------------------------------------------

def add_walkthrough_lighting(stage, door_center, door_min, door_max):
    """
    Aggressive lighting for exterior-to-interior walk-through.

    Overkill on purpose — multiple lights from every angle to guarantee
    visibility. We can scale back once we confirm the scene is visible.
    """
    cx, cy, cz = door_center

    # Determine approach axis
    dx = door_max[0] - door_min[0]
    dy = door_max[1] - door_min[1]
    door_along_y = dy >= dx

    # --- Sun 1: Primary directional from above-front ---
    sun1 = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun1"))
    sun1.CreateIntensityAttr(10000)
    sun1.CreateAngleAttr(1.0)
    UsdGeom.Xformable(sun1).AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    # --- Sun 2: Secondary from opposite side to fill shadows ---
    sun2 = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun2"))
    sun2.CreateIntensityAttr(5000)
    sun2.CreateAngleAttr(1.0)
    UsdGeom.Xformable(sun2).AddRotateXYZOp().Set(Gf.Vec3f(-30, -150, 0))

    # --- Dome: Very strong ambient fill from all directions ---
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome.CreateIntensityAttr(3000)

    # --- Interior sphere light: bright overhead inside the building ---
    interior1 = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/InteriorLight1"))
    interior1.CreateIntensityAttr(50000)
    interior1.CreateRadiusAttr(5.0)
    if door_along_y:
        pos1 = Gf.Vec3f(cx + 5.0, cy, cz + 8.0)
    else:
        pos1 = Gf.Vec3f(cx, cy + 5.0, cz + 8.0)
    UsdGeom.Xformable(interior1).AddTranslateOp().Set(pos1)

    # --- Interior sphere light 2: deeper inside ---
    interior2 = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/InteriorLight2"))
    interior2.CreateIntensityAttr(50000)
    interior2.CreateRadiusAttr(5.0)
    if door_along_y:
        pos2 = Gf.Vec3f(cx + 12.0, cy, cz + 8.0)
    else:
        pos2 = Gf.Vec3f(cx, cy + 12.0, cz + 8.0)
    UsdGeom.Xformable(interior2).AddTranslateOp().Set(pos2)

    # --- Exterior sphere light: in front of door, illuminates facade ---
    exterior = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/ExteriorLight"))
    exterior.CreateIntensityAttr(50000)
    exterior.CreateRadiusAttr(5.0)
    if door_along_y:
        pos_ext = Gf.Vec3f(cx - 8.0, cy, cz + 5.0)
    else:
        pos_ext = Gf.Vec3f(cx, cy - 8.0, cz + 5.0)
    UsdGeom.Xformable(exterior).AddTranslateOp().Set(pos_ext)

    # --- Rect light: flat panel above the doorway, like a floodlight ---
    rect = UsdLux.RectLight.Define(stage, Sdf.Path("/World/DoorFloodlight"))
    rect.CreateIntensityAttr(20000)
    rect.CreateWidthAttr(10.0)
    rect.CreateHeightAttr(10.0)
    if door_along_y:
        rect_pos = Gf.Vec3f(cx - 3.0, cy, cz + 10.0)
        rect_rot = Gf.Vec3f(-90, 0, 0)  # Point downward
    else:
        rect_pos = Gf.Vec3f(cx, cy - 3.0, cz + 10.0)
        rect_rot = Gf.Vec3f(-90, 0, 0)
    xf = UsdGeom.Xformable(rect)
    xf.AddTranslateOp().Set(rect_pos)
    xf.AddRotateXYZOp().Set(rect_rot)

    print(f"Lighting: 2 suns (10k+5k), dome (3k), 2 interior spheres (50k each), "
          f"1 exterior sphere (50k), 1 rect floodlight (20k)", flush=True)
    print(f"Interior lights at: ({pos1[0]:.1f}, {pos1[1]:.1f}, {pos1[2]:.1f}) "
          f"and ({pos2[0]:.1f}, {pos2[1]:.1f}, {pos2[2]:.1f})", flush=True)


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def look_at_orientation(cam_pos, target):
    """Compute camera orientation quaternion to look from cam_pos at target."""
    d = target - cam_pos
    yaw = math.degrees(math.atan2(d[1], d[0]))
    pitch = math.degrees(math.atan2(-d[2], math.sqrt(d[0]**2 + d[1]**2)))
    return rot_utils.euler_angles_to_quats(
        np.array([0, pitch, yaw]), degrees=True
    )


def ease_in_out(t):
    """Smooth cubic ease-in-out: slow start, fast middle, slow end."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4 * t * t * t
    return 1 - (-2 * t + 2) ** 3 / 2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60, flush=True)
    print("Door Walk-Through Showcase — MSOE Innovation Center", flush=True)
    print("=" * 60, flush=True)

    # Load pre-built physics USD
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    physics_usd = os.path.join(
        _project_dir, "assets", "environments", "msoe_building",
        "innovation_center_physics.usd"
    )
    print(f"Loading: {physics_usd}", flush=True)

    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        physics_usd, Sdf.Path("/World/Building")
    )
    print("Building loaded.", flush=True)

    # Find the target door
    door_prim = find_door_prim(stage)
    if door_prim is None:
        print(f"ERROR: Could not find door {DOOR_PRIM_SUFFIX}", flush=True)
        simulation_app.close()
        return

    door_path = str(door_prim.GetPath())
    print(f"Found door: {door_path}", flush=True)

    # Get door bounding box
    bb_min, bb_max = get_door_bbox(door_prim)
    door_center = [
        (bb_min[0] + bb_max[0]) / 2.0,
        (bb_min[1] + bb_max[1]) / 2.0,
        (bb_min[2] + bb_max[2]) / 2.0,
    ]
    cx, cy, cz = door_center
    print(f"Door center: ({cx:.2f}, {cy:.2f}, {cz:.2f})", flush=True)
    print(f"Door bbox: min=({bb_min[0]:.2f}, {bb_min[1]:.2f}, {bb_min[2]:.2f}) "
          f"max=({bb_max[0]:.2f}, {bb_max[1]:.2f}, {bb_max[2]:.2f})", flush=True)

    # Find DriveAPI
    drive = find_drive_on_door(stage, door_prim)
    if drive is None:
        print("ERROR: No DriveAPI found on door joint", flush=True)
        simulation_app.close()
        return
    print("DriveAPI found on door joint.", flush=True)

    # Determine door orientation and camera path axis
    dx = bb_max[0] - bb_min[0]
    dy = bb_max[1] - bb_min[1]
    eye_height = cz  # at door center height

    if dy >= dx:
        # Door extends along Y, face normal along X.
        # Camera approaches from -X (outside) to +X (inside).
        start_pos = np.array([cx - 8.0, cy, eye_height])
        door_pos = np.array([cx, cy, eye_height])
        end_pos = np.array([cx + 6.0, cy, eye_height])
    else:
        # Door extends along X, face normal along Y.
        start_pos = np.array([cx, cy - 8.0, eye_height])
        door_pos = np.array([cx, cy, eye_height])
        end_pos = np.array([cx, cy + 6.0, eye_height])

    print(f"Camera path: start=({start_pos[0]:.1f}, {start_pos[1]:.1f}, {start_pos[2]:.1f}) "
          f"→ door=({door_pos[0]:.1f}, {door_pos[1]:.1f}, {door_pos[2]:.1f}) "
          f"→ end=({end_pos[0]:.1f}, {end_pos[1]:.1f}, {end_pos[2]:.1f})", flush=True)

    # Setup lighting
    add_walkthrough_lighting(stage, door_center, bb_min, bb_max)

    # Create camera at start position
    orient = look_at_orientation(start_pos, door_pos)
    camera = Camera(
        prim_path="/World/WalkthroughCamera",
        position=start_pos,
        frequency=30,
        resolution=(1920, 1080),
        orientation=orient,
    )

    world.reset()
    camera.initialize()

    # Warm up renderer
    print("Warming up renderer...", flush=True)
    for _ in range(30):
        world.step(render=True)

    # --- Animation timeline ---
    # 360 frames = 12s at 30fps
    #
    # Camera path (LINEAR — steady speed, no acceleration):
    #   t=0.0 → start_pos (8ft outside)
    #   t~0.57 → door_pos  (at the door plane, ~frame 205)
    #   t=1.0 → end_pos   (6ft inside)
    #
    # Door animation (opens well before camera arrives):
    #   0-30:    closed (brief establishing)
    #   30-120:  opens 0° → 80° (ease-in-out)
    #   120-360: holds at 80°

    total_frames = 360
    door_open_start = 30
    door_open_end = 120
    max_angle = 80.0

    print(f"Recording {total_frames} frames (12s at 30fps)...", flush=True)

    with VideoRecorder("door_walkthrough_showcase", fps=30) as recorder:
        for i in range(total_frames):
            # --- Camera position ---
            # Linear interpolation: steady speed from start to end
            t_cam = i / (total_frames - 1)
            cam_pos = start_pos + t_cam * (end_pos - start_pos)

            # Look-at target: always look straight ahead along the path
            look_ahead_t = min(1.0, t_cam + 0.08)
            look_target = start_pos + look_ahead_t * (end_pos - start_pos)

            orient = look_at_orientation(cam_pos, look_target)
            camera.set_world_pose(position=cam_pos, orientation=orient)

            # --- Door angle ---
            if i < door_open_start:
                angle = 0.0
            elif i < door_open_end:
                t_door = (i - door_open_start) / (door_open_end - door_open_start)
                angle = ease_in_out(t_door) * max_angle
            else:
                angle = max_angle

            set_drive_target(drive, angle)
            world.step(render=True)

            rgba = camera.get_rgba()
            if rgba is not None and i > 0:
                recorder.capture_rgba(rgba)

            if i % 60 == 0:
                print(f"  Frame {i}/{total_frames} — "
                      f"cam=({cam_pos[0]:.1f}, {cam_pos[1]:.1f}, {cam_pos[2]:.1f}), "
                      f"door={angle:.1f}°", flush=True)

    print("Done! Video saved to videos/door_walkthrough_showcase.mp4", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
