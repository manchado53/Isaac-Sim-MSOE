#!/usr/bin/env python3
"""
Door Physics Tutorial — Isaac Sim

Builds a simple room (4 walls + floor) with a hinged door that swings open
using UsdPhysics.DriveAPI (angular position drive). Every physics API call
is applied directly so you can see exactly what each one does.

Concepts covered:
  - Static collision geometry (walls, floor)
  - Dynamic rigid body (door)
  - Revolute joint (hinge) with limits
  - Joint drive (motor) to swing the door open programmatically
  - Camera sensor + VideoRecorder for MP4 output

Run on Rosie:
    sbatch submit_isaac_job.sh simulations/rigid_body/door_physics_demo.py
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

# Add project root so we can import sim_utils
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _project_dir)

from sim_utils.video_capture import VideoRecorder
from sim_utils.building_physics import (
    create_revolute_joint,
    setup_physics_scene,
    setup_door_drive,
    set_drive_target,
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def create_static_box(stage, path, center, scale, color=None):
    """
    Create a scaled cube and apply CollisionAPI so it acts as a static wall.

    Static geometry does NOT need RigidBodyAPI — omitting it tells PhysX that
    this object is immovable, which is cheaper to simulate.
    """
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*center))

    cube_path = f"{path}/Cube"
    cube = UsdGeom.Cube.Define(stage, cube_path)
    cube.AddScaleOp().Set(Gf.Vec3f(*scale))

    # CollisionAPI on a prim WITHOUT RigidBodyAPI → static collider.
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    if color is not None:
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    return xform, cube


# ---------------------------------------------------------------------------
# Room construction
# ---------------------------------------------------------------------------

def build_room(stage):
    """
    Build a simple room: floor + 4 walls.
    The front wall (Y = -3) is split into two segments with a doorway gap.
    """
    room_root = "/World/Room"
    UsdGeom.Xform.Define(stage, room_root)

    room_w = 6.0    # X extent (full width)
    room_d = 6.0    # Y extent (full depth)
    room_h = 2.5    # wall height
    wall_t = 0.1    # wall thickness
    door_w = 1.2    # doorway width

    # Floor
    create_static_box(stage, f"{room_root}/Floor",
                      center=(0, 0, -wall_t / 2),
                      scale=(room_w / 2, room_d / 2, wall_t / 2),
                      color=(0.35, 0.35, 0.35))

    # Back wall (Y = +room_d/2)
    create_static_box(stage, f"{room_root}/WallBack",
                      center=(0, room_d / 2, room_h / 2),
                      scale=(room_w / 2, wall_t / 2, room_h / 2),
                      color=(0.85, 0.85, 0.8))

    # Left wall (X = -room_w/2)
    create_static_box(stage, f"{room_root}/WallLeft",
                      center=(-room_w / 2, 0, room_h / 2),
                      scale=(wall_t / 2, room_d / 2, room_h / 2),
                      color=(0.85, 0.85, 0.8))

    # Right wall (X = +room_w/2)
    create_static_box(stage, f"{room_root}/WallRight",
                      center=(room_w / 2, 0, room_h / 2),
                      scale=(wall_t / 2, room_d / 2, room_h / 2),
                      color=(0.85, 0.85, 0.8))

    # Front wall (Y = -room_d/2), split into two segments around doorway
    seg_w = (room_w - door_w) / 2
    left_cx = -room_w / 2 + seg_w / 2
    right_cx = room_w / 2 - seg_w / 2

    create_static_box(stage, f"{room_root}/WallFrontLeft",
                      center=(left_cx, -room_d / 2, room_h / 2),
                      scale=(seg_w / 2, wall_t / 2, room_h / 2),
                      color=(0.85, 0.85, 0.8))

    create_static_box(stage, f"{room_root}/WallFrontRight",
                      center=(right_cx, -room_d / 2, room_h / 2),
                      scale=(seg_w / 2, wall_t / 2, room_h / 2),
                      color=(0.85, 0.85, 0.8))

    return {
        "door_y": -room_d / 2,
        "door_x_min": -door_w / 2,
        "door_x_max":  door_w / 2,
        "door_height": room_h,
        "wall_thickness": wall_t,
    }


# ---------------------------------------------------------------------------
# Door creation (dynamic rigid body)
# ---------------------------------------------------------------------------

def create_door(stage, info):
    """
    Create a dynamic door panel in the doorway and attach it with a revolute
    joint (hinge) on the left edge. The door swings inward (+Y direction).
    """
    door_path = "/World/Door"
    door_w = info["door_x_max"] - info["door_x_min"]
    door_h = info["door_height"] - 0.05   # small gap so it doesn't scrape
    door_t = 0.05

    door_cx = (info["door_x_min"] + info["door_x_max"]) / 2
    door_cy = info["door_y"]
    door_cz = door_h / 2

    # --- Xform: the rigid body lives here ---
    door_xform = UsdGeom.Xform.Define(stage, door_path)
    door_xform.AddTranslateOp().Set(Gf.Vec3d(door_cx, door_cy, door_cz))
    door_prim = door_xform.GetPrim()

    # RigidBodyAPI makes this prim physics-simulated (it will move).
    UsdPhysics.RigidBodyAPI.Apply(door_prim)

    # MassAPI — set explicit mass so PhysX doesn't guess from volume.
    mass_api = UsdPhysics.MassAPI.Apply(door_prim)
    mass_api.CreateMassAttr(15.0)

    # --- Visible cube shape ---
    cube_path = f"{door_path}/DoorPanel"
    door_cube = UsdGeom.Cube.Define(stage, cube_path)
    door_cube.AddScaleOp().Set(Gf.Vec3f(door_w / 2, door_t / 2, door_h / 2))
    door_cube.CreateDisplayColorAttr([Gf.Vec3f(0.7, 0.2, 0.2)])  # red so it's visible
    cube_prim = door_cube.GetPrim()

    # CollisionAPI + convexHull (required for dynamic bodies — triangle mesh
    # only works on statics).
    UsdPhysics.CollisionAPI.Apply(cube_prim)
    mesh_col = UsdPhysics.MeshCollisionAPI.Apply(cube_prim)
    mesh_col.CreateApproximationAttr("convexHull")

    # --- Revolute joint (hinge) ---
    # body0 = None → world frame (static)
    # body1 = door → the dynamic door
    # axis = "Z"  → vertical rotation (door swings horizontally)
    # local_pos0 = world position of hinge (left edge of doorway)
    # local_pos1 = door-local offset to its left edge
    hinge_world = Gf.Vec3f(info["door_x_min"], float(door_cy), float(door_cz))
    hinge_local = Gf.Vec3f(-door_w / 2, 0, 0)

    joint = create_revolute_joint(
        stage,
        joint_path="/World/DoorHinge",
        body0_path=None,
        body1_path=door_path,
        axis="Z",
        lower_limit=0.0,
        upper_limit=90.0,
        local_pos0=hinge_world,
        local_pos1=hinge_local,
    )

    return joint


# ---------------------------------------------------------------------------
# Lighting & camera
# ---------------------------------------------------------------------------

def add_lighting(stage):
    distant = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    distant.CreateIntensityAttr(700)

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(300)


def create_camera():
    """
    Angled top-down camera — positioned above and to the side of the room,
    looking down at the doorway at roughly 45 degrees.
    """
    # Camera sits above-front-right of the room, looking down into it
    cam_pos = np.array([5.0, -5.0, 20.0])
    target  = np.array([0.0, 0.0, 0.0])     # center of the room floor

    dx, dy, dz = target - cam_pos
    yaw   = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(-dz, math.sqrt(dx**2 + dy**2)))

    orient = rot_utils.euler_angles_to_quats(
        np.array([0, pitch, yaw]), degrees=True
    )

    return Camera(
        prim_path="/World/DoorCamera",
        position=cam_pos,
        frequency=30,
        resolution=(1280, 720),
        orientation=orient,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60, flush=True)
    print("Door Physics Demo", flush=True)
    print("=" * 60, flush=True)

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    setup_physics_scene(stage)

    doorway_info = build_room(stage)
    joint = create_door(stage, doorway_info)
    drive = setup_door_drive(joint, stiffness=500.0, damping=200.0, max_force=5000.0)

    add_lighting(stage)
    camera = create_camera()

    world.reset()
    camera.initialize()

    # Warm up rendering
    for _ in range(20):
        world.step(render=True)

    # --- Simulation: 300 frames (10s at 30fps) ---
    # Frames 0-59:    settle (door closed)
    # Frames 60-179:  door opens 0° → 90°
    # Frames 180-299: door holds open at 90°
    total_frames = 300
    open_start = 60
    open_end = 180

    print(f"Recording {total_frames} frames...", flush=True)

    with VideoRecorder("rigid_body/door_physics_demo", fps=30) as recorder:
        for i in range(total_frames):
            if i < open_start:
                set_drive_target(drive, 0.0)
            elif i < open_end:
                t = (i - open_start) / (open_end - open_start)
                set_drive_target(drive, t * 90.0)
            else:
                set_drive_target(drive, 90.0)

            world.step(render=True)

            rgba = camera.get_rgba()
            if rgba is not None and i > 0:
                recorder.capture_rgba(rgba)

            if i % 60 == 0:
                print(f"  Frame {i}/{total_frames}", flush=True)

    print("Done.", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
