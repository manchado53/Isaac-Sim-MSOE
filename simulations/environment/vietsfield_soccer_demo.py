#!/usr/bin/env python3
"""
VietsField Soccer Demo: Nova Carter pushes a soccer ball into a goal.

Loads VietsField_physics.usd, finds the two Goal_Netting prims, raycasts
down to the field surface, spawns Nova Carter behind a DynamicSphere,
and runs a simple P-controller that approaches the ball then steers it
toward the target goal. Recorded with a chase-cam behind the robot.

Usage:
    sbatch submit_isaac_job.sh simulations/environment/vietsfield_soccer_demo.py
    sbatch submit_isaac_job.sh simulations/environment/vietsfield_soccer_demo.py <usd_path>

Output:
    videos/vietsfield_soccer_goal.mp4
"""

import os
import sys
import math
import time

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import numpy as np
import carb
import omni.usd
import isaacsim.core.utils.numpy.rotations as rot_utils
from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, Sdf, Gf

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
sys.path.insert(0, _project_dir)

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicSphere
from isaacsim.robot.wheeled_robots.controllers.differential_controller import (
    DifferentialController,
)
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.storage.native import get_assets_root_path

from sim_utils.building_physics import setup_physics_scene, raycast_down
from sim_utils.follow_camera import FollowCamera
from sim_utils.video_capture import VideoRecorder


# --- Tuning ---------------------------------------------------------------
BALL_RADIUS = 0.11       # m (FIFA size-5)
BALL_MASS = 0.43         # kg
BALL_LEAD = 1.2          # m in front of robot when spawning
APPROACH_DIST = 0.55     # m: switch from "approach ball" to "push"
MAX_LINEAR = 1.8         # m/s
MAX_ANGULAR = 2.0        # rad/s
KP_HEADING = 2.5
KP_LINEAR = 1.5
FALLBACK_SURFACE_Z = 9.5 # m — used only if raycast fails

MAX_FRAMES = 1500        # ~25 s @ 60 Hz
CELEBRATE_FRAMES = 120   # ~2 s

# Default USD if none passed
DEFAULT_USD = os.path.join(
    _project_dir, "assets", "environments", "msoe_building",
    "VietsField_physics.usd",
)


# --- Helpers --------------------------------------------------------------

def quat_yaw(q):
    """Yaw (radians) from [w, x, y, z] quaternion."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_pi(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def apply_goal_colliders(stage):
    """Apply static triangle-mesh colliders to all Goal-related prims.

    Traverses the stage looking for prims whose name contains 'Goal' (netting,
    posts, crossbar, etc.) and applies CollisionAPI + triangle-mesh approximation
    so the ball bounces off the goal structure.
    """
    count = 0
    for prim in stage.Traverse():
        name = prim.GetName()
        if "Goal" not in name:
            continue
        # Only apply to prims that have renderable geometry (Mesh or Xform with children)
        if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Xform)):
            continue
        # Skip if already has collision
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        UsdPhysics.CollisionAPI.Apply(prim)
        mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_api.CreateApproximationAttr("none")  # "none" = exact triangle mesh
        count += 1
    print(f"[physics] Applied static colliders to {count} Goal prims", flush=True)


def find_goals(stage):
    """Return a list of dicts {name, center_xy, x_min, x_max, y_min, y_max, z_min, z_max}
    for each Goal_Netting prim found in the stage."""
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    goals = []
    seen_names = set()
    for prim in stage.Traverse():
        name = prim.GetName()
        if "Goal_Netting" not in name:
            continue
        # Dedup parent+child pairs: prefer the parent Xform (has the transform)
        if name in seen_names:
            continue
        try:
            rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            bmin = rng.GetMin()
            bmax = rng.GetMax()
            # Reject bboxes expressed in un-scaled Revit units (~100x too big)
            # or zero-extent un-placed instances. Netting can be <5 cm thin in
            # one axis, so only require the largest dimension to be sensible.
            extent = [bmax[i] - bmin[i] for i in range(3)]
            if max(extent) > 200.0 or max(extent) < 1.0:
                continue
            # Reject prims whose bbox is entirely outside the VietsField stage
            # bbox (mis-placed family-template geometry at the origin).
            if bmax[0] < -130 or bmin[0] > 75 or bmax[1] < -50 or bmin[1] > 60:
                continue
        except Exception:
            continue
        seen_names.add(name)
        goals.append({
            "name": name,
            "path": prim.GetPath().pathString,
            "x_min": float(bmin[0]), "x_max": float(bmax[0]),
            "y_min": float(bmin[1]), "y_max": float(bmax[1]),
            "z_min": float(bmin[2]), "z_max": float(bmax[2]),
            "center_xy": (
                0.5 * (float(bmin[0]) + float(bmax[0])),
                0.5 * (float(bmin[1]) + float(bmax[1])),
            ),
        })
    return goals


def pick_two_goals(goals):
    """Return (goal_a, goal_b) with the largest x-separation, or (g, None)."""
    if len(goals) == 0:
        return None, None
    if len(goals) == 1:
        return goals[0], None
    best = (goals[0], goals[1], 0.0)
    for i in range(len(goals)):
        for j in range(i + 1, len(goals)):
            dx = abs(goals[i]["center_xy"][0] - goals[j]["center_xy"][0])
            if dx > best[2]:
                best = (goals[i], goals[j], dx)
    return best[0], best[1]


def find_field_surface(origin_xy, fallback_z):
    """Raycast down from high above origin_xy to find the field surface z.
    Returns surface_z (float). Falls back to fallback_z if no hit."""
    x, y = origin_xy
    for z_high in (35.0, 25.0, 15.0):
        hit, dist, pos = raycast_down((x, y, z_high), max_distance=60.0)
        if hit and dist is not None and pos is not None:
            sz = float(pos[2])
            print(f"[spawn] surface raycast hit at z={sz:.2f} from z_high={z_high}",
                  flush=True)
            return sz
    print(f"[spawn] raycast miss, using fallback z={fallback_z}", flush=True)
    return fallback_z


def add_field_lighting(stage):
    """Outdoor-bright lighting for the stadium."""
    UsdGeom.Xform.Define(stage, "/World/DemoLights")

    dome = UsdLux.DomeLight.Define(stage, "/World/DemoLights/Dome")
    dome.CreateIntensityAttr(3500.0)

    sun = UsdLux.DistantLight.Define(stage, "/World/DemoLights/Sun")
    sun.CreateIntensityAttr(7500.0)
    sun.CreateAngleAttr(1.5)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-45, 0, 35))


# --- Main -----------------------------------------------------------------

def main():
    t_start = time.time()
    usd_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_USD

    if not os.path.exists(usd_path):
        print(f"ERROR: USD not found: {usd_path}", flush=True)
        simulation_app.close()
        sys.exit(1)
    print(f"[demo] USD: {usd_path}", flush=True)

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    building_prim = stage.DefinePrim("/World/Building", "Xform")
    building_prim.GetReferences().AddReference(
        usd_path, primPath=Sdf.Path("/World/Building")
    )
    setup_physics_scene(stage)
    add_field_lighting(stage)
    apply_goal_colliders(stage)

    # --- Find goals -------------------------------------------------------
    goals = find_goals(stage)
    print(f"[goals] Found {len(goals)} Goal_Netting prims:", flush=True)
    for g in goals:
        cx, cy = g["center_xy"]
        print(f"  {g['name']}: center=({cx:.1f}, {cy:.1f}) "
              f"x=[{g['x_min']:.1f}, {g['x_max']:.1f}] "
              f"y=[{g['y_min']:.1f}, {g['y_max']:.1f}] "
              f"z=[{g['z_min']:.1f}, {g['z_max']:.1f}]", flush=True)

    goal_a, goal_b = pick_two_goals(goals)
    if goal_a is None:
        print("ERROR: No goals found in USD", flush=True)
        simulation_app.close()
        sys.exit(1)

    # Pick target (the +x goal if two were found) and put the ball close
    # enough that Carter can score inside MAX_FRAMES (~25 s @ 1.7 m/s ≈ 42 m
    # of travel, so spawn 12 m short of the goal for a snappy demo).
    SPAWN_DISTANCE_FROM_GOAL = 12.0
    if goal_b is not None:
        target_goal = goal_a if goal_a["center_xy"][0] > goal_b["center_xy"][0] else goal_b
        other_goal = goal_b if target_goal is goal_a else goal_a
        dir_away = np.array([
            other_goal["center_xy"][0] - target_goal["center_xy"][0],
            other_goal["center_xy"][1] - target_goal["center_xy"][1],
        ])
        dir_away = dir_away / (np.linalg.norm(dir_away) + 1e-6)
        midfield = (
            target_goal["center_xy"][0] + dir_away[0] * SPAWN_DISTANCE_FROM_GOAL,
            target_goal["center_xy"][1] + dir_away[1] * SPAWN_DISTANCE_FROM_GOAL,
        )
    else:
        target_goal = goal_a
        gx, gy = goal_a["center_xy"]
        sign = -1.0 if gx > 0 else 1.0
        midfield = (gx + sign * SPAWN_DISTANCE_FROM_GOAL, gy)

    tgt_x, tgt_y = target_goal["center_xy"]
    print(f"[plan] midfield=({midfield[0]:.1f}, {midfield[1]:.1f}) → "
          f"goal={target_goal['name']} at ({tgt_x:.1f}, {tgt_y:.1f})", flush=True)

    # --- Find field surface via raycast ----------------------------------
    # Need one warm step before raycast can work
    world.reset()
    for _ in range(3):
        world.step(render=False)
    surface_z = find_field_surface(midfield, FALLBACK_SURFACE_Z)

    # --- Compute spawn positions -----------------------------------------
    goal_dir = np.array([tgt_x - midfield[0], tgt_y - midfield[1]])
    goal_dir = goal_dir / (np.linalg.norm(goal_dir) + 1e-6)

    ball_spawn = np.array([midfield[0], midfield[1], surface_z + BALL_RADIUS + 0.05])
    robot_spawn_xy = np.array([midfield[0], midfield[1]]) - goal_dir * BALL_LEAD
    robot_spawn = np.array([robot_spawn_xy[0], robot_spawn_xy[1], surface_z + 0.15])

    robot_yaw = math.atan2(goal_dir[1], goal_dir[0])
    robot_orient = rot_utils.euler_angles_to_quats(
        np.array([0.0, 0.0, math.degrees(robot_yaw)]), degrees=True
    )

    print(f"[spawn] ball  at {ball_spawn.tolist()}", flush=True)
    print(f"[spawn] robot at {robot_spawn.tolist()} yaw={math.degrees(robot_yaw):.1f}°",
          flush=True)

    # --- Spawn ball -------------------------------------------------------
    ball = world.scene.add(DynamicSphere(
        prim_path="/World/SoccerBall",
        name="soccer_ball",
        position=ball_spawn,
        radius=BALL_RADIUS,
        color=np.array([1.0, 1.0, 1.0]),
        mass=BALL_MASS,
    ))

    # --- Spawn Nova Carter -----------------------------------------------
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        print("ERROR: Could not find Isaac Sim assets folder", flush=True)
        simulation_app.close()
        sys.exit(1)
    carter_asset = assets_root_path + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"

    carter = world.scene.add(WheeledRobot(
        prim_path="/World/NovaCarter",
        name="nova_carter",
        wheel_dof_names=["joint_wheel_left", "joint_wheel_right"],
        create_robot=True,
        usd_path=carter_asset,
        position=robot_spawn,
        orientation=robot_orient,
    ))
    controller = DifferentialController(
        name="carter_ctrl", wheel_radius=0.14, wheel_base=0.4,
    )

    # --- Follow camera ---------------------------------------------------
    follow_cam = FollowCamera(
        prim_path="/World/FollowCam",
        local_offset=[-18.0, 0.0, 9.0],
        resolution=(1920, 1080),
        fps=30,
        z_clamp=(surface_z + 2.0, surface_z + 25.0),
        look_height=0.3,
    )

    # Second reset so ball + robot + cam are all registered in the scene
    world.reset()
    simulation_app.update()
    follow_cam.initialize()

    # Warmup
    for _ in range(30):
        world.step(render=True)

    # --- Run --------------------------------------------------------------
    # Goal-detection geometry: target goal's x-plane + y-span
    goal_x = tgt_x
    goal_y_lo = target_goal["y_min"]
    goal_y_hi = target_goal["y_max"]
    goal_z_lim = surface_z + 3.0
    # Direction from midfield to goal (unit vector) — used for "crossed plane" check
    cross_sign = 1.0 if goal_x > midfield[0] else -1.0

    phase = "approach"
    phase_start_frame = 0
    scored_frame = None

    video_name = "vietsfield_soccer_goal"
    print(f"[run] Recording {video_name}.mp4 ({MAX_FRAMES} frames max)", flush=True)

    with VideoRecorder(video_name, fps=30) as recorder:
        for frame in range(MAX_FRAMES):
            # Current poses
            r_pos, r_quat = carter.get_world_pose()
            b_pos, _ = ball.get_world_pose()
            r_yaw = quat_yaw(r_quat)

            rx, ry = float(r_pos[0]), float(r_pos[1])
            bx, by, bz = float(b_pos[0]), float(b_pos[1]), float(b_pos[2])

            # Goal detection — ball has crossed the goal plane
            crossed = cross_sign * (bx - goal_x) > 0.0
            within_y = (goal_y_lo - 1.0) <= by <= (goal_y_hi + 1.0)
            above_ground = bz < goal_z_lim
            scored = crossed and within_y and above_ground

            if scored and phase != "celebrate":
                phase = "celebrate"
                phase_start_frame = frame
                scored_frame = frame
                t_scored = frame / 60.0
                print(f"[goal] BALL IN NET at t={t_scored:.2f}s "
                      f"ball_xy=({bx:.2f}, {by:.2f}, {bz:.2f})", flush=True)

            # Choose heading target per phase
            if phase == "approach":
                tx, ty = bx, by
                dist_to_target = math.hypot(tx - rx, ty - ry)
                if dist_to_target < APPROACH_DIST:
                    phase = "push"
                    phase_start_frame = frame
                    print(f"[phase] approach→push at frame {frame} "
                          f"(dist_to_ball={dist_to_target:.2f})", flush=True)
            elif phase == "push":
                tx, ty = goal_x, 0.5 * (goal_y_lo + goal_y_hi)
                dist_to_target = math.hypot(tx - rx, ty - ry)
            else:  # celebrate
                tx, ty = rx, ry
                dist_to_target = 0.0

            # Controller
            if phase == "celebrate":
                elapsed = frame - phase_start_frame
                if elapsed < CELEBRATE_FRAMES:
                    # Spin in place
                    cmd = np.array([0.0, 1.5])
                else:
                    cmd = np.array([0.0, 0.0])
            else:
                desired_yaw = math.atan2(ty - ry, tx - rx)
                yaw_err = wrap_pi(desired_yaw - r_yaw)
                ang = max(-MAX_ANGULAR, min(MAX_ANGULAR, KP_HEADING * yaw_err))
                # Slow linear when heading error is large
                lin = min(MAX_LINEAR, KP_LINEAR * max(0.2, dist_to_target))
                lin *= max(0.15, math.cos(yaw_err))
                cmd = np.array([lin, ang])

            carter.apply_wheel_actions(controller.forward(command=cmd))

            # Follow-cam: track a point just ahead of the robot so both the
            # Carter and the ball stay in frame during the push.
            focus_pos = np.array([
                0.5 * (rx + bx),
                0.5 * (ry + by),
                surface_z + 0.3,
            ])
            # Use the robot's orientation so the cam stays behind it.
            follow_cam.update(focus_pos, r_quat)

            world.step(render=True)

            rgba = follow_cam.get_rgba()
            if rgba is not None and frame > 0:
                recorder.capture_rgba(rgba)

            # Stop after celebration
            if phase == "celebrate" and (frame - phase_start_frame) >= CELEBRATE_FRAMES:
                print(f"[run] End of celebration at frame {frame}", flush=True)
                break

            if frame % 120 == 0:
                print(f"  f={frame:4d} phase={phase:9s} "
                      f"robot=({rx:6.2f},{ry:6.2f}) "
                      f"ball=({bx:6.2f},{by:6.2f},{bz:5.2f})", flush=True)

    if scored_frame is None:
        print(f"[goal] TIMEOUT — no goal scored in {MAX_FRAMES} frames", flush=True)

    print(f"[done] runtime {time.time() - t_start:.1f}s", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
