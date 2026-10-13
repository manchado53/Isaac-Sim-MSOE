#!/usr/bin/env python3
"""
VietsField Soccer Demo: Nova Carter pushes a soccer ball into a goal.

Loads VietsField_physics.usd, finds the two Goal_Netting prims, raycasts
down to the field surface, spawns Nova Carter behind a DynamicSphere,
and runs a simple P-controller that approaches the ball then steers it
toward the target goal. Recorded with a chase-cam behind the robot.

Usage:
    sbatch scripts/slurm/submit_isaac_job.sh simulations/environment/vietsfield_soccer_demo.py
    sbatch scripts/slurm/submit_isaac_job.sh simulations/environment/vietsfield_soccer_demo.py <usd_path>

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
from isaacsim.core.api.objects import DynamicSphere, FixedCuboid
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


def build_goal_colliders(world, target_goal, surface_z, approach_dir_x):
    """Build a primitive-based soccer goal cage aligned to the Revit goal bbox.

    Two vertical posts, a crossbar, and a back wall — the open front faces
    midfield so the ball rolls in and Carter can drive into the goal. The
    Revit mesh stays as visual; these primitives provide the physics.

    Args:
        world: isaacsim World (for scene.add).
        target_goal: dict from find_goals() with x_min/x_max/y_min/y_max/z_min/z_max.
        surface_z: ground z-height.
        approach_dir_x: +1 if Carter approaches from -x (goal front faces -x),
                        -1 if Carter approaches from +x.
    """
    g = target_goal
    goal_y_center = 0.5 * (g["y_min"] + g["y_max"])
    goal_width = g["y_max"] - g["y_min"]
    POST_THICK = 0.12          # m — regulation-ish post diameter
    CROSSBAR_H = 2.44          # m — regulation goal height
    BACK_OFFSET = 0.1          # m — back wall inset from x_max/x_min

    # Front = face closer to midfield. If Carter approaches from -x
    # (approach_dir_x == +1), the front is at x_min. Otherwise x_max.
    front_x = g["x_min"] if approach_dir_x > 0 else g["x_max"]
    back_x  = g["x_max"] if approach_dir_x > 0 else g["x_min"]

    # Post base z is on the field; post center z is half the post height up.
    post_z = surface_z + CROSSBAR_H / 2.0
    crossbar_z = surface_z + CROSSBAR_H + POST_THICK / 2.0

    parts = [
        # Left post (y_min side)
        dict(name="goal_left_post", prim="/World/Goal/LeftPost",
             position=[front_x, g["y_min"] + POST_THICK / 2.0, post_z],
             scale=[POST_THICK, POST_THICK, CROSSBAR_H]),
        # Right post (y_max side)
        dict(name="goal_right_post", prim="/World/Goal/RightPost",
             position=[front_x, g["y_max"] - POST_THICK / 2.0, post_z],
             scale=[POST_THICK, POST_THICK, CROSSBAR_H]),
        # Crossbar (horizontal, full width of goal)
        dict(name="goal_crossbar", prim="/World/Goal/Crossbar",
             position=[front_x, goal_y_center, crossbar_z],
             scale=[POST_THICK, goal_width, POST_THICK]),
        # Back wall — stops the ball after it rolls in
        dict(name="goal_backwall", prim="/World/Goal/BackWall",
             position=[back_x - approach_dir_x * BACK_OFFSET,
                      goal_y_center, post_z],
             scale=[POST_THICK, goal_width, CROSSBAR_H]),
    ]
    for p in parts:
        world.scene.add(FixedCuboid(
            prim_path=p["prim"],
            name=p["name"],
            position=np.array(p["position"]),
            scale=np.array(p["scale"]),
            color=np.array([1.0, 1.0, 1.0]),
        ))
    print(f"[physics] Built {len(parts)} goal primitives at "
          f"front_x={front_x:.2f}, back_x={back_x:.2f}, "
          f"y=[{g['y_min']:.2f}, {g['y_max']:.2f}]", flush=True)


def find_goals(stage):
    """Locate the placed Soccer_Goal instances in the Revit export.

    In the Viets Field USD each soccer goal is an Xform named like
    Specialty_Equipment_NNNNNN sitting under
    /Specialty_Equipment/Soccer_Goal/Soccer_Goal/. They're 2.5m deep x 7.6m
    wide x 2.6m tall (regulation). The open front faces midfield. The outer
    /Soccer_Goal parent wraps both instances and has a bbox spanning the whole
    field, which is why we can't just match by the name 'Soccer_Goal'.
    """
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    # Find the Soccer_Goal folder(s) that hold the actual instance Xforms.
    instance_parents = []
    for prim in stage.Traverse():
        if prim.GetName() != "Soccer_Goal":
            continue
        if "/FamilyData/" in prim.GetPath().pathString:
            continue
        instance_parents.append(prim)

    goals = []
    seen_keys = set()
    for parent in instance_parents:
        for prim in parent.GetChildren():
            try:
                rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                if rng.IsEmpty():
                    continue
                bmin = rng.GetMin()
                bmax = rng.GetMax()
                extent = [bmax[i] - bmin[i] for i in range(3)]
                # Regulation-ish dimensions: each axis roughly 2-8m.
                if max(extent) > 15.0 or max(extent) < 1.0:
                    continue
                if bmax[2] < 2.0:  # skip anything parked at the origin
                    continue
            except Exception:
                continue
            key = (round(0.5 * (bmin[0] + bmax[0]), 1),
                   round(0.5 * (bmin[1] + bmax[1]), 1))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            goals.append({
                "name": prim.GetName(),
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

    # --- Find goals -------------------------------------------------------
    goals = find_goals(stage)
    print(f"[goals] Found {len(goals)} Soccer_Goal prims:", flush=True)
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

    # --- Build primitive goal colliders -----------------------------------
    # approach_dir_x: +1 if Carter comes from -x (goal front at x_min),
    #                 -1 if Carter comes from +x (goal front at x_max).
    approach_dir_x = 1 if tgt_x > midfield[0] else -1
    build_goal_colliders(world, target_goal, surface_z, approach_dir_x)

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
    # Cinematic side-angle: starts wide (whole stadium in frame), then lerps
    # to a medium distance that keeps Carter and the goal both framed.
    CAM_OFFSET_START = np.array([-18.0, 30.0, 14.0])
    CAM_OFFSET_END   = np.array([-7.0, 12.0, 5.0])
    follow_cam = FollowCamera(
        prim_path="/World/FollowCam",
        local_offset=CAM_OFFSET_START.tolist(),
        resolution=(1920, 1080),
        fps=30,
        z_clamp=(surface_z + 2.0, surface_z + 50.0),
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
    # Goal-detection geometry: real Soccer_Goal bbox (2.5m deep × 7.6m wide
    # × 2.6m tall). The open front faces midfield; ball is "in" the goal
    # when it's inside this volume.
    goal_x_lo, goal_x_hi = target_goal["x_min"], target_goal["x_max"]
    goal_y_lo, goal_y_hi = target_goal["y_min"], target_goal["y_max"]
    goal_z_lim = surface_z + (target_goal["z_max"] - target_goal["z_min"]) + 0.5
    # Push target = dead center of the goal bbox so Carter ends up inside.
    push_x = 0.5 * (goal_x_lo + goal_x_hi)
    push_y = 0.5 * (goal_y_lo + goal_y_hi)

    phase = "approach"
    phase_start_frame = 0
    scored_frame = None
    # Cinematic zoom: progress 0 at spawn → 1 when Carter reaches push target.
    spawn_to_push_dist = math.hypot(
        push_x - float(robot_spawn[0]), push_y - float(robot_spawn[1])
    )

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

            # Goal detection — ball has pushed past the midline of the goal
            # (so Carter has time to drive all the way into the goal first).
            goal_mid_x = 0.5 * (goal_x_lo + goal_x_hi)
            past_mid = (bx > goal_mid_x) if approach_dir_x > 0 else (bx < goal_mid_x)
            in_y = goal_y_lo <= by <= goal_y_hi
            in_z = bz <= goal_z_lim
            scored = past_mid and in_y and in_z

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
                tx, ty = push_x, push_y
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

            # Cinematic zoom — interpolate camera offset based on how far
            # Carter has progressed toward the push target. Holds at END
            # once scoring fires so the celebration is framed consistently.
            if scored_frame is not None:
                progress = 1.0
            elif spawn_to_push_dist > 0.1:
                remaining = math.hypot(push_x - rx, push_y - ry)
                progress = 1.0 - (remaining / spawn_to_push_dist)
            else:
                progress = 0.0
            progress = max(0.0, min(1.0, progress))
            # Smoothstep easing so the zoom feels organic
            eased = progress * progress * (3.0 - 2.0 * progress)
            offset = CAM_OFFSET_START + eased * (CAM_OFFSET_END - CAM_OFFSET_START)
            follow_cam.set_local_offset(offset.tolist())

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
