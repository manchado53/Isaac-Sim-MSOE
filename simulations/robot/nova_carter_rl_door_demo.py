#!/usr/bin/env python3
"""
nova_carter_rl_door_demo.py — RL policy navigates to a door; VLM inspects its state.

The trained SB3 PPO policy (exported to ONNX) drives Nova Carter to
Doors_13606713 on Floor 4, then Qwen2-VL classifies the door as open or closed.

PyTorch/SB3 cannot coexist with Isaac Sim (CUDA conflict → segfault on shutdown).
Policy inference uses onnxruntime (no torch) inside the Isaac Sim process.
Export the checkpoint once with:  python training/export_policy_onnx.py --checkpoint <zip>

Usage:
  sbatch submit_isaac_job.sh simulations/robot/nova_carter_rl_door_demo.py
  sbatch submit_isaac_job.sh simulations/robot/nova_carter_rl_door_demo.py --door-angle 0
  sbatch submit_isaac_job.sh simulations/robot/nova_carter_rl_door_demo.py --door-angle 90

Args:
  --door-angle  door angle in degrees — 0=closed, 90=open (default: 45.0)
  --onnx        path to best_model.onnx (default: most recent run dir)
  --vecnorm     path to best_model_vecnorm.npz (default: same dir as --onnx)
"""

import argparse
import math
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# CLI args — parsed before any Isaac Sim import
# ---------------------------------------------------------------------------

_DEFAULT_RUN_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    "training", "logs", "sb3", "NovaCarter-Patrol", "2026-03-27_02-43-38",
)

_parser = argparse.ArgumentParser(description="RL+VLM door inspection demo.")
_parser.add_argument(
    "--door-angle", type=float, default=45.0,
    help="Door angle in degrees — 0=closed, 90=open (default: 45.0)",
)
_parser.add_argument(
    "--onnx", default=None,
    help="Path to best_model.onnx. Default: most recent run dir.",
)
_parser.add_argument(
    "--vecnorm", default=None,
    help="Path to best_model_vecnorm.npz. Default: same dir as --onnx.",
)
_parser.add_argument(
    "--job-id", default=os.environ.get("SLURM_JOB_ID", "local"),
    help="Job ID used to name the frame cache (default: SLURM_JOB_ID env var).",
)
_args, _unknown = _parser.parse_known_args()

# Resolve ONNX and VecNorm paths
_onnx_path = _args.onnx or os.path.join(_DEFAULT_RUN_DIR, "best_model.onnx")
_vecnorm_path = _args.vecnorm or _onnx_path.replace(".onnx", "_vecnorm.npz")

DOOR_ANGLE_DEG = _args.door_angle
JOB_ID = _args.job_id

# Timestamp for output files — set once so nothing is overwritten across runs
RUN_STAMP = time.strftime("%Y%m%d_%H%M%S")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Door we trained on: Doors_13606713
DOOR_CENTER_XY  = np.array([-20.48, 80.15])

# Robot spawns 1.5 m east of door — within TARGET_RADIUS so inspection starts immediately.
ROBOT_SPAWN_XY  = np.array([-18.98, 80.15])
ROBOT_SPAWN_Z   = 27.3          # Floor 4 slab Z=26.8 + 0.5 m clearance

# Distance thresholds
TARGET_RADIUS   = 2.0           # metres — switch to door inspection phase

# RL policy parameters (must match training config)
N_RAYS    = 20
MAX_RANGE = 10.0    # LiDAR max range in metres
LIDAR_Z   = 0.3    # sensor height above robot base (metres)
ACTION_SCALE = (0.5, 1.0)   # (linear_vel, angular_vel) from nova_carter_patrol_env_cfg.py:302

# VLM timing
SCAN_INTERVAL   = 45            # ~1.5 s between anomaly-scan queries (30 Hz sim)
N_DOOR_CHECKS   = 5             # VLM calls once stopped at door
DOOR_CHECK_GAP  = 20            # sim steps between door-check frames

# Camera
CAM_RESOLUTION  = (640, 480)
FPS             = 15

# Paths
_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
ADAPTER_PATH = os.path.join(_project_dir, "data", "qwen2vl_door_lora")
VLM_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ANOMALY_PROMPT = (
    "You are an autonomous security patrol robot navigating an indoor facility. "
    "Look carefully at this image and describe what you observe. "
    "Specifically flag any anomalies, safety hazards, or unusual situations such as: "
    "fire, smoke, water leaks, fallen objects, obstructions, or people in distress. "
    "If the scene looks completely normal, confirm that. "
    "Be concise — two or three sentences."
)

DOOR_PROMPT = (
    "Is this door open or closed? "
    "Answer with exactly: Door Status: [OPEN]. or Door Status: [CLOSED]."
)

print(f"[Demo] Door angle: {DOOR_ANGLE_DEG}° ({'OPEN' if DOOR_ANGLE_DEG > 5 else 'CLOSED'})")

# ---------------------------------------------------------------------------
# Isaac Sim initialisation — must be first isaacsim/omni import
# ---------------------------------------------------------------------------

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import carb
import omni.usd
from pxr import Sdf, Usd, UsdLux, UsdGeom, Gf
from scipy.spatial.transform import Rotation as ScipyR

import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

assets_root = get_assets_root_path()
if assets_root is None:
    carb.log_error("Could not find Isaac Sim assets root")
    simulation_app.close()
    sys.exit(1)

world = World(stage_units_in_meters=1.0)

_building_usd = os.path.join(_project_dir, "assets", "environments", "msoe_building", "innovation_center_finalV1.usd")
if not os.path.isfile(_building_usd):
    print(f"ERROR: Building USD not found: {_building_usd}")
    simulation_app.close()
    sys.exit(1)

print(f"[Sim] Loading building: {_building_usd}")
stage = omni.usd.get_context().get_stage()
stage.GetRootLayer().subLayerPaths.insert(0, _building_usd)

# ---------------------------------------------------------------------------
# Door angle control
# ---------------------------------------------------------------------------

_DOOR_BASE = (
    "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026"
    "/Geometry/Doors/RS___Curtain_Wall___Type_FG___New/New"
)
_DOOR_PATH = f"{_DOOR_BASE}/Doors_13606713"

_DOOR_MESHES = [
    {"path": _DOOR_PATH + "/AL_375207",       "ox": -2047.728271484375, "oy": 8014.9365234375,  "oz": 2677.16015625},
    {"path": _DOOR_PATH + "/Glass_GL_1_18439","ox": -2047.728271484375, "oy": 8014.9365234375,  "oz": 2702.559814453125},
]
_HINGE_X_CM = -2047.728271484375
_HINGE_Y_CM =  7969.07


def _door_matrix(ox_cm, oy_cm, oz_cm, angle_deg):
    θ = math.radians(angle_deg)
    c, s = math.cos(θ), math.sin(θ)
    dx, dy = ox_cm - _HINGE_X_CM, oy_cm - _HINGE_Y_CM
    return Gf.Matrix4d(
         c,  s, 0, 0,
        -s,  c, 0, 0,
         0,  0, 1, 0,
        c*dx - s*dy + _HINGE_X_CM,
        s*dx + c*dy + _HINGE_Y_CM,
        oz_cm, 1,
    )


def _set_door_angle(angle_deg):
    for m in _DOOR_MESHES:
        prim = stage.GetPrimAtPath(m["path"])
        if not prim.IsValid():
            print(f"[Sim] WARN: door mesh not found: {m['path']}")
            return
        for xop in UsdGeom.Xformable(prim).GetOrderedXformOps():
            if xop.GetOpName() == "xformOp:transform":
                xop.Set(_door_matrix(m["ox"], m["oy"], m["oz"], angle_deg))
                break


print(f"[Sim] Setting door angle to {DOOR_ANGLE_DEG}° "
      f"({'OPEN' if DOOR_ANGLE_DEG > 5 else 'CLOSED'})")
_set_door_angle(DOOR_ANGLE_DEG)

# ---------------------------------------------------------------------------
# Hide vegetation
# ---------------------------------------------------------------------------

_VEG_KEYWORDS = ("tree", "plant", "veget", "shrub", "foli", "planting",
                 "landscap", "palm", "bush", "hedge", "grass", "flora")


def _hide_vegetation(stg):
    from pxr import UsdGeom as _UG
    hidden = 0
    for prim in stg.Traverse():
        name = prim.GetName().lower()
        if any(k in name for k in _VEG_KEYWORDS):
            _UG.Imageable(prim).MakeInvisible()
            hidden += 1
    return hidden


_n_hidden = _hide_vegetation(stage)
print(f"[Sim] Vegetation hidden: {_n_hidden} prims")

# Lighting
dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(50000)
fill = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/FillLight"))
fill.CreateIntensityAttr(300000)
fill.CreateRadiusAttr(0.1)
fill_xform = UsdGeom.Xformable(fill.GetPrim())

follow_fill = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/FollowFillLight"))
follow_fill.CreateIntensityAttr(600000)
follow_fill.CreateRadiusAttr(0.1)
follow_fill_xform = UsdGeom.Xformable(follow_fill.GetPrim())

# Floor 4 collision plane
world.scene.add_default_ground_plane()
world.scene.add_default_ground_plane(
    z_position=26.75,
    name="floor4_plane",
    prim_path="/World/Floor4Plane",
    static_friction=0.3,
    dynamic_friction=0.3,
    restitution=0.0,
)

# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------

spawn_orientation = rot_utils.euler_angles_to_quats(np.array([0.0, 0.0, 180.0]), degrees=True)

carter = world.scene.add(
    WheeledRobot(
        prim_path="/World/NovaCarter",
        name="nova_carter",
        wheel_dof_names=["joint_wheel_left", "joint_wheel_right"],
        create_robot=True,
        usd_path=assets_root + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd",
        position=np.array([ROBOT_SPAWN_XY[0], ROBOT_SPAWN_XY[1], ROBOT_SPAWN_Z]),
        orientation=spawn_orientation,
    )
)

controller = DifferentialController(name="ctrl", wheel_radius=0.14, wheel_base=0.4)

# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

ONBOARD_OFFSET = np.array([0.20, 0.0, 0.65])
FOLLOW_OFFSET  = np.array([1.5, -1.5, 1.2])
FOLLOW_ORIENT  = rot_utils.euler_angles_to_quats(np.array([0, 10, 135]), degrees=True)

W, H = CAM_RESOLUTION

onboard_cam = Camera(
    prim_path="/World/OnboardCam",
    position=np.array([ROBOT_SPAWN_XY[0], ROBOT_SPAWN_XY[1] + 0.2, ROBOT_SPAWN_Z + 0.65]),
    frequency=FPS,
    resolution=(W, H),
    orientation=spawn_orientation,
)
follow_cam = Camera(
    prim_path="/World/FollowCam",
    position=np.array([ROBOT_SPAWN_XY[0] + FOLLOW_OFFSET[0],
                       ROBOT_SPAWN_XY[1] + FOLLOW_OFFSET[1],
                       ROBOT_SPAWN_Z     + FOLLOW_OFFSET[2]]),
    frequency=FPS,
    resolution=(W, H),
    orientation=FOLLOW_ORIENT,
)

world.reset()
simulation_app.update()
onboard_cam.initialize()
follow_cam.initialize()

usd_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/OnboardCam"))
usd_cam.GetFocalLengthAttr().Set(5.0)
usd_cam.GetClippingRangeAttr().Set((0.01, 10000.0))

usd_follow_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/FollowCam"))
usd_follow_cam.GetFocalLengthAttr().Set(5.0)
usd_follow_cam.GetClippingRangeAttr().Set((0.01, 10000.0))

carter.set_world_pose(
    position=np.array([ROBOT_SPAWN_XY[0], ROBOT_SPAWN_XY[1], ROBOT_SPAWN_Z]),
    orientation=spawn_orientation,
)
print("[Sim] Robot orientation set to yaw=180° (facing -X west toward door)")

print("[Sim] Warming up renderer...")
for _ in range(15):
    world.step(render=True)

# ---------------------------------------------------------------------------
# RL policy — load ONNX session and VecNormalize stats (no PyTorch)
# ---------------------------------------------------------------------------

import onnxruntime as ort
from omni.physx import get_physx_scene_query_interface

_physx_qi = get_physx_scene_query_interface()

print(f"[RL] Loading ONNX policy from: {_onnx_path}")
if not os.path.isfile(_onnx_path):
    print(f"ERROR: ONNX file not found: {_onnx_path}")
    print("  Run:  python training/export_policy_onnx.py --checkpoint <best_model.zip>")
    simulation_app.close()
    sys.exit(1)

_onnx_session = ort.InferenceSession(_onnx_path, providers=["CPUExecutionProvider"])

# VecNormalize stats — fallback to identity (mean=0, var=1) if file missing
if os.path.isfile(_vecnorm_path):
    _vn = np.load(_vecnorm_path)
    _obs_mean = _vn["mean"].astype(np.float32)
    _obs_var  = _vn["var"].astype(np.float32)
    print(f"[RL] VecNorm stats loaded from: {_vecnorm_path}")
else:
    print(f"[RL] WARN: VecNorm file not found ({_vecnorm_path}) — using identity normalisation.")
    _obs_mean = np.zeros(N_RAYS + 6, dtype=np.float32)
    _obs_var  = np.ones( N_RAYS + 6, dtype=np.float32)


# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------

def _normalize_angle(a):
    while a >  math.pi: a -= 2.0 * math.pi
    while a < -math.pi: a += 2.0 * math.pi
    return a


def get_robot_yaw(quat):
    """Extract world yaw (radians) from quaternion [w, x, y, z]."""
    r = ScipyR.from_quat([quat[1], quat[2], quat[3], quat[0]])
    return r.as_euler("xyz", degrees=False)[2]


def get_lidar_obs(robot_xy, robot_z, robot_yaw):
    """Simulate LiDAR via PhysX raycasting — matches training observation."""
    rays = np.ones(N_RAYS, dtype=np.float32)
    for i, angle in enumerate(np.linspace(0, 2 * math.pi, N_RAYS, endpoint=False)):
        wa = angle + robot_yaw
        direction = (float(math.cos(wa)), float(math.sin(wa)), 0.0)
        origin = (float(robot_xy[0]), float(robot_xy[1]), float(robot_z + LIDAR_Z))
        hit = _physx_qi.raycast_closest(origin, direction, MAX_RANGE)
        if hit and hit.get("hit"):
            rays[i] = min(float(hit["distance"]) / MAX_RANGE, 1.0)
    return rays


def build_obs(robot_xy, robot_z, robot_yaw, lin_vel, ang_vel, goal_xy):
    """Build the 26-dim observation vector and apply VecNormalize."""
    lidar = get_lidar_obs(robot_xy, robot_z, robot_yaw)

    diff = goal_xy - robot_xy
    c, s = math.cos(robot_yaw), math.sin(robot_yaw)
    goal_body = np.array([c * diff[0] + s * diff[1],
                          -s * diff[0] + c * diff[1]], dtype=np.float32)

    target_yaw = math.atan2(diff[1], diff[0])
    heading_err = _normalize_angle(target_yaw - robot_yaw)
    goal_heading = np.array([math.cos(heading_err), math.sin(heading_err)], dtype=np.float32)

    fwd_vel  = np.array([lin_vel[0] * c + lin_vel[1] * s], dtype=np.float32)
    yaw_rate = np.array([ang_vel[2]], dtype=np.float32)

    obs = np.concatenate([lidar, goal_body, goal_heading, fwd_vel, yaw_rate])
    obs_norm = np.clip((obs - _obs_mean) / np.sqrt(_obs_var + 1e-8), -10.0, 10.0)
    return obs_norm[np.newaxis, :].astype(np.float32)   # shape (1, 26)


def rl_action(robot_xy, robot_z, robot_yaw, lin_vel, ang_vel, goal_xy):
    """Run one RL policy step; return [linear_cmd, angular_cmd]."""
    obs = build_obs(robot_xy, robot_z, robot_yaw, lin_vel, ang_vel, goal_xy)
    raw = _onnx_session.run(["action"], {"obs": obs})[0]   # shape (1, 2)
    return [float(raw[0, 0]) * ACTION_SCALE[0],
            float(raw[0, 1]) * ACTION_SCALE[1]]


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

scan_frames = []    # (frame_rgba, distance, step) tuples — approach scan frames
door_frames = []    # (frame_rgba, distance, step) tuples — door inspection frames

STATE_APPROACH = "APPROACH"
STATE_INSPECT  = "INSPECT"
STATE_DONE     = "DONE"

state     = STATE_APPROACH
step      = 0
door_held = 0
_first_step = True
MAX_STEPS = 1800    # 60 s safety cutoff

all_frames        = []   # (rgba, distance, state) — onboard, every step
all_follow_frames = []   # rgba only — follow cam, every step

print(f"\n[Sim] Robot spawned at XY={ROBOT_SPAWN_XY}, RL goal XY={DOOR_CENTER_XY}")
print(f"[Sim] Door centre XY={DOOR_CENTER_XY}  |  Phase switch at {TARGET_RADIUS} m\n")

while step < MAX_STEPS and state != STATE_DONE:
    pos, quat = carter.get_world_pose()
    robot_xy  = pos[:2]
    distance  = float(np.linalg.norm(robot_xy - DOOR_CENTER_XY))

    if _first_step:
        print(f"[Sim] Robot world pos: X={pos[0]:.2f} Y={pos[1]:.2f} Z={pos[2]:.2f}")
        print(f"[Sim] Initial distance to door: {distance:.2f} m")
        _first_step = False

    # Move robot fill light (1 m above robot)
    lp = pos + np.array([0.0, 0.0, 1.0])
    fill_xform.ClearXformOpOrder()
    fill_xform.AddTranslateOp().Set((float(lp[0]), float(lp[1]), float(lp[2])))

    # Move follow fill light
    flp = pos + FOLLOW_OFFSET * 0.5 + np.array([0.0, 0.0, 1.5])
    follow_fill_xform.ClearXformOpOrder()
    follow_fill_xform.AddTranslateOp().Set((float(flp[0]), float(flp[1]), float(flp[2])))

    # Attach onboard camera to robot
    r_obj = ScipyR.from_quat([quat[1], quat[2], quat[3], quat[0]])
    cam_world_pos = pos + r_obj.apply(ONBOARD_OFFSET)
    onboard_cam.set_world_pose(position=cam_world_pos, orientation=quat)

    # Follow camera tracks robot (fixed world offset)
    follow_cam.set_world_pose(
        position=pos + FOLLOW_OFFSET,
        orientation=FOLLOW_ORIENT,
    )

    # Drive command
    if state == STATE_APPROACH:
        lin_vel = carter.get_linear_velocity()
        ang_vel = carter.get_angular_velocity()
        robot_yaw = get_robot_yaw(quat)
        cmd = rl_action(robot_xy, pos[2], robot_yaw, lin_vel, ang_vel, DOOR_CENTER_XY)

        # Transition: robot is close enough — switch to inspection
        if distance <= TARGET_RADIUS:
            state = STATE_INSPECT
            print(f"\n[Sim] *** PHASE SWITCH — DOOR INSPECTION *** (distance={distance:.2f} m)\n")
            cmd = [0.0, 0.0]

        # Collect approach scan frames
        elif step % SCAN_INTERVAL == 0:
            rgba = onboard_cam.get_rgba()
            if rgba is not None:
                frame = np.clip(rgba, 0, 255).astype(np.uint8)
                scan_frames.append((frame.copy(), distance, step))
                print(f"[Sim]   step {step:4d} | dist {distance:5.2f} m | "
                      f"[ANOMALY SCAN] frame #{len(scan_frames)} queued")

    elif state == STATE_INSPECT:
        cmd = [0.0, 0.0]   # robot stopped at door

        if door_held % DOOR_CHECK_GAP == 0 and len(door_frames) < N_DOOR_CHECKS:
            rgba = onboard_cam.get_rgba()
            if rgba is not None:
                frame = np.clip(rgba, 0, 255).astype(np.uint8)
                door_frames.append((frame.copy(), distance, step))
                print(f"[Sim]   step {step:4d} | dist {distance:5.2f} m | "
                      f"[DOOR INSPECT] frame #{len(door_frames)}/{N_DOOR_CHECKS} queued")
                if len(door_frames) >= N_DOOR_CHECKS:
                    state = STATE_DONE

        door_held += 1

    carter.apply_wheel_actions(controller.forward(command=cmd))
    world.step(render=True)

    rgba = onboard_cam.get_rgba()
    if rgba is not None:
        all_frames.append((np.clip(rgba, 0, 255).astype(np.uint8), distance, state))

    fl = follow_cam.get_rgba()
    if fl is not None:
        all_follow_frames.append(np.clip(fl, 0, 255).astype(np.uint8))

    if step % 60 == 0:
        print(f"[Sim]   step {step}/{MAX_STEPS} | dist {distance:.2f} m | state {state}")

    step += 1

world.stop()
print(f"\n[Sim] Complete — {len(scan_frames)} scan frames, {len(door_frames)} door frames")
print(f"[Sim] Onboard frames: {len(all_frames)}  |  Follow frames: {len(all_follow_frames)}")

# ---------------------------------------------------------------------------
# Save frame cache — Phase 2 (VLM + video) runs in a separate process because
# simulation_app.close() calls os._exit(0), killing everything after it.
# ---------------------------------------------------------------------------

_cache_path = os.path.join(_project_dir, f"door_demo_cache_{JOB_ID}.npz")
print(f"[Sim] Saving frame cache → {_cache_path}")

# Unpack list-of-tuples into separate arrays for npz storage
_sf_rgba  = np.array([f[0] for f in scan_frames], dtype=np.uint8) if scan_frames else np.zeros((0,1,1,4), dtype=np.uint8)
_sf_dist  = np.array([f[1] for f in scan_frames], dtype=np.float32)
_sf_step  = np.array([f[2] for f in scan_frames], dtype=np.int32)
_df_rgba  = np.array([f[0] for f in door_frames], dtype=np.uint8) if door_frames else np.zeros((0,1,1,4), dtype=np.uint8)
_df_dist  = np.array([f[1] for f in door_frames], dtype=np.float32)
_df_step  = np.array([f[2] for f in door_frames], dtype=np.int32)
_af_rgba  = np.array([f[0] for f in all_frames], dtype=np.uint8)  if all_frames  else np.zeros((0,1,1,4), dtype=np.uint8)
_af_dist  = np.array([f[1] for f in all_frames], dtype=np.float32)
_af_state = np.array([f[2] for f in all_frames], dtype="U16")
_ff_rgba  = np.array(all_follow_frames, dtype=np.uint8) if all_follow_frames else np.zeros((0,1,1,4), dtype=np.uint8)

np.savez_compressed(
    _cache_path,
    scan_rgba=_sf_rgba, scan_dist=_sf_dist, scan_step=_sf_step,
    door_rgba=_df_rgba, door_dist=_df_dist, door_step=_df_step,
    all_rgba=_af_rgba,  all_dist=_af_dist,  all_state=_af_state,
    follow_rgba=_ff_rgba,
    door_angle=np.float32(DOOR_ANGLE_DEG),
    run_stamp=np.array(RUN_STAMP, dtype="U32"),
    adapter_path=np.array(ADAPTER_PATH, dtype="U256"),
    door_center_xy=DOOR_CENTER_XY,
    target_radius=np.float32(TARGET_RADIUS),
)
print(f"[Sim] Cache saved ({len(all_frames)} onboard frames, {len(all_follow_frames)} follow frames).")

# simulation_app.close() calls os._exit(0) — nothing after this runs.
simulation_app.close()

