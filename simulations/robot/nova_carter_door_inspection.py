#!/usr/bin/env python3
"""
nova_carter_door_inspection.py — Two-phase VLM door inspection demo.

The Nova Carter robot spawns south of the training door, drives toward it,
and runs two distinct VLM prompting phases:

  PHASE 1 — ANOMALY SCAN (approach, distance > TARGET_RADIUS)
    Base Qwen2-VL-2B-Instruct with no adapter.
    Prompt: general scene description + anomaly / safety hazard detection.
    Demonstrates why a VLM beats a CNN — it uses pre-trained world knowledge
    to flag hazards (fire, water, obstacles, people) without task-specific
    training.

  PHASE 2 — DOOR INSPECTION (≤ TARGET_RADIUS from door)
    Fine-tuned model with LoRA + visual projector adapter loaded.
    Prompt: "Is this door open or closed?"
    Demonstrates the domain-adapted classifier achieving 93.3% accuracy.

The door state (OPEN / CLOSED) is set manually before running this script.
Run twice — once with the door closed, once with the door open — to compare.

Usage:
  sbatch submit_isaac_job.sh simulations/robot/nova_carter_door_inspection.py

Outputs (never overwrites — each run gets a unique timestamp):
  videos/door_inspection_YYYYMMDD_HHMMSS.mp4          — annotated onboard video
  data/door_inspection_frames_YYYYMMDD_HHMMSS/        — one PNG per VLM call
    scan_01_dist8.4m.png  ... (approach anomaly-scan frames)
    door_01_dist2.1m.png  ... (door inspection frames)
  data/door_inspection_YYYYMMDD_HHMMSS.txt            — full VLM prediction log

Config to tune:
  ROBOT_SPAWN_XY — start position (south of door, same floor)
  TARGET_RADIUS  — metres at which phase switches to door inspection
  ADAPTER_PATH   — path to fine-tuned LoRA adapter directory
"""

import math
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Door state — set this before running
# ---------------------------------------------------------------------------
# 0.0  = fully closed  (≤ 5° → CLOSED class)
# 90.0 = fully open
DOOR_ANGLE_DEG = 0.0

# Timestamp used for ALL outputs — guarantees nothing is overwritten across runs
RUN_STAMP = time.strftime("%Y%m%d_%H%M%S")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Door we trained on: Doors_13606713 at X=-20.48, Y=80.15 (from door_collection.py)
DOOR_CENTER_XY  = np.array([-20.48, 80.15])

# Robot spawns 4 m EAST of the door, inside the inner corridor.
# Training data was collected from X=-18.48 (2m east), robot facing -X (west).
# Spawning further east (X>-12) hits a glass partition — the inner corridor
# clear zone starts around X=-13.  Spawn at X=-16.5 (4 m from door, 1.5 m
# behind the training camera position) so the robot has a short approach run.
ROBOT_SPAWN_XY  = np.array([-16.5, 80.15])
ROBOT_SPAWN_Z   = 27.3          # Floor 4 slab Z=26.8 + 0.5 m clearance

# Switch from anomaly scan → door inspection when robot is this close
TARGET_RADIUS   = 2.0           # metres

# Approach stops this close to the door centre.
# Training cameras were at exactly 2 m east — stop at 2.0 m to match.
STOP_RADIUS     = 2.0           # metres — matches training camera distance

# Robot drive params
APPROACH_SPEED  = 0.35          # m/s linear (conservative on Floor 4)
ANGULAR_GAIN    = 2.5           # proportional gain for heading correction

# How often to call the VLM during approach (every N sim steps at 30 Hz)
SCAN_INTERVAL   = 45            # ~1.5 s between anomaly-scan queries
N_DOOR_CHECKS   = 5             # VLM calls once stopped at the door
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

# ---------------------------------------------------------------------------
# Isaac Sim initialisation — must be first import
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

_building_usd = os.path.join(_project_dir, "innovation_center_finalV1.usd")
if not os.path.isfile(_building_usd):
    print(f"ERROR: Building USD not found: {_building_usd}")
    simulation_app.close()
    sys.exit(1)

print(f"[Sim] Loading building: {_building_usd}")
stage = omni.usd.get_context().get_stage()
stage.GetRootLayer().subLayerPaths.insert(0, _building_usd)

# ---------------------------------------------------------------------------
# Door angle control — same matrix-rewrite technique as door_static_collection.py
# ---------------------------------------------------------------------------

_DOOR_BASE = (
    "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026"
    "/Geometry/Doors/RS___Curtain_Wall___Type_FG___New/New"
)
_DOOR_PATH = f"{_DOOR_BASE}/Doors_13606713"

# Child mesh origins (cm) from door_rotation_diagnostic run 238045
_DOOR_MESHES = [
    {"path": _DOOR_PATH + "/AL_375207",       "ox": -2047.728271484375, "oy": 8014.9365234375,  "oz": 2677.16015625},
    {"path": _DOOR_PATH + "/Glass_GL_1_18439","ox": -2047.728271484375, "oy": 8014.9365234375,  "oz": 2702.559814453125},
]
_HINGE_X_CM = -2047.728271484375
_HINGE_Y_CM =  7969.07   # world Y-min 79.6907 m × 100 (south edge hinge)


def _door_matrix(ox_cm, oy_cm, oz_cm, angle_deg):
    """4×4 USD row-vector transform for rotating a door mesh around the hinge."""
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
# Hide vegetation — trees in the atrium block the follow camera view.
# Traverse the stage and invisibilise any prim whose name contains a
# vegetation keyword.  Runs on the already-loaded stage, no extra cost.
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

# Follow-cam fill light — travels with follow camera (inverse-square falloff
# makes the robot fill light too dim at 6 m, so a separate source is needed)
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
# Robot — spawn facing +Y (toward door)
# ---------------------------------------------------------------------------

# Yaw=180° rotates robot from default +X heading to -X (west, toward door).
# Training data was collected with robot facing -X at the same Y corridor.
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
# Onboard camera (attached to robot each step)
# ---------------------------------------------------------------------------

ONBOARD_OFFSET = np.array([0.20, 0.0, 0.65])   # front-centre of robot, 0.65 m up

# Follow cam: 45° behind the robot — offset equally in +X (east/behind) and
# -Y (south/right-side), 1.2 m up.  Robot faces -X, so this gives a cinematic
# rear-quarter view.  Camera yaw = 135° so it points back toward the robot
# (northwest in world space) with a slight 10° downward pitch.
# Total horizontal distance ≈ 2 m; stays well inside the inner corridor.
FOLLOW_OFFSET = np.array([1.5, -1.5, 1.2])
FOLLOW_ORIENT = rot_utils.euler_angles_to_quats(np.array([0, 10, 135]), degrees=True)

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
                       ROBOT_SPAWN_Z    + FOLLOW_OFFSET[2]]),
    frequency=FPS,
    resolution=(W, H),
    orientation=FOLLOW_ORIENT,
)

world.reset()
simulation_app.update()
onboard_cam.initialize()
follow_cam.initialize()

# Wide FOV
usd_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/OnboardCam"))
usd_cam.GetFocalLengthAttr().Set(5.0)
usd_cam.GetClippingRangeAttr().Set((0.01, 10000.0))

usd_follow_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/FollowCam"))
usd_follow_cam.GetFocalLengthAttr().Set(5.0)
usd_follow_cam.GetClippingRangeAttr().Set((0.01, 10000.0))

# Apply robot orientation AFTER world.reset() — the constructor `orientation`
# param is not reliably applied by Isaac Sim's WheeledRobot after reset().
# Setting it explicitly here guarantees the robot faces +Y toward the door.
carter.set_world_pose(
    position=np.array([ROBOT_SPAWN_XY[0], ROBOT_SPAWN_XY[1], ROBOT_SPAWN_Z]),
    orientation=spawn_orientation,
)
print(f"[Sim] Robot orientation set to yaw=180° (facing -X west toward door)")

# Renderer warmup
print("[Sim] Warming up renderer...")
for _ in range(15):
    world.step(render=True)

# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------

def get_robot_yaw(quat):
    """Extract world yaw (radians) from quaternion [w, x, y, z]."""
    r = ScipyR.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy: [x,y,z,w]
    return r.as_euler("xyz", degrees=False)[2]


def normalize_angle(a):
    """Wrap angle to [-pi, pi]."""
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


def steer_toward_door(robot_xy, robot_quat, distance):
    """
    Proportional heading controller that steers the robot toward DOOR_CENTER_XY.
    Returns [linear_speed, angular_speed] for DifferentialController.forward().
    """
    if distance <= STOP_RADIUS:
        return [0.0, 0.0]

    direction    = DOOR_CENTER_XY - robot_xy
    target_yaw   = math.atan2(direction[1], direction[0])
    current_yaw  = get_robot_yaw(robot_quat)
    yaw_error    = normalize_angle(target_yaw - current_yaw)

    angular  = float(np.clip(ANGULAR_GAIN * yaw_error, -2.0, 2.0))
    # Slow down proportionally if heading error is large
    linear   = APPROACH_SPEED * max(0.0, 1.0 - abs(yaw_error) / math.pi)
    return [linear, angular]

# ---------------------------------------------------------------------------
# Simulation loop — collect frames while robot drives to door
# ---------------------------------------------------------------------------

# Stored data for VLM phase
scan_frames    = []    # (frame_rgba, distance, step) tuples — approach scan frames
door_frames    = []    # (frame_rgba, distance, step) tuples — door inspection frames

STATE_APPROACH = "APPROACH"
STATE_INSPECT  = "INSPECT"
STATE_DONE     = "DONE"

state         = STATE_APPROACH
step          = 0
door_held     = 0       # steps held at door for inspection frames
_first_step   = True
MAX_STEPS     = 1800    # 60 s safety cutoff

print(f"\n[Sim] Robot spawned at XY={ROBOT_SPAWN_XY}, facing door at XY={DOOR_CENTER_XY}")
print(f"[Sim] Phase switch at {TARGET_RADIUS} m | Stop at {STOP_RADIUS} m\n")

all_frames        = []   # (rgba, distance, state) — onboard, every step
all_follow_frames = []   # rgba only — follow cam, every step (unannotated)

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

    # Move follow fill light (halfway between follow cam and robot, slightly above)
    flp = pos + FOLLOW_OFFSET * 0.5 + np.array([0.0, 0.0, 1.5])
    follow_fill_xform.ClearXformOpOrder()
    follow_fill_xform.AddTranslateOp().Set((float(flp[0]), float(flp[1]), float(flp[2])))

    # Attach onboard camera to robot
    r_obj = ScipyR.from_quat([quat[1], quat[2], quat[3], quat[0]])
    cam_world_pos = pos + r_obj.apply(ONBOARD_OFFSET)
    onboard_cam.set_world_pose(position=cam_world_pos, orientation=quat)

    # Follow camera tracks robot (fixed world offset, fixed orientation)
    follow_cam.set_world_pose(
        position=pos + FOLLOW_OFFSET,
        orientation=FOLLOW_ORIENT,
    )

    # Drive command
    if state == STATE_APPROACH:
        cmd = steer_toward_door(robot_xy, quat, distance)

        # Transition: close enough — switch to inspection
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
        cmd = [0.0, 0.0]   # robot is stopped

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
# Phase 2 — VLM Inference (base model for scan, fine-tuned for door check)
# ---------------------------------------------------------------------------

print("\n[VLM] Loading dependencies...")
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
attn_impl = "flash_attention_2" if (
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
) else "sdpa"

def rgba_to_pil(rgba):
    return Image.fromarray(rgba[:, :, :3].astype(np.uint8), "RGB")


def load_font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
              "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except (IOError, AttributeError):
            continue
    return None


def run_vlm(model, processor, pil_image, prompt):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text",  "text": prompt},
        ],
    }]
    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_image], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    generated = out_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


# ------------------------------------------------------------------
# 2a — ANOMALY SCAN (base model, no adapter)
# ------------------------------------------------------------------
print(f"\n[VLM] Loading BASE model for anomaly scan ({len(scan_frames)} frames)...")
t0 = time.time()
processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    VLM_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation=attn_impl,
)
base_model.eval()
print(f"[VLM] Base model loaded in {time.time()-t0:.1f}s")

scan_results = []   # (distance, step, response)
for i, (frame, dist, step_n) in enumerate(scan_frames):
    pil = rgba_to_pil(frame)
    t_start = time.time()
    resp = run_vlm(base_model, processor, pil, ANOMALY_PROMPT)
    elapsed = time.time() - t_start
    scan_results.append((dist, step_n, resp))
    print(f"\n[VLM] ANOMALY SCAN {i+1}/{len(scan_frames)}  dist={dist:.2f} m  ({elapsed:.1f}s)")
    print(f"      Response: {resp}")

# Free base model before loading fine-tuned
del base_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ------------------------------------------------------------------
# 2b — DOOR INSPECTION (fine-tuned model + adapter)
# ------------------------------------------------------------------

door_results = []

if not door_frames:
    print("\n[VLM] WARNING: No door frames collected — robot did not reach door.")
    print("  Check STOP_RADIUS, MAX_STEPS, and building geometry.")
    print("  Skipping door inspection phase.")
else:
    print(f"\n[VLM] Loading FINE-TUNED model for door inspection ({len(door_frames)} frames)...")
    sys.stdout.flush()

    try:
        if not os.path.isdir(ADAPTER_PATH):
            print(f"[VLM] WARN: adapter not found at {ADAPTER_PATH} — using base model for door check")
            sys.stdout.flush()
            fine_model = Qwen2VLForConditionalGeneration.from_pretrained(
                VLM_MODEL_ID, torch_dtype=torch.float16, device_map="auto", attn_implementation=attn_impl,
            )
            fine_model.eval()
        else:
            from peft import PeftModel
            t0 = time.time()
            print(f"[VLM] Loading base model weights...")
            sys.stdout.flush()
            fine_model = Qwen2VLForConditionalGeneration.from_pretrained(
                VLM_MODEL_ID, torch_dtype=torch.float16, device_map="auto", attn_implementation=attn_impl,
            )
            print(f"[VLM] Attaching LoRA adapter from {ADAPTER_PATH}...")
            sys.stdout.flush()
            fine_model = PeftModel.from_pretrained(fine_model, ADAPTER_PATH)
            # Skip merge_and_unload — use PeftModel directly for inference.
            # merge_and_unload() can crash with modules_to_save adapters in some
            # peft versions; inference on the wrapped PeftModel is identical.
            fine_model.eval()
            print(f"[VLM] Fine-tuned model ready in {time.time()-t0:.1f}s")
            sys.stdout.flush()
    except Exception as _e:
        import traceback
        print(f"\n[VLM] ERROR loading fine-tuned model: {_e}")
        traceback.print_exc()
        sys.stdout.flush()
        print("[VLM] Falling back to base model for door check...")
        sys.stdout.flush()
        fine_model = Qwen2VLForConditionalGeneration.from_pretrained(
            VLM_MODEL_ID, torch_dtype=torch.float16, device_map="auto", attn_implementation=attn_impl,
        )
        fine_model.eval()

    for i, (frame, dist, step_n) in enumerate(door_frames):
        pil = rgba_to_pil(frame)
        t_start = time.time()
        resp = run_vlm(fine_model, processor, pil, DOOR_PROMPT)
        elapsed = time.time() - t_start
        parsed = "OPEN" if "[OPEN]" in resp.upper() else "CLOSED" if "[CLOSED]" in resp.upper() else "UNKNOWN"
        door_results.append((dist, step_n, resp, parsed))
        print(f"\n[VLM] DOOR CHECK {i+1}/{N_DOOR_CHECKS}  dist={dist:.2f} m  ({elapsed:.1f}s)")
        print(f"      Response : {resp}")
        print(f"      Parsed   : {parsed}")

    del fine_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Majority vote on door state
door_votes = [r[3] for r in door_results]
final_state = max(set(door_votes), key=door_votes.count) if door_votes else "UNKNOWN"
print(f"\n[VLM] FINAL DOOR STATE: {final_state}  "
      f"(votes: OPEN={door_votes.count('OPEN')}, CLOSED={door_votes.count('CLOSED')})")

# ---------------------------------------------------------------------------
# Log to file
# ---------------------------------------------------------------------------

# Output paths — all stamped with RUN_STAMP so nothing is ever overwritten
frames_out_dir   = os.path.join(_project_dir, "data",   f"door_inspection_frames_{RUN_STAMP}")
log_path         = os.path.join(_project_dir, "data",   f"door_inspection_{RUN_STAMP}.txt")
video_path       = os.path.join(_project_dir, "videos", f"door_inspection_{RUN_STAMP}.mp4")
follow_video_path= os.path.join(_project_dir, "videos", f"door_inspection_follow_{RUN_STAMP}.mp4")
os.makedirs(frames_out_dir, exist_ok=True)
os.makedirs(os.path.join(_project_dir, "videos"), exist_ok=True)

# ---------------------------------------------------------------------------
# Save individual annotated frames for each VLM call (inspectable immediately)
# ---------------------------------------------------------------------------

def draw_label(pil_img, header, body, dist, badge_color, badge_text):
    """Overlay phase label, distance, and VLM response text onto a PIL image."""
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(pil_img)
    w, h = pil_img.size
    font_sm = load_font(12)
    font_bd = load_font(11)

    # Bottom caption bar
    import textwrap
    lines  = textwrap.wrap(body, width=80)[:8]
    line_h = 16
    bar_h  = len(lines) * line_h + 12
    bar_y  = h - bar_h
    draw.rectangle([(0, bar_y), (w, h)], fill=(10, 10, 10))
    y = bar_y + 6
    for line in lines:
        draw.text((8, y), line, fill=(230, 230, 230), font=font_sm)
        y += line_h

    # Phase badge top-left
    draw.rectangle([(0, 0), (200, 22)], fill=badge_color)
    draw.text((6, 4), f"◉ {header}", fill=(255, 255, 255), font=font_bd)

    # Door verdict badge top-right (for door check frames)
    if badge_text:
        draw.rectangle([(w - 140, 0), (w, 22)], fill=badge_color)
        draw.text((w - 134, 4), badge_text, fill=(255, 255, 255), font=font_bd)

    # Distance
    draw.text((8, 28), f"Door dist: {dist:.1f} m", fill=(200, 200, 200), font=font_bd)
    return pil_img


# Save scan frames
for i, (dist, step_n, resp) in enumerate(scan_results):
    frame_rgba = scan_frames[i][0]
    pil = rgba_to_pil(frame_rgba)
    pil = draw_label(pil,
                     header="ANOMALY SCAN",
                     body=resp,
                     dist=dist,
                     badge_color=(180, 90, 0),
                     badge_text=None)
    fname = os.path.join(frames_out_dir, f"scan_{i+1:02d}_dist{dist:.1f}m.png")
    pil.save(fname)
    print(f"[Frames] scan frame saved → {fname}")

# Save door inspection frames
for i, (dist, step_n, resp, parsed) in enumerate(door_results):
    frame_rgba = door_frames[i][0]
    pil = rgba_to_pil(frame_rgba)
    badge_color = (30, 160, 60) if parsed == "OPEN" else \
                  (180, 30, 30) if parsed == "CLOSED" else (80, 40, 140)
    pil = draw_label(pil,
                     header="DOOR INSPECTION",
                     body=resp,
                     dist=dist,
                     badge_color=badge_color,
                     badge_text=f"DOOR: {parsed}")
    fname = os.path.join(frames_out_dir, f"door_{i+1:02d}_dist{dist:.1f}m_{parsed}.png")
    pil.save(fname)
    print(f"[Frames] door frame saved → {fname}")

print(f"[Frames] All VLM frames → {frames_out_dir}/")

with open(log_path, "w") as lf:
    lf.write(f"Door Inspection Log — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lf.write(f"Door centre XY: {DOOR_CENTER_XY}\n")
    lf.write(f"Door angle set: {DOOR_ANGLE_DEG}° ({'OPEN' if DOOR_ANGLE_DEG > 5 else 'CLOSED'})\n")
    lf.write(f"Target radius:  {TARGET_RADIUS} m\n\n")
    lf.write("=== ANOMALY SCAN (base model) ===\n")
    for dist, step_n, resp in scan_results:
        lf.write(f"  step {step_n:4d} | dist {dist:.2f} m\n  {resp}\n\n")
    lf.write("=== DOOR INSPECTION (fine-tuned) ===\n")
    for dist, step_n, resp, parsed in door_results:
        lf.write(f"  step {step_n:4d} | dist {dist:.2f} m | {parsed}\n  {resp}\n\n")
    lf.write(f"=== FINAL VERDICT: {final_state} ===\n")
print(f"[Log] Saved → {log_path}")

# ---------------------------------------------------------------------------
# Phase 3 — Annotated video
# ---------------------------------------------------------------------------

print("\n[Video] Writing annotated video...")

import subprocess, shutil, textwrap

def find_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    local = os.path.expanduser("~/bin/ffmpeg")
    if os.path.isfile(local) and os.access(local, os.X_OK):
        return local
    return shutil.which("ffmpeg")


font_main  = load_font(12)   # smaller to fit more VLM text on screen
font_badge = load_font(11)

# Build a lookup: step → (phase, caption, parsed_class)
# For each rendered frame, find the nearest preceding VLM result
scan_step_map = {s: (r,) for (_, s, r) in scan_results}
door_step_map = {s: (r, p) for (_, s, r, p) in door_results}

def get_annotation(frame_state, frame_dist, frame_step):
    """Return (phase_label, caption, badge_color, badge_text)."""
    if frame_state == STATE_APPROACH:
        # Find most recent scan result at or before this step
        candidates = [(s, r) for (_, s, r) in scan_results if s <= frame_step]
        caption = candidates[-1][1] if candidates else "Scanning scene..."
        return ("ANOMALY SCAN", caption, (200, 120, 0), "SCANNING")

    else:  # INSPECT or DONE
        # Find most recent door result
        candidates = [(p, r) for (_, s, r, p) in door_results if s <= frame_step]
        if candidates:
            parsed, resp = candidates[-1]
            badge_color = (30, 160, 60) if parsed == "OPEN" else \
                          (180, 30, 30) if parsed == "CLOSED" else (80, 40, 140)
            badge_text  = f"DOOR: {parsed}"
            return ("DOOR INSPECTION", resp, badge_color, badge_text)
        return ("DOOR INSPECTION", "Inspecting door...", (80, 40, 140), "DOOR: ?")


def annotate_frame(rgba, frame_state, frame_dist, frame_step, frame_num, total):
    phase, caption, badge_color, badge_text = get_annotation(frame_state, frame_dist, frame_step)
    img  = Image.fromarray(rgba[:, :, :3].astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(img)
    h, w = rgba.shape[:2]

    # Caption bar at bottom — show full VLM response, up to 8 lines
    lines  = textwrap.wrap(caption, width=80)[:8]
    line_h = 16
    bar_h  = len(lines) * line_h + 12
    bar_y  = h - bar_h
    draw.rectangle([(0, bar_y), (w, h)], fill=(10, 10, 10))
    y = bar_y + 6
    for line in lines:
        draw.text((8, y), line, fill=(230, 230, 230), font=font_main)
        y += line_h

    # Phase badge (top left)
    phase_color = (180, 90, 0) if phase == "ANOMALY SCAN" else badge_color
    draw.rectangle([(0, 0), (190, 22)], fill=phase_color)
    draw.text((6, 4), f"◉ {phase}", fill=(255, 255, 255), font=font_badge)

    # Door state badge (top right if in inspection)
    if phase == "DOOR INSPECTION" and "?" not in badge_text:
        draw.rectangle([(w - 130, 0), (w, 22)], fill=badge_color)
        draw.text((w - 124, 4), badge_text, fill=(255, 255, 255), font=font_badge)

    # Distance indicator
    draw.text((8, 28), f"Door dist: {frame_dist:.1f} m", fill=(200, 200, 200), font=font_badge)

    # Frame counter
    draw.text((w - 70, 28), f"{frame_num}/{total}", fill=(150, 150, 150), font=font_badge)

    return np.array(img)


ffmpeg_exe = find_ffmpeg()
videos_dir = os.path.join(_project_dir, "videos")
os.makedirs(videos_dir, exist_ok=True)
# video_path already defined above with RUN_STAMP

if ffmpeg_exe and all_frames:
    cmd = [
        ffmpeg_exe, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "rgb24", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        video_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    total_frames = len(all_frames)
    for i, (rgba, dist, fstate) in enumerate(all_frames):
        annotated = annotate_frame(rgba, fstate, dist, i, i + 1, total_frames)
        proc.stdin.write(annotated.tobytes())
        if (i + 1) % 60 == 0:
            print(f"[Video]   {i+1}/{total_frames} frames written...")

    proc.stdin.close()
    proc.wait()
    if proc.returncode == 0:
        print(f"[Video] ✓ {video_path}")
    else:
        print(f"[Video] ffmpeg error: {proc.stderr.read().decode()[:200]}")
else:
    print("[Video] ffmpeg not found or no frames — skipping onboard video")

# Follow camera video (unannotated — clean third-person view)
if ffmpeg_exe and all_follow_frames:
    cmd2 = [
        ffmpeg_exe, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "rgb24", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        follow_video_path,
    ]
    proc2 = subprocess.Popen(cmd2, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    total_follow = len(all_follow_frames)
    for i, rgba in enumerate(all_follow_frames):
        proc2.stdin.write(rgba[:, :, :3].tobytes())
        if (i + 1) % 60 == 0:
            print(f"[Video]   follow {i+1}/{total_follow} frames written...")
    proc2.stdin.close()
    proc2.wait()
    if proc2.returncode == 0:
        print(f"[Video] ✓ {follow_video_path}")
    else:
        print(f"[Video] follow ffmpeg error: {proc2.stderr.read().decode()[:200]}")
else:
    print("[Video] ffmpeg not found or no follow frames — skipping follow video")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("  DOOR INSPECTION SUMMARY")
print("="*60)
print(f"  Approach frames scanned : {len(scan_results)}")
print(f"  Door check frames       : {len(door_results)}")
print(f"  Final door verdict      : {final_state}")
print(f"  Vote breakdown          : OPEN={door_votes.count('OPEN')}  "
      f"CLOSED={door_votes.count('CLOSED')}  "
      f"UNKNOWN={door_votes.count('UNKNOWN')}")
print(f"  Frames (inspect now)    : {frames_out_dir}/")
print(f"  Log                     : {log_path}")
print(f"  Onboard video           : {video_path}")
print(f"  Follow video            : {follow_video_path}")
print("="*60)

# ---------------------------------------------------------------------------
# Close sim (triggers os._exit internally — must be last)
# ---------------------------------------------------------------------------

simulation_app.close()
