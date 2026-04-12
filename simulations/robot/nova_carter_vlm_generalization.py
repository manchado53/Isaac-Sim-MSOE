#!/usr/bin/env python3
"""
nova_carter_vlm_generalization.py — VLM generalization test on an unseen door.

Tests whether the fine-tuned Qwen2-VL door classifier (trained exclusively on
Doors_13606713 at X=-20.48, Y=80.15) can generalize to a structurally similar
but visually distinct door at a different location on Floor 4.

Target door: Doors_13185454 at X=-41.41, Y=82.60 (north area of Floor 4).
This door is the same Revit family (RS___Curtain_Wall___Type_FG___New) as the
training door but was NEVER seen during data collection or fine-tuning.

Only the fine-tuned model is run (no anomaly scan phase) to isolate the
generalization question. The robot approaches the new door, stops at 2 m, and
runs N_DOOR_CHECKS VLM queries with the domain-adapted classifier.

Door angle:
  Set DOOR_ANGLE_DEG before running. 0.0 = closed, 45.0 or 90.0 = open.
  The script uses dynamic child-mesh discovery so it works on any door without
  hard-coded mesh paths. The hinge is assumed to be at the minimum-Y edge of
  the child mesh origins (same side as Doors_13606713 — valid for all curtain
  wall doors in this building that open northward).

  NOTE: If the rotation looks wrong for this door, run door_rotation_diagnostic.py
  targeting this door to get the exact hinge coordinates and hard-code them.

Usage:
  sbatch submit_isaac_job.sh simulations/robot/nova_carter_vlm_generalization.py

Outputs (never overwrites — each run gets a unique timestamp):
  videos/vlm_generalization_YYYYMMDD_HHMMSS.mp4       — annotated onboard video
  videos/vlm_generalization_follow_YYYYMMDD_HHMMSS.mp4 — follow-cam video
  data/vlm_generalization_frames_YYYYMMDD_HHMMSS/      — one PNG per VLM call
  data/vlm_generalization_YYYYMMDD_HHMMSS.txt          — full VLM prediction log
"""

import math
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Door state — set this before running
# ---------------------------------------------------------------------------
# 0.0  = fully closed  (≤ 5° → model should predict CLOSED)
# 45.0 = half open
# 90.0 = fully open    (model should predict OPEN)
DOOR_ANGLE_DEG = 0.0

# Timestamp used for ALL outputs
RUN_STAMP = time.strftime("%Y%m%d_%H%M%S")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Generalization target: Doors_13185454 — NEVER seen during training.
# Same Revit curtain-wall family as training door (Doors_13606713) but
# located on the north side of the building, 21 m west of the training door.
DOOR_CENTER_XY  = np.array([-41.41, 82.60])

_DOOR_BASE = (
    "/World/Building/_22_129___MSOE_Engineering_Building___A___2_25_2026"
    "/Geometry/Doors/RS___Curtain_Wall___Type_FG___New/New"
)
TARGET_DOOR_PATH = f"{_DOOR_BASE}/Doors_13185454"

# Robot spawns 4.9 m EAST of the new door (X=-36.5), facing west (yaw=180°).
# This matches the approach direction used during training data collection.
# Adjust if building geometry blocks the approach path.
ROBOT_SPAWN_XY  = np.array([-36.5, 82.60])
ROBOT_SPAWN_Z   = 27.3          # Floor 4 slab Z=26.8 + 0.5 m clearance

# Robot stops and runs VLM when this close to the door
TARGET_RADIUS   = 2.0           # metres (matches training camera distance)
STOP_RADIUS     = 2.0           # metres

# Drive params
APPROACH_SPEED  = 0.35          # m/s
ANGULAR_GAIN    = 2.5           # proportional gain for heading correction

# VLM queries once stopped at door
N_DOOR_CHECKS   = 5
DOOR_CHECK_GAP  = 20            # sim steps between captures

# Camera
CAM_RESOLUTION  = (640, 480)
FPS             = 15

# Paths
_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
ADAPTER_PATH = os.path.join(_project_dir, "data", "qwen2vl_door_lora")
VLM_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# ---------------------------------------------------------------------------
# Prompt (same as training/inspection — ensures fair comparison)
# ---------------------------------------------------------------------------

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

_building_usd = os.path.join(_project_dir, "MSOEBuildingV4_physics.usd")
if not os.path.isfile(_building_usd):
    print(f"ERROR: Building USD not found: {_building_usd}")
    simulation_app.close()
    sys.exit(1)

print(f"[Sim] Loading building: {_building_usd}")
stage = omni.usd.get_context().get_stage()
stage.GetRootLayer().subLayerPaths.insert(0, _building_usd)

# ---------------------------------------------------------------------------
# Door angle control — dynamically discover child meshes + hinge
# ---------------------------------------------------------------------------

def _discover_door_meshes(stg, door_prim_path):
    """
    Walk children of door_prim_path and collect any prim that has an
    xformOp:transform op.  Returns list of dicts {path, ox, oy, oz} in cm.
    """
    door_prim = stg.GetPrimAtPath(door_prim_path)
    if not door_prim.IsValid():
        print(f"[Sim] WARN: door prim not found: {door_prim_path}")
        return []
    meshes = []
    for child in Usd.PrimRange(door_prim):
        if child.GetPath().pathString == door_prim_path:
            continue
        xformable = UsdGeom.Xformable(child)
        for op in xformable.GetOrderedXformOps():
            if op.GetOpName() == "xformOp:transform":
                m = op.Get()
                ox, oy, oz = float(m[3][0]), float(m[3][1]), float(m[3][2])
                meshes.append({"path": child.GetPath().pathString,
                               "ox": ox, "oy": oy, "oz": oz})
                break
    return meshes


def _door_matrix(ox_cm, oy_cm, oz_cm, hinge_x_cm, hinge_y_cm, angle_deg):
    """4×4 USD row-vector transform for rotating a mesh around the hinge."""
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    dx = ox_cm - hinge_x_cm
    dy = oy_cm - hinge_y_cm
    return Gf.Matrix4d(
         c,  s, 0, 0,
        -s,  c, 0, 0,
         0,  0, 1, 0,
        c * dx - s * dy + hinge_x_cm,
        s * dx + c * dy + hinge_y_cm,
        oz_cm, 1,
    )


def _set_door_angle(stg, door_prim_path, angle_deg):
    """Discover child meshes dynamically and apply rotation around computed hinge."""
    if angle_deg == 0.0:
        print(f"[Sim] Door angle = 0° — leaving door in default USD state")
        return

    meshes = _discover_door_meshes(stg, door_prim_path)
    if not meshes:
        print(f"[Sim] WARN: no child meshes found under {door_prim_path} — skipping rotation")
        return

    # Hinge X = shared X origin of child meshes (the fixed pillar edge)
    hinge_x_cm = meshes[0]["ox"]
    # Hinge Y = minimum Y among all mesh origins (south/fixed edge of door)
    hinge_y_cm = min(m["oy"] for m in meshes)

    print(f"[Sim] Door meshes discovered: {len(meshes)}")
    print(f"[Sim] Computed hinge: X={hinge_x_cm:.2f} cm, Y={hinge_y_cm:.2f} cm")

    for m in meshes:
        prim = stg.GetPrimAtPath(m["path"])
        if not prim.IsValid():
            print(f"[Sim] WARN: mesh prim not found: {m['path']}")
            continue
        for xop in UsdGeom.Xformable(prim).GetOrderedXformOps():
            if xop.GetOpName() == "xformOp:transform":
                xop.Set(_door_matrix(m["ox"], m["oy"], m["oz"],
                                     hinge_x_cm, hinge_y_cm, angle_deg))
                break


print(f"[Sim] Setting door angle to {DOOR_ANGLE_DEG}° "
      f"({'OPEN' if DOOR_ANGLE_DEG > 5 else 'CLOSED'})")
_set_door_angle(stage, TARGET_DOOR_PATH, DOOR_ANGLE_DEG)

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

# Yaw 180° → faces -X (west, toward door)
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

# 45° rear-quarter follow cam (same as inspection script)
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

# Apply orientation AFTER world.reset() — constructor param is unreliable
carter.set_world_pose(
    position=np.array([ROBOT_SPAWN_XY[0], ROBOT_SPAWN_XY[1], ROBOT_SPAWN_Z]),
    orientation=spawn_orientation,
)
print(f"[Sim] Robot spawned at XY={ROBOT_SPAWN_XY}, yaw=180° (facing west toward door)")

# Renderer warmup
print("[Sim] Warming up renderer...")
for _ in range(15):
    world.step(render=True)

# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------


def get_robot_yaw(quat):
    r = ScipyR.from_quat([quat[1], quat[2], quat[3], quat[0]])
    return r.as_euler("xyz", degrees=False)[2]


def normalize_angle(a):
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


def steer_toward_door(robot_xy, robot_quat, distance):
    if distance <= STOP_RADIUS:
        return [0.0, 0.0]
    direction   = DOOR_CENTER_XY - robot_xy
    target_yaw  = math.atan2(direction[1], direction[0])
    current_yaw = get_robot_yaw(robot_quat)
    yaw_error   = normalize_angle(target_yaw - current_yaw)
    angular     = float(np.clip(ANGULAR_GAIN * yaw_error, -2.0, 2.0))
    linear      = APPROACH_SPEED * max(0.0, 1.0 - abs(yaw_error) / math.pi)
    return [linear, angular]


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

door_frames = []   # (frame_rgba, distance, step)

STATE_APPROACH = "APPROACH"
STATE_INSPECT  = "INSPECT"
STATE_DONE     = "DONE"

state     = STATE_APPROACH
step      = 0
door_held = 0
_first    = True
MAX_STEPS = 1800   # 60 s safety cutoff

all_frames        = []
all_follow_frames = []

print(f"\n[Sim] Target door: {TARGET_DOOR_PATH}")
print(f"[Sim] Door center XY={DOOR_CENTER_XY}  |  Spawn XY={ROBOT_SPAWN_XY}")
print(f"[Sim] Door angle: {DOOR_ANGLE_DEG}°  |  Stop radius: {STOP_RADIUS} m\n")

while step < MAX_STEPS and state != STATE_DONE:
    pos, quat = carter.get_world_pose()
    robot_xy  = pos[:2]
    distance  = float(np.linalg.norm(robot_xy - DOOR_CENTER_XY))

    if _first:
        print(f"[Sim] Initial pos: X={pos[0]:.2f} Y={pos[1]:.2f} Z={pos[2]:.2f}")
        print(f"[Sim] Initial distance to door: {distance:.2f} m")
        _first = False

    # Robot fill light
    lp = pos + np.array([0.0, 0.0, 1.0])
    fill_xform.ClearXformOpOrder()
    fill_xform.AddTranslateOp().Set((float(lp[0]), float(lp[1]), float(lp[2])))

    # Follow fill light
    flp = pos + FOLLOW_OFFSET * 0.5 + np.array([0.0, 0.0, 1.5])
    follow_fill_xform.ClearXformOpOrder()
    follow_fill_xform.AddTranslateOp().Set((float(flp[0]), float(flp[1]), float(flp[2])))

    # Attach onboard camera to robot
    r_obj = ScipyR.from_quat([quat[1], quat[2], quat[3], quat[0]])
    cam_world_pos = pos + r_obj.apply(ONBOARD_OFFSET)
    onboard_cam.set_world_pose(position=cam_world_pos, orientation=quat)

    # Follow camera
    follow_cam.set_world_pose(
        position=pos + FOLLOW_OFFSET,
        orientation=FOLLOW_ORIENT,
    )

    # Drive / inspect
    if state == STATE_APPROACH:
        cmd = steer_toward_door(robot_xy, quat, distance)

        if distance <= TARGET_RADIUS:
            state = STATE_INSPECT
            print(f"\n[Sim] *** AT DOOR — starting inspection *** (dist={distance:.2f} m)\n")
            cmd = [0.0, 0.0]

    elif state == STATE_INSPECT:
        cmd = [0.0, 0.0]

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
print(f"\n[Sim] Sim complete — {len(door_frames)} door frames collected")
print(f"[Sim] Onboard frames: {len(all_frames)}  |  Follow frames: {len(all_follow_frames)}")

# ---------------------------------------------------------------------------
# VLM Inference — fine-tuned model only
# ---------------------------------------------------------------------------

print("\n[VLM] Loading dependencies...")
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

device    = "cuda" if torch.cuda.is_available() else "cpu"
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


door_results = []   # (dist, step_n, response, parsed_class)

if not door_frames:
    print("\n[VLM] WARNING: No door frames — robot did not reach door.")
    print("  Check ROBOT_SPAWN_XY, DOOR_CENTER_XY, building geometry.")
else:
    print(f"\n[VLM] Loading FINE-TUNED model ({len(door_frames)} frames to classify)...")
    print(f"[VLM] Adapter: {ADAPTER_PATH}")
    sys.stdout.flush()

    try:
        if not os.path.isdir(ADAPTER_PATH):
            print(f"[VLM] WARN: adapter not found — falling back to base model")
            sys.stdout.flush()
            fine_model = Qwen2VLForConditionalGeneration.from_pretrained(
                VLM_MODEL_ID, torch_dtype=torch.float16,
                device_map="auto", attn_implementation=attn_impl,
            )
        else:
            from peft import PeftModel
            t0 = time.time()
            processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
            print("[VLM] Loading base model weights...")
            sys.stdout.flush()
            fine_model = Qwen2VLForConditionalGeneration.from_pretrained(
                VLM_MODEL_ID, torch_dtype=torch.float16,
                device_map="auto", attn_implementation=attn_impl,
            )
            print(f"[VLM] Attaching LoRA adapter from {ADAPTER_PATH}...")
            sys.stdout.flush()
            fine_model = PeftModel.from_pretrained(fine_model, ADAPTER_PATH)
            # Do NOT call merge_and_unload() — crashes with modules_to_save
            # adapters in peft 0.18.1.  PeftModel inference is identical.
            fine_model.eval()
            print(f"[VLM] Fine-tuned model ready in {time.time()-t0:.1f}s")
            sys.stdout.flush()
    except Exception as _e:
        import traceback
        print(f"\n[VLM] ERROR loading fine-tuned model: {_e}")
        traceback.print_exc()
        sys.stdout.flush()
        print("[VLM] Falling back to base model...")
        sys.stdout.flush()
        fine_model = Qwen2VLForConditionalGeneration.from_pretrained(
            VLM_MODEL_ID, torch_dtype=torch.float16,
            device_map="auto", attn_implementation=attn_impl,
        )
    else:
        # processor already loaded inside try block only if adapter branch taken
        pass

    # Ensure processor is loaded (may have been skipped if base model fallback)
    if "processor" not in dir() or processor is None:
        processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)

    fine_model.eval()

    for i, (frame, dist, step_n) in enumerate(door_frames):
        pil = rgba_to_pil(frame)
        t_s = time.time()
        resp = run_vlm(fine_model, processor, pil, DOOR_PROMPT)
        elapsed = time.time() - t_s
        parsed = ("OPEN"   if "[OPEN]"   in resp.upper() else
                  "CLOSED" if "[CLOSED]" in resp.upper() else "UNKNOWN")
        door_results.append((dist, step_n, resp, parsed))
        print(f"\n[VLM] CHECK {i+1}/{N_DOOR_CHECKS}  dist={dist:.2f} m  ({elapsed:.1f}s)")
        print(f"      Response : {resp}")
        print(f"      Parsed   : {parsed}")
        sys.stdout.flush()

    del fine_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Majority vote
door_votes = [r[3] for r in door_results]
final_state = max(set(door_votes), key=door_votes.count) if door_votes else "UNKNOWN"
print(f"\n[VLM] FINAL VERDICT: {final_state}  "
      f"(OPEN={door_votes.count('OPEN')}, CLOSED={door_votes.count('CLOSED')})")
print(f"[VLM] Ground truth : {'OPEN' if DOOR_ANGLE_DEG > 5 else 'CLOSED'} "
      f"(angle = {DOOR_ANGLE_DEG}°)")
correct = (final_state == ("OPEN" if DOOR_ANGLE_DEG > 5 else "CLOSED"))
print(f"[VLM] Correct      : {'YES' if correct else 'NO'}")

# ---------------------------------------------------------------------------
# Save frames and log
# ---------------------------------------------------------------------------

frames_out_dir    = os.path.join(_project_dir, "data",   f"vlm_generalization_frames_{RUN_STAMP}")
log_path          = os.path.join(_project_dir, "data",   f"vlm_generalization_{RUN_STAMP}.txt")
video_path        = os.path.join(_project_dir, "videos", f"vlm_generalization_{RUN_STAMP}.mp4")
follow_video_path = os.path.join(_project_dir, "videos", f"vlm_generalization_follow_{RUN_STAMP}.mp4")
os.makedirs(frames_out_dir, exist_ok=True)
os.makedirs(os.path.join(_project_dir, "videos"), exist_ok=True)


def draw_label(pil_img, dist, response, parsed):
    import textwrap
    from PIL import ImageDraw as _ID
    draw = _ID.Draw(pil_img)
    w, h = pil_img.size
    font_sm = load_font(12)
    font_bd = load_font(11)

    lines  = textwrap.wrap(response, width=80)[:8]
    line_h = 16
    bar_h  = len(lines) * line_h + 12
    bar_y  = h - bar_h
    draw.rectangle([(0, bar_y), (w, h)], fill=(10, 10, 10))
    y = bar_y + 6
    for line in lines:
        draw.text((8, y), line, fill=(230, 230, 230), font=font_sm)
        y += line_h

    badge_color = ((30, 160, 60)  if parsed == "OPEN"   else
                   (180, 30, 30)  if parsed == "CLOSED" else (80, 40, 140))
    draw.rectangle([(0, 0), (240, 22)], fill=badge_color)
    draw.text((6, 4), f"◉ DOOR INSPECTION (generalization)", fill=(255, 255, 255), font=font_bd)
    draw.rectangle([(w - 140, 0), (w, 22)], fill=badge_color)
    draw.text((w - 134, 4), f"DOOR: {parsed}", fill=(255, 255, 255), font=font_bd)
    draw.text((8, 28), f"Door dist: {dist:.1f} m", fill=(200, 200, 200), font=font_bd)
    draw.text((8, 44), f"Target: {TARGET_DOOR_PATH.split('/')[-1]}", fill=(200, 200, 200), font=font_bd)
    return pil_img


for i, (dist, step_n, resp, parsed) in enumerate(door_results):
    pil  = rgba_to_pil(door_frames[i][0])
    pil  = draw_label(pil, dist, resp, parsed)
    fname = os.path.join(frames_out_dir, f"door_{i+1:02d}_dist{dist:.1f}m_{parsed}.png")
    pil.save(fname)
    print(f"[Frames] {fname}")

with open(log_path, "w") as lf:
    lf.write(f"VLM Generalization Test — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lf.write(f"Target door    : {TARGET_DOOR_PATH}\n")
    lf.write(f"Door center XY : {DOOR_CENTER_XY}\n")
    lf.write(f"Door angle set : {DOOR_ANGLE_DEG}° ({'OPEN' if DOOR_ANGLE_DEG > 5 else 'CLOSED'})\n")
    lf.write(f"Adapter path   : {ADAPTER_PATH}\n\n")
    lf.write("=== DOOR INSPECTION (fine-tuned model on unseen door) ===\n")
    for dist, step_n, resp, parsed in door_results:
        lf.write(f"  step {step_n:4d} | dist {dist:.2f} m | {parsed}\n  {resp}\n\n")
    lf.write(f"=== FINAL VERDICT: {final_state} ===\n")
    lf.write(f"=== GROUND TRUTH: {'OPEN' if DOOR_ANGLE_DEG > 5 else 'CLOSED'} ===\n")
    lf.write(f"=== CORRECT: {'YES' if correct else 'NO'} ===\n")
print(f"[Log] {log_path}")

# ---------------------------------------------------------------------------
# Video — annotated onboard
# ---------------------------------------------------------------------------

print("\n[Video] Writing videos...")

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


font_main  = load_font(12)
font_badge = load_font(11)

door_step_map = {s: (r, p) for (_, s, r, p) in door_results}


def annotate_frame(rgba, frame_state, frame_dist, frame_step, frame_num, total):
    import textwrap
    img  = Image.fromarray(rgba[:, :, :3].astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(img)
    h, w = rgba.shape[:2]

    # Most recent door result at or before this step
    candidates = [(p, r) for (_, s, r, p) in door_results if s <= frame_step]
    if candidates:
        parsed, resp = candidates[-1]
        badge_color = ((30, 160, 60)  if parsed == "OPEN"   else
                       (180, 30, 30)  if parsed == "CLOSED" else (80, 40, 140))
        badge_text  = f"DOOR: {parsed}"
        caption     = resp
    else:
        badge_color = (80, 40, 140)
        badge_text  = ""
        caption     = "Approaching door..."

    lines  = textwrap.wrap(caption, width=80)[:8]
    line_h = 16
    bar_h  = len(lines) * line_h + 12
    bar_y  = h - bar_h
    draw.rectangle([(0, bar_y), (w, h)], fill=(10, 10, 10))
    y = bar_y + 6
    for line in lines:
        draw.text((8, y), line, fill=(230, 230, 230), font=font_main)
        y += line_h

    draw.rectangle([(0, 0), (240, 22)], fill=badge_color if candidates else (60, 60, 60))
    draw.text((6, 4), "◉ DOOR INSPECTION (generalization)", fill=(255, 255, 255), font=font_badge)

    if candidates and badge_text:
        draw.rectangle([(w - 130, 0), (w, 22)], fill=badge_color)
        draw.text((w - 124, 4), badge_text, fill=(255, 255, 255), font=font_badge)

    draw.text((8, 28), f"Door dist: {frame_dist:.1f} m", fill=(200, 200, 200), font=font_badge)
    draw.text((w - 70, 28), f"{frame_num}/{total}", fill=(150, 150, 150), font=font_badge)

    return np.array(img)


ffmpeg_exe = find_ffmpeg()

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
            print(f"[Video]   {i+1}/{total_frames} frames...")
    proc.stdin.close()
    proc.wait()
    if proc.returncode == 0:
        print(f"[Video] {video_path}")
    else:
        print(f"[Video] ffmpeg error: {proc.stderr.read().decode()[:200]}")
else:
    print("[Video] ffmpeg not found or no frames — skipping onboard video")

# Follow camera video
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
            print(f"[Video]   follow {i+1}/{total_follow} frames...")
    proc2.stdin.close()
    proc2.wait()
    if proc2.returncode == 0:
        print(f"[Video] {follow_video_path}")
    else:
        print(f"[Video] follow ffmpeg error: {proc2.stderr.read().decode()[:200]}")
else:
    print("[Video] ffmpeg not found or no follow frames — skipping follow video")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("  VLM GENERALIZATION TEST SUMMARY")
print("="*60)
print(f"  Target door   : {TARGET_DOOR_PATH.split('/')[-1]}")
print(f"  Door angle    : {DOOR_ANGLE_DEG}° ({'OPEN' if DOOR_ANGLE_DEG > 5 else 'CLOSED'})")
print(f"  Frames checked: {len(door_results)}")
print(f"  Final verdict : {final_state}")
print(f"  Ground truth  : {'OPEN' if DOOR_ANGLE_DEG > 5 else 'CLOSED'}")
print(f"  Correct       : {'YES' if correct else 'NO'}")
print(f"  Vote breakdown: OPEN={door_votes.count('OPEN')}  "
      f"CLOSED={door_votes.count('CLOSED')}  "
      f"UNKNOWN={door_votes.count('UNKNOWN')}")
print(f"  Frames saved  : {frames_out_dir}/")
print(f"  Log           : {log_path}")
print(f"  Onboard video : {video_path}")
print(f"  Follow video  : {follow_video_path}")
print("="*60)

simulation_app.close()
