#!/usr/bin/env python3
"""
Nova Carter — VLM Patrol Demo

Drives Nova Carter through the MSOE Innovation Center (Kern Engineering
Building) and runs Qwen2-VL inference on the robot's onboard camera feed.
The resulting scene descriptions are overlaid on the output video.

Pipeline:
  Phase 1 — Simulation : collect onboard + follow RGBA frames in memory
  Phase 2 — VLM        : run Qwen2-VL-2B-Instruct on sampled frames locally
  Phase 3 — Video      : overlay captions on onboard frames, write MP4s

Outputs (in videos/):
  vlm_demo_onboard.mp4  — annotated 1st-person feed with AI captions
  vlm_demo_follow.mp4   — 3rd-person debug view (unannotated)

Setup (run once on a GPU node before submitting the job):
  ~/isaacsim/python.sh -m pip install transformers accelerate qwen-vl-utils Pillow

No API key required — Qwen2-VL runs locally on the T4 GPU after Isaac Sim closes.
First run downloads ~4 GB of model weights to ~/.cache/huggingface/.

Tuning notes:
  The Innovation Center USD is a Revit export without collision geometry.
  A ground plane is added to support the robot. If the robot spawns
  underground or above the floor, adjust ROBOT_SPAWN_Z below.
  If the building interior isn't visible in camera, adjust ROBOT_SPAWN_XY.
"""

import os
import sys
import textwrap
import subprocess
import shutil

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_FRAMES     = 720          # 48s of sim time (steps @ 30 Hz physics)
FPS            = 15           # output video FPS (we capture every other render step)
VLM_INTERVAL   = 15           # run VLM every Nth *collected* frame (~2 s of video)
VLM_MODEL_ID   = "Qwen/Qwen2-VL-2B-Instruct"
CAM_RESOLUTION = (640, 480)

# ---------------------------------------------------------------------------
# Spawn tuning — adjust if the robot clips into geometry or misses the building.
#
# The Innovation Center USD is a Revit export in feet (metersPerUnit=0.3048).
# Isaac Sim reads metersPerUnit from the file, so coordinates here are metres.
#
#   ROBOT_SPAWN_Z  — lift above world origin to sit on the ground floor.
#                    Try 1.0, 2.0, 5.0 if the robot falls on first run.
#   ROBOT_SPAWN_XY — move into a corridor. Tune after checking the output video.
# ---------------------------------------------------------------------------
ROBOT_SPAWN_XY = (0.0, 0.0)
ROBOT_SPAWN_Z  = 1.0

# ---------------------------------------------------------------------------
# Phase 1: Isaac Sim
# SimulationApp must be created before any omni.* or isaacsim.* imports.
# ---------------------------------------------------------------------------

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import carb
import omni.usd
from pxr import Sdf, UsdLux, UsdGeom

import isaacsim.core.utils.numpy.rotations as rot_utils
from scipy.spatial.transform import Rotation as R
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
sys.path.insert(0, _project_dir)

# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------

assets_root = get_assets_root_path()
if assets_root is None:
    carb.log_error("Could not find Isaac Sim assets root")
    simulation_app.close()
    sys.exit(1)

world = World(stage_units_in_meters=1.0)

_building_usd = os.path.join(_project_dir, "innovation_center_finalV1.usd")
if not os.path.isfile(_building_usd):
    print(f"ERROR: Building USD not found at {_building_usd}")
    simulation_app.close()
    sys.exit(1)

print(f"[Phase 1] Loading: {_building_usd}")
add_reference_to_stage(_building_usd, prim_path="/World/InnovationCenter")

stage = omni.usd.get_context().get_stage()

# Revit exports have no lights — add ambient + directional
UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight")).CreateIntensityAttr(3000)
UsdLux.DomeLight.Define(stage,    Sdf.Path("/World/DomeLight")).CreateIntensityAttr(1500)

# Ground plane — the building USD has no collision geometry; robot drives on this
world.scene.add_default_ground_plane()

# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------

carter = world.scene.add(
    WheeledRobot(
        prim_path="/World/NovaCarter",
        name="nova_carter",
        wheel_dof_names=["joint_wheel_left", "joint_wheel_right"],
        create_robot=True,
        usd_path=assets_root + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd",
        position=np.array([ROBOT_SPAWN_XY[0], ROBOT_SPAWN_XY[1], ROBOT_SPAWN_Z]),
    )
)

controller = DifferentialController(name="ctrl", wheel_radius=0.14, wheel_base=0.4)

# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

ONBOARD_OFFSET = np.array([0.20, 0.0, 0.65])  # metres from robot centre
FOLLOW_OFFSET  = np.array([5.0, -5.0, 4.0])
FOLLOW_ORIENT  = rot_utils.euler_angles_to_quats(np.array([0, 30, 135]), degrees=True)

W, H = CAM_RESOLUTION

onboard_cam = Camera(
    prim_path="/World/OnboardCam",
    position=np.array([0.36, 0.0, 0.5]),
    frequency=FPS,
    resolution=(W, H),
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True),
)
follow_cam = Camera(
    prim_path="/World/FollowCam",
    position=FOLLOW_OFFSET,
    frequency=FPS,
    resolution=(W, H),
    orientation=FOLLOW_ORIENT,
)

world.reset()
onboard_cam.initialize()

# Wide FOV for onboard — better situational awareness
usd_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/OnboardCam"))
usd_cam.GetFocalLengthAttr().Set(5.0)
usd_cam.GetClippingRangeAttr().Set((0.01, 10000.0))

follow_cam.initialize()

# ---------------------------------------------------------------------------
# Navigation sequence  (start_step, end_step, [linear_vel m/s, angular_vel rad/s])
# ---------------------------------------------------------------------------

nav_sequence = [
    (0,    180, [0.5,  0.0]),        # straight
    (180,  270, [0.0,  np.pi / 6]), # turn right
    (270,  450, [0.5,  0.0]),        # straight
    (450,  540, [0.0,  np.pi / 6]), # turn right
    (540,  720, [0.5,  0.0]),        # straight
]

def get_command(step):
    for start, end, cmd in nav_sequence:
        if start <= step < end:
            return cmd
    return [0.0, 0.0]

# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

onboard_frames = []  # (H, W, 4) uint8 — every collected frame
follow_frames  = []  # (H, W, 4) uint8
vlm_frame_idxs = []  # indices into onboard_frames to send to VLM

print(f"[Phase 1] Simulating {MAX_FRAMES} steps...")

for step in range(MAX_FRAMES):
    if world.is_playing():
        carter.apply_wheel_actions(controller.forward(command=get_command(step)))

        pos, quat = carter.get_world_pose()

        # Keep onboard camera attached to robot heading
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        onboard_cam.set_world_pose(
            position=pos + rot.apply(ONBOARD_OFFSET),
            orientation=quat,
        )
        follow_cam.set_world_pose(
            position=pos + FOLLOW_OFFSET,
            orientation=FOLLOW_ORIENT,
        )

    world.step(render=True)  # pose must be set BEFORE step (async pipeline)

    if world.is_playing() and step % 2 == 0 and step > 10:
        ob = onboard_cam.get_rgba()
        fl = follow_cam.get_rgba()

        if ob is not None:
            frame = np.clip(ob, 0, 255).astype(np.uint8)
            onboard_frames.append(frame)
            if len(onboard_frames) % VLM_INTERVAL == 0:
                vlm_frame_idxs.append(len(onboard_frames) - 1)

        if fl is not None:
            follow_frames.append(np.clip(fl, 0, 255).astype(np.uint8))

    if step % 120 == 0:
        print(f"[Phase 1]   step {step}/{MAX_FRAMES}  "
              f"| frames: {len(onboard_frames)}")

world.stop()
simulation_app.close()
print(f"[Phase 1] Done — {len(onboard_frames)} onboard frames, "
      f"{len(vlm_frame_idxs)} queued for VLM.")

# ---------------------------------------------------------------------------
# Phase 2: Qwen2-VL Inference
# Isaac Sim is fully closed; GPU memory is free for the VLM.
# ---------------------------------------------------------------------------

print(f"\n[Phase 2] Loading {VLM_MODEL_ID}...")

try:
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError as e:
    print(f"[Phase 2] ERROR: Missing dependency — {e}")
    print("  Fix: ~/isaacsim/python.sh -m pip install "
          "transformers accelerate qwen-vl-utils Pillow")
    sys.exit(1)

# Load model — float16 keeps VRAM usage to ~4 GB (T4 has 16 GB)
vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
    VLM_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",       # places model on GPU automatically
)
vlm_model.eval()

processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
print(f"[Phase 2] Model loaded on {next(vlm_model.parameters()).device}")

VLM_PROMPT = (
    "You are the vision system of an autonomous patrol robot navigating the "
    "MSOE Kern Engineering Innovation Center, a modern university engineering "
    "building currently under construction. Describe what the robot sees in "
    "1-2 sentences: the space ahead, architectural features, rooms, corridors, "
    "equipment, or anything notable. Maximum 25 words."
)

def rgba_to_pil(rgba: np.ndarray) -> Image.Image:
    """Convert (H, W, 4) uint8 RGBA numpy array to a PIL RGB image."""
    return Image.fromarray(rgba[:, :, :3].astype(np.uint8), "RGB")

def run_vlm(pil_image: Image.Image) -> str:
    """Run Qwen2-VL inference on a PIL image. Returns the caption string."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text",  "text": VLM_PROMPT},
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(vlm_model.device)

    with torch.no_grad():
        generated_ids = vlm_model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,   # deterministic — no temperature/top_p/top_k
        )

    # Trim input tokens; decode only the newly generated portion
    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

# frame_index → caption
captions: dict[int, str] = {}

for i, frame_idx in enumerate(vlm_frame_idxs):
    try:
        pil_img = rgba_to_pil(onboard_frames[frame_idx])
        caption = run_vlm(pil_img)
    except Exception as e:
        caption = f"[VLM error: {type(e).__name__}]"

    captions[frame_idx] = caption
    print(f"[Phase 2]   [{i+1:2d}/{len(vlm_frame_idxs)}] "
          f"frame {frame_idx:4d}: {caption[:70]}")

print(f"[Phase 2] Done — {len(captions)} captions generated.")

# Free GPU memory before video assembly
del vlm_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Phase 3: Annotate frames and write videos
# ---------------------------------------------------------------------------

print(f"\n[Phase 3] Writing videos...")

videos_dir = os.path.join(_project_dir, "videos")
os.makedirs(videos_dir, exist_ok=True)

ONBOARD_OUT = os.path.join(videos_dir, "vlm_demo_onboard.mp4")
FOLLOW_OUT  = os.path.join(videos_dir, "vlm_demo_follow.mp4")

# --- ffmpeg (mirrors VideoRecorder._get_ffmpeg_path) ---

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

def open_writer(path: str, width: int, height: int, fps: int):
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found — imageio-ffmpeg should be bundled with Isaac Sim")
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{width}x{height}", "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# --- Caption overlay ---

def get_caption(frame_idx: int) -> str:
    candidates = [fi for fi in captions if fi <= frame_idx]
    if not candidates:
        return "Initializing scene analysis..."
    return captions[max(candidates)]

def load_font(size: int):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, AttributeError):
            continue
    return None

font_main  = load_font(15)
font_badge = load_font(12)

def annotate(rgba: np.ndarray, caption: str, frame_num: int, total: int) -> np.ndarray:
    """Overlay caption bar and HUD onto frame. Returns RGB uint8 array."""
    img  = Image.fromarray(rgba[:, :, :3].astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(img)
    h, w = rgba.shape[:2]

    # Caption bar — bottom
    lines   = textwrap.wrap(caption, width=62)
    line_h  = 20
    padding = 8
    bar_h   = len(lines) * line_h + 2 * padding
    bar_y   = h - bar_h
    draw.rectangle([(0, bar_y), (w, h)], fill=(10, 10, 10))
    y = bar_y + padding
    for line in lines:
        draw.text((padding, y), line, fill=(255, 255, 255), font=font_main)
        y += line_h

    # Badge — top left
    draw.rectangle([(0, 0), (150, 22)], fill=(80, 40, 140))   # purple for Qwen
    draw.text((6, 4), "QWEN2-VL VISION", fill=(255, 255, 255), font=font_badge)

    # Frame counter — top right
    draw.text((w - 70, 4), f"{frame_num}/{total}",
              fill=(180, 180, 180), font=font_badge)

    return np.array(img)

# --- Write annotated onboard video ---

total = len(onboard_frames)
proc  = open_writer(ONBOARD_OUT, W, H, FPS)

for i, frame in enumerate(onboard_frames):
    annotated = annotate(frame, get_caption(i), i + 1, total)
    proc.stdin.write(annotated.tobytes())
    if (i + 1) % 50 == 0:
        print(f"[Phase 3]   onboard {i+1}/{total}")

proc.stdin.close()
proc.wait()
if proc.returncode == 0:
    print(f"[Phase 3] ✓ {ONBOARD_OUT}")
else:
    print(f"[Phase 3] ffmpeg error: {proc.stderr.read().decode()}")

# --- Write follow video ---

if follow_frames:
    proc2 = open_writer(FOLLOW_OUT, W, H, FPS)
    for frame in follow_frames:
        proc2.stdin.write(frame[:, :, :3].tobytes())
    proc2.stdin.close()
    proc2.wait()
    if proc2.returncode == 0:
        print(f"[Phase 3] ✓ {FOLLOW_OUT}")
    else:
        print(f"[Phase 3] ffmpeg error: {proc2.stderr.read().decode()}")

print(f"\nDemo complete!")
print(f"  Onboard (annotated): {ONBOARD_OUT}")
print(f"  Follow view:         {FOLLOW_OUT}")
