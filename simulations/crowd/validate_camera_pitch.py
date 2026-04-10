"""Validate camera pitch direction in Isaac Sim.

Places a camera at a fixed interior nav node, tries 5 pitch values,
saves a labeled contact sheet so we know exactly which sign tilts down.

Output: videos/camera_pitch_validation.png
Run:    ~/isaacsim/python.sh simulations/crowd/validate_camera_pitch.py
"""
import os, sys, json
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["PRIVACY_CONSENT"] = "Y"

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 640, "height": 480})

import numpy as np
import omni.usd
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera
from scipy.spatial.transform import Rotation as R
from pxr import UsdLux, Sdf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
USD_PATH    = os.path.join(PROJECT_DIR, "assets/environments/msoe_building/innovation_center_finalV1.usd")
GRAPH_PATH  = os.path.join(PROJECT_DIR, "assets/Graph/floor_4_nav_graph_world_xflipped.json")
OUT_PATH    = os.path.join(PROJECT_DIR, "videos/camera_pitch_validation.png")

FLOOR_Z    = 31.039
EYE_Z      = FLOOR_Z + 1.6   # eye height above floor
FOLLOW_OFFSET = np.array([-3.0, 0.0, 1.8])

# Use node_3 — confirmed good interior node
with open(GRAPH_PATH) as f:
    gdata = json.load(f)
n = next(n for n in gdata["nodes"] if n["id"] == "node_3")
NAV_X = n["world_xy_cm"][0] / 100
NAV_Y = n["world_xy_cm"][1] / 100

# Camera sits 3m behind node_3 facing +X (yaw=0)
YAW      = 0.0
cam_pos  = np.array([NAV_X - 3.0, NAV_Y, EYE_Z])

omni.usd.get_context().open_stage(USD_PATH)
world = World(stage_units_in_meters=1.0)

stage = omni.usd.get_context().get_stage()
dome  = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(2000)

world.reset()
app.update()

cam = Camera(
    prim_path="/World/PitchTestCam",
    position=cam_pos,
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True),
    frequency=30,
    resolution=(640, 480),
)
cam.set_focal_length(18.0)

world.reset()
app.update()
cam.initialize()

for _ in range(15):
    world.step(render=True)

# Pitch values to test (radians) — negative and positive
PITCH_VALUES = [-0.30, -0.15, 0.0, +0.15, +0.30]

frames = []
for pitch in PITCH_VALUES:
    # Intrinsic zy: yaw around world-Z first, then pitch around camera-local Y
    q = R.from_euler("zy", [YAW, pitch]).as_quat()
    quat_wxyz = np.array([q[3], q[0], q[1], q[2]])

    cam.set_world_pose(position=cam_pos, orientation=quat_wxyz)
    world.step(render=True)
    world.step(render=True)

    rgba = cam.get_rgba()
    if rgba is not None:
        frames.append((pitch, rgba[:, :, :3].astype(np.uint8)))
    else:
        frames.append((pitch, np.zeros((480, 640, 3), dtype=np.uint8)))

print(f"[INFO] Captured {len(frames)} frames", flush=True)

# Build contact sheet
from PIL import Image, ImageDraw, ImageFont
TW, TH = 640, 480
LABEL  = 40
sheet  = Image.new("RGB", (len(frames) * TW, TH + LABEL), (20, 20, 20))
draw   = ImageDraw.Draw(sheet)

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
except Exception:
    font = ImageFont.load_default()

for i, (pitch, rgb) in enumerate(frames):
    x = i * TW
    sheet.paste(Image.fromarray(rgb), (x, 0))
    draw.rectangle([x, TH, x + TW, TH + LABEL], fill=(20, 20, 20))
    label = f"pitch={pitch:+.2f} rad  ({'DOWN?' if pitch > 0 else 'UP?' if pitch < 0 else 'FLAT'})"
    draw.text((x + 8, TH + 8), label, fill=(255, 220, 0), font=font)

sheet.save(OUT_PATH)
print(f"[INFO] Saved: {OUT_PATH}", flush=True)

app.close()
