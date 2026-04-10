"""Render a snapshot from each nav node position and save a contact sheet.

Places a camera at eye height (floor_z + 1.6m) above each node, faces +X,
renders one frame, then tiles all shots into a single PNG contact sheet.

Output: videos/node_snapshots/contact_sheet.png  +  one PNG per node
Run:    ~/isaacsim/python.sh simulations/crowd/snapshot_nodes.py
"""

import os, sys, json
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["PRIVACY_CONSENT"] = "Y"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "width": 320, "height": 240})

import numpy as np
import omni.usd
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera
from pxr import UsdLux, UsdGeom, Sdf, Gf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
USD_PATH    = os.path.join(PROJECT_DIR, "assets/environments/msoe_building/innovation_center_finalV1.usd")
GRAPH_PATH  = os.path.join(PROJECT_DIR, "assets/Graph/floor_4_nav_graph_world_xflipped.json")
OUT_DIR     = os.path.join(PROJECT_DIR, "videos/node_snapshots")
os.makedirs(OUT_DIR, exist_ok=True)

FLOOR_Z    = 31.039   # actual floor Z from raycasting
EYE_HEIGHT = 1.6      # camera height above floor

# ── Load nav graph ─────────────────────────────────────────────────────────────
with open(GRAPH_PATH) as f:
    gdata = json.load(f)
nodes = [(n["id"], n["world_xy_cm"][0]/100, n["world_xy_cm"][1]/100)
         for n in gdata["nodes"]]
print(f"[INFO] {len(nodes)} nodes loaded", flush=True)

# ── Load stage + world ────────────────────────────────────────────────────────
omni.usd.get_context().open_stage(USD_PATH)
world = World(stage_units_in_meters=1.0)

stage = omni.usd.get_context().get_stage()
dome  = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(2000)

world.reset()
simulation_app.update()

# ── Camera (low res for speed) ────────────────────────────────────────────────
cam_z = FLOOR_Z + EYE_HEIGHT
cam = Camera(
    prim_path="/World/SnapCam",
    position=np.array([nodes[0][1], nodes[0][2], cam_z]),
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True),
    frequency=30,
    resolution=(320, 240),
)
# Slight downward pitch
cam.set_focal_length(12.0)

world.reset()
simulation_app.update()
cam.initialize()

# Warmup
for _ in range(10):
    world.step(render=True)
print("[INFO] Warmup done", flush=True)

# Identity quaternion — camera faces +X world direction for all nodes
identity_quat = np.array([1.0, 0.0, 0.0, 0.0])

# ── Snapshot each node ────────────────────────────────────────────────────────
frames = []
for i, (nid, nx, ny) in enumerate(nodes):
    cam.set_world_pose(
        position=np.array([nx, ny, cam_z]),
        orientation=identity_quat,
    )
    world.step(render=True)
    world.step(render=True)  # extra step to let renderer catch up

    rgba = cam.get_rgba()
    if rgba is not None:
        rgb = rgba[:, :, :3].astype(np.uint8)
        frames.append((nid, rgb))
        # Save individual PNG
        from PIL import Image
        Image.fromarray(rgb).save(os.path.join(OUT_DIR, f"{nid}.png"))
    else:
        print(f"[WARN] No frame for {nid}", flush=True)

    if (i + 1) % 10 == 0:
        print(f"[INFO] Captured {i+1}/{len(nodes)}", flush=True)

print(f"[INFO] {len(frames)} frames captured", flush=True)

# ── Contact sheet ──────────────────────────────────────────────────────────────
from PIL import Image, ImageDraw, ImageFont

COLS   = 6
ROWS   = (len(frames) + COLS - 1) // COLS
TW, TH = 320, 240
LABEL  = 30
sheet  = Image.new("RGB", (COLS * TW, ROWS * (TH + LABEL)), (30, 30, 30))
draw   = ImageDraw.Draw(sheet)

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
except Exception:
    font = ImageFont.load_default()

for idx, (nid, rgb) in enumerate(frames):
    col = idx % COLS
    row = idx // COLS
    x   = col * TW
    y   = row * (TH + LABEL)
    sheet.paste(Image.fromarray(rgb), (x, y))
    # dark background behind label for readability
    draw.rectangle([x, y + TH, x + TW, y + TH + LABEL], fill=(20, 20, 20))
    draw.text((x + 6, y + TH + 5), f"#{idx+1} {nid}", fill=(255, 220, 0), font=font)

sheet_path = os.path.join(OUT_DIR, "contact_sheet.png")
sheet.save(sheet_path)
print(f"[INFO] Contact sheet saved: {sheet_path}", flush=True)

simulation_app.close()
