"""Render top-down building view with nav graph overlaid.
Saves videos/navgraph_overlay.png — use this to identify edges that cross walls.
Run on a GPU node: ~/isaacsim/python.sh simulations/crowd/navgraph_overlay.py
"""
import os, sys, json
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["PRIVACY_CONSENT"] = "Y"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "width": 2048, "height": 1024})

import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera
from pxr import UsdLux, UsdGeom, Sdf, Gf
import isaacsim.core.utils.numpy.rotations as rot_utils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from sim_utils.video_capture import VideoRecorder

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
USD_PATH    = os.path.join(PROJECT_DIR, "assets/environments/msoe_building/innovation_center_finalV1.usd")
GRAPH_PATH  = os.path.join(PROJECT_DIR, "assets/Graph/floor_4_nav_graph_world_xflipped.json")

with open(GRAPH_PATH) as f:
    gdata = json.load(f)
z_m   = gdata["z_height_cm"] / 100.0
nodes = {n["id"]: np.array([n["world_xy_cm"][0]/100, n["world_xy_cm"][1]/100]) for n in gdata["nodes"]}
edges = [(e["from"], e["to"]) for e in gdata["edges"]]

# ── Load building ─────────────────────────────────────────────────────────────
omni.usd.get_context().open_stage(USD_PATH)
world = World(stage_units_in_meters=1.0)

stage = omni.usd.get_context().get_stage()
dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(10000)

# ── Top-down orthographic camera ──────────────────────────────────────────────
xs = [v[0] for v in nodes.values()]
ys = [v[1] for v in nodes.values()]
cx, cy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2

# Camera high above floor, looking straight down
cam_pos = np.array([cx, cy, z_m + 80.0])
# Looking straight down: pitch=-90, yaw=0 → [0, -90, 0]
cam_orient = rot_utils.euler_angles_to_quats(np.array([0.0, -90.0, 0.0]), degrees=True)

top_cam = Camera(
    prim_path="/World/TopCam",
    position=cam_pos,
    orientation=cam_orient,
    frequency=30,
    resolution=(2048, 1024),
)

world.reset()
simulation_app.update()
top_cam.initialize()

for _ in range(60):
    world.step(render=True)

rgba = top_cam.get_rgba()
if rgba is None:
    print("[WARN] Camera returned None — trying again")
    for _ in range(30):
        world.step(render=True)
    rgba = top_cam.get_rgba()

if rgba is not None:
    print(f"[INFO] Got frame: shape={rgba.shape} mean={rgba.mean():.1f}")
    rgb = rgba[:, :, :3]

    # ── Overlay nav graph ─────────────────────────────────────────────────────
    # Map world XY → pixel coords using camera FOV approximation
    # Camera is 80m above floor, looking straight down over the nav area
    # Rough mapping: world units → pixels
    img_h, img_w = rgb.shape[:2]
    x_range = max(xs) - min(xs) + 20
    y_range = max(ys) - min(ys) + 20
    scale = min(img_w / x_range, img_h / y_range) * 0.7
    ox = img_w/2 - cx * scale
    oy = img_h/2 - cy * scale

    fig, ax = plt.subplots(figsize=(img_w/100, img_h/100), dpi=100)
    ax.imshow(rgb, extent=[0, img_w, 0, img_h], origin="upper")

    for src, dst in edges:
        a, b = nodes[src], nodes[dst]
        px = [a[0]*scale + ox, b[0]*scale + ox]
        py = [img_h - (a[1]*scale + oy), img_h - (b[1]*scale + oy)]
        ax.plot(px, py, "r-", lw=1.5, alpha=0.8)

    for nid, pos in nodes.items():
        px = pos[0]*scale + ox
        py = img_h - (pos[1]*scale + oy)
        ax.plot(px, py, "yo", ms=4)
        ax.annotate(nid, (px, py), fontsize=5, color="yellow",
                    xytext=(2, 2), textcoords="offset points")

    ax.set_xlim(0, img_w)
    ax.set_ylim(0, img_h)
    ax.axis("off")
    fig.tight_layout(pad=0)
    out = os.path.join(PROJECT_DIR, "videos/navgraph_overlay.png")
    plt.savefig(out, dpi=100, bbox_inches="tight")
    print(f"[INFO] Saved {out}")
    plt.close()
else:
    print("[ERROR] Camera returned None both attempts")

simulation_app.close()
