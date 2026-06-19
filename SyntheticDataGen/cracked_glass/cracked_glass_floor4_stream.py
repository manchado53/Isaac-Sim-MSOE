#!/usr/bin/env python3
"""
Cracked-glass floor-4 livestream demo.

Loads InnovationCenter.usd in headless Isaac Sim with WebRTC livestream
enabled, picks N random glass panels on floor 4, binds the diffusion-
authored v4 cracked OmniGlass material to each, prints + saves the
world positions of every cracked panel, and keeps the simulation
running so a remote viewer can connect via the Isaac Sim WebRTC client.

Usage:
    ./python.sh SyntheticDataGen/cracked_glass/cracked_glass_floor4_stream.py
    ./python.sh SyntheticDataGen/cracked_glass/cracked_glass_floor4_stream.py \\
        --stream-ip 100.105.179.92 --n-panels 40 --seed 42

The viewer connects from another machine on the same network with the
Isaac Sim WebRTC Streaming Client, pointed at <stream-ip>:49100.
"""

import argparse
import datetime
import json
import os
import random
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
sys.path.insert(0, _project_dir)
sys.path.insert(0, _script_dir)


def parse_args():
    p = argparse.ArgumentParser(
        description="Cracked-glass floor-4 livestream demo (Innovation Center)"
    )
    p.add_argument("--stream-ip", default="100.107.199.57",
                   help="IP the WebRTC server advertises to clients "
                        "(default: rosita Tailscale IP, matches "
                        "navigation/nav2_orchestrator.py)")
    p.add_argument("--n-panels", type=int, default=40,
                   help="Number of glass panels to crack on floor 4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=os.path.join(
        _project_dir, "logs", "cracked_glass_floor4_positions.json"))
    p.add_argument("--usd", default=os.path.join(
        _project_dir, "assets", "environments",
        "InnovationCenter", "InnovationCenter.usd"))
    p.add_argument("--variants", default="edge_v1,v4,tempered_v5",
                   help="Comma-separated crack variants to distribute across "
                        "the selected panels. Each must have a "
                        "crack_normal_<variant>.png in the texture dir. "
                        "Default covers all three damage archetypes: edge "
                        "(stress crack), v4 (impact shatter), tempered "
                        "(safety-glass dice).")
    return p.parse_args()


args = parse_args()

# -----------------------------------------------------------------------------
# Launch Isaac Sim with WebRTC livestream enabled.
# Pattern copied from navigation/nav2_orchestrator.py:106-139, which mirrors
# the official Isaac Sim livestream example. The explicit renderer +
# display_options keys are required for the NVCF service to initialize the
# RTX pipeline; without them the health endpoint stays at "App not ready".
# -----------------------------------------------------------------------------
from isaacsim import SimulationApp  # noqa: E402

launch_config = {
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1080,
    "headless": True,
    "hide_ui": False,
    "renderer": "RaytracedLighting",
    "display_options": 3286,
}
print(f"[stream] Launching Isaac Sim with WebRTC stream at {args.stream_ip}",
      flush=True)
simulation_app = SimulationApp(launch_config)

simulation_app.set_setting("/app/window/drawMouse", True)
simulation_app.set_setting("/app/livestream/allowResize", True)
simulation_app.set_setting("/app/livestream/nvcf/quitOnSessionEnded", False)
simulation_app.set_setting("/app/livestream/publicEndpointAddress", args.stream_ip)
simulation_app.set_setting("/app/livestream/port", 49100)
simulation_app.set_setting("/app/renderer/skipWhileMinimized", False)
simulation_app.set_setting("/app/renderer/sleepMsOutOfFocus", 0)

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
enable_extension("omni.services.livestream.nvcf")
print("[stream] omni.services.livestream.nvcf enabled", flush=True)

# -----------------------------------------------------------------------------
# Isaac Sim API imports (only valid after SimulationApp)
# -----------------------------------------------------------------------------
import omni.usd  # noqa: E402
from pxr import Sdf, Usd, UsdGeom, UsdShade  # noqa: E402

from isaacsim.core.api import World  # noqa: E402
from sim_utils.building_info import get_floor  # noqa: E402
from sim_utils.building_physics import (  # noqa: E402
    setup_physics_scene,
    _find_mesh_descendant,
)
from glass_anomaly import (  # noqa: E402
    create_cracked_glass_materials,
    enumerate_glass_prims,
)
from sim_utils.test_cameras import add_camera_headlight  # noqa: E402

TEXTURE_DIR = os.path.join(_project_dir, "assets", "materials", "cracked_glass")


def bind_visual_material_deep(stage, prim, material_path):
    """Walk down to the Mesh descendant, unbind any pre-existing default-
    purpose material, then bind the new material at both Xform + Mesh.

    Some panels in InnovationCenter.usd have materials authored directly
    on the inner Mesh prim; binding 'strongerThanDescendants' at the
    wrapper Xform doesn't override a child Mesh binding, so the swap
    silently no-ops. This is documented as Gotcha 1 in
    docs/cracked-glass-pipeline.md.
    """
    mat = UsdShade.Material(stage.GetPrimAtPath(material_path))
    targets = [prim]
    mesh = _find_mesh_descendant(prim, max_depth=4)
    if mesh is not None:
        targets.append(mesh)
    for target in targets:
        if target is None or not target.IsValid():
            continue
        api = UsdShade.MaterialBindingAPI.Apply(target)
        try:
            api.UnbindDirectBinding()
        except Exception:
            pass
        api.Bind(mat, UsdShade.Tokens.strongerThanDescendants)


def panel_world_center(prim, bbox_cache):
    bb = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return (
        float((bb.GetMin()[0] + bb.GetMax()[0]) / 2.0),
        float((bb.GetMin()[1] + bb.GetMax()[1]) / 2.0),
        float((bb.GetMin()[2] + bb.GetMax()[2]) / 2.0),
    )


# -----------------------------------------------------------------------------
# Load the building and author materials
# -----------------------------------------------------------------------------
random.seed(args.seed)

print(f"[stream] USD: {args.usd}", flush=True)
if not os.path.isfile(args.usd):
    print(f"[stream] ERROR: USD not found at {args.usd}", flush=True)
    sys.exit(1)

world = World()
stage = omni.usd.get_context().get_stage()

building_prim = stage.DefinePrim("/World/Building", "Xform")
building_prim.GetReferences().AddReference(
    args.usd, primPath=Sdf.Path("/World/Building")
)
setup_physics_scene(stage)
# Default lighting: headlight parented to the viewport camera so wherever
# the user navigates there's local illumination. Fixed-position stage
# lights ('add_interior_lighting') only covered 4 named cameras and left
# the rest of the building dark during free-fly navigation.
add_camera_headlight(stage)

variant_list = tuple(v.strip() for v in args.variants.split(",") if v.strip())
if not variant_list:
    print("[stream] ERROR: --variants must list at least one variant",
          flush=True)
    sys.exit(2)

variant_mat_paths = {}
for v in variant_list:
    paths = create_cracked_glass_materials(
        stage,
        texture_dir=TEXTURE_DIR,
        root_path=f"/World/Looks/Crack_{v}",
        variants=(v,),
    )
    if not paths:
        print(f"[stream] WARN: variant '{v}' produced no material — skipping",
              flush=True)
        continue
    variant_mat_paths[v] = paths[0]

if not variant_mat_paths:
    print("[stream] ERROR: no valid variants — aborting", flush=True)
    sys.exit(3)
print(f"[stream] Cracked materials: {list(variant_mat_paths.keys())}",
      flush=True)

# -----------------------------------------------------------------------------
# Filter glass to floor 4
# -----------------------------------------------------------------------------
floor4 = get_floor("InnovationCenter", 4)
deck_z = floor4["floor_deck_z"]            # 27.5 m
ceiling_z = deck_z + 4.0                   # 31.5 m (floor_spacing_m["3_to_4"])
print(f"[stream] Floor 4 z range: [{deck_z:.1f}, {ceiling_z:.1f})", flush=True)

all_glass = enumerate_glass_prims(stage)
print(f"[stream] Glass prims (all): {len(all_glass)}", flush=True)
if not all_glass:
    print("[stream] ERROR: no glass prims found — aborting", flush=True)
    sys.exit(3)

bbox_cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(),
    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
)


def center_z(p):
    bb = bbox_cache.ComputeWorldBound(p).ComputeAlignedRange()
    return float((bb.GetMin()[2] + bb.GetMax()[2]) / 2.0)


floor4_glass = [p for p in all_glass if deck_z <= center_z(p) < ceiling_z]
print(f"[stream] Glass prims (floor 4): {len(floor4_glass)}", flush=True)
if len(floor4_glass) == 0:
    print("[stream] ERROR: no glass on floor 4 — check z bounds", flush=True)
    sys.exit(4)

n = min(args.n_panels, len(floor4_glass))
selected = random.sample(floor4_glass, n)
print(f"[stream] Cracking {n} panels with variants {list(variant_mat_paths.keys())}",
      flush=True)

# -----------------------------------------------------------------------------
# Bind a randomly-chosen cracked material to each panel and report positions
# -----------------------------------------------------------------------------
variant_names = list(variant_mat_paths.keys())
positions = []
variant_counts = {v: 0 for v in variant_names}
for i, panel in enumerate(selected):
    v = random.choice(variant_names)
    bind_visual_material_deep(stage, panel, variant_mat_paths[v])
    variant_counts[v] += 1
    cx, cy, cz = panel_world_center(panel, bbox_cache)
    rec = {
        "idx": i,
        "prim": str(panel.GetPath()),
        "name": panel.GetName(),
        "variant": v,
        "world_xyz": [cx, cy, cz],
    }
    positions.append(rec)
    print(f"  [{i:02d}] {panel.GetName():40s}  {v:12s}  "
          f"x={cx:+8.2f}  y={cy:+8.2f}  z={cz:+6.2f}",
          flush=True)
print(f"[stream] Variant distribution: {variant_counts}", flush=True)

os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w") as f:
    json.dump({
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "usd": args.usd,
        "floor": 4,
        "floor_z_range": [deck_z, ceiling_z],
        "stream_ip": args.stream_ip,
        "stream_port": 49100,
        "variants": list(variant_mat_paths.keys()),
        "variant_counts": variant_counts,
        "seed": args.seed,
        "n_glass_total": len(all_glass),
        "n_glass_floor4": len(floor4_glass),
        "n_cracked": len(positions),
        "panels": positions,
    }, f, indent=2)
print(f"[stream] Position sidecar written to {args.output}", flush=True)

# -----------------------------------------------------------------------------
# Position the default viewport camera at floor 4 so the connecting client
# sees cracked panels immediately. Override /OmniverseKit_Persp's xform with
# a translate + look-at toward the floor centroid.
# -----------------------------------------------------------------------------
xy_min = floor4["xy_min"]
xy_max = floor4["xy_max"]
xc = (xy_min[0] + xy_max[0]) / 2.0
yc = (xy_min[1] + xy_max[1]) / 2.0
cam_pos = (xy_max[0] + 25.0, xy_max[1] + 25.0, deck_z + 5.0)
look_at = (xc, yc, deck_z + 1.5)


def look_at_quat(eye, target, up=(0, 0, 1)):
    """Compute a quaternion (w, x, y, z) for an eye looking at target."""
    import math
    fx, fy, fz = (target[0] - eye[0], target[1] - eye[1], target[2] - eye[2])
    fl = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
    fx, fy, fz = (fx / fl, fy / fl, fz / fl)
    # Right = up × forward
    rx = up[1] * fz - up[2] * fy
    ry = up[2] * fx - up[0] * fz
    rz = up[0] * fy - up[1] * fx
    rl = math.sqrt(rx * rx + ry * ry + rz * rz) or 1.0
    rx, ry, rz = (rx / rl, ry / rl, rz / rl)
    # True up = forward × right
    ux = fy * rz - fz * ry
    uy = fz * rx - fx * rz
    uz = fx * ry - fy * rx
    # Camera convention: -Z forward, +Y up, +X right
    m00, m01, m02 = rx, ux, -fx
    m10, m11, m12 = ry, uy, -fy
    m20, m21, m22 = rz, uz, -fz
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m02 - m20) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    return (qw, qx, qy, qz)


from pxr import Gf  # noqa: E402

# Create a fresh camera prim with clean xform ops (modifying
# /OmniverseKit_Persp in place fails because it already has a populated
# xformOpOrder — ClearXformOpOrder only clears the order, not the ops).
cam_path = "/World/StreamCam"
cam_prim = UsdGeom.Camera.Define(stage, cam_path)
xform = UsdGeom.Xformable(cam_prim)
xform.ClearXformOpOrder()
t_op = xform.AddTranslateOp()
t_op.Set(Gf.Vec3d(*cam_pos))
qw, qx, qy, qz = look_at_quat(cam_pos, look_at)
o_op = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
o_op.Set(Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz)))
cam_prim.GetFocalLengthAttr().Set(18.0)
cam_prim.GetHorizontalApertureAttr().Set(20.955)

# Switch the active viewport to our new camera so connecting clients
# look through it.
try:
    from omni.kit.viewport.utility import get_active_viewport
    vp = get_active_viewport()
    if vp is not None:
        vp.camera_path = cam_path
        print(f"[stream] Active viewport switched to {cam_path}", flush=True)
except Exception as e:
    print(f"[stream] WARN: could not switch active viewport: {e}", flush=True)
print(f"[stream] Stream cam at {cam_pos} -> {look_at}", flush=True)

# -----------------------------------------------------------------------------
# Keep-alive loop. Do NOT call simulation_app.close() (causes segfault under
# streaming per CLAUDE.md). Ctrl+C breaks out of is_running().
# -----------------------------------------------------------------------------
world.reset()
print(f"[stream] Ready. Connect WebRTC client to {args.stream_ip}:49100",
      flush=True)
print("[stream] Press Ctrl+C to stop the simulation", flush=True)

while simulation_app.is_running():
    world.step(render=True)
