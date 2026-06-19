#!/usr/bin/env python3
"""
Generic Isaac Sim USD livestream viewer.

Loads any USD as a reference into a fresh Isaac Sim stage, frames a
default camera around the building's world bounding box, enables WebRTC
livestream so a viewer on another machine can connect, and keeps the
simulation stepping indefinitely.

Defaults are set up to render `MSOEBuildingLights.usd`; pass --usd to
load anything else.

Usage:
    ./python.sh simulations/environment/stream_usd.py
    ./python.sh simulations/environment/stream_usd.py \\
        --usd path/to/another.usd --stream-ip 100.107.199.57
"""

import argparse
import os
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))
sys.path.insert(0, _project_dir)


def parse_args():
    p = argparse.ArgumentParser(
        description="Generic Isaac Sim USD livestream viewer"
    )
    p.add_argument("--usd", default=os.path.join(
        _project_dir, "assets", "environments",
        "InnovationCenter", "MSOEBuildingLights.usd"))
    p.add_argument("--stream-ip", default="100.107.199.57",
                   help="IP the WebRTC server advertises to clients")
    p.add_argument("--stream-port", type=int, default=49100)
    return p.parse_args()


args = parse_args()

if not os.path.isfile(args.usd):
    print(f"[stream] ERROR: USD not found: {args.usd}", flush=True)
    sys.exit(1)

# -----------------------------------------------------------------------------
# Launch Isaac Sim with WebRTC livestream enabled. Pattern from
# navigation/nav2_orchestrator.py:106-139 — the explicit renderer + display
# options are required for the NVCF service to initialize the RTX pipeline.
# pxr cannot be imported until after SimulationApp() runs.
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
print(f"[stream] Launching Isaac Sim with WebRTC stream at "
      f"{args.stream_ip}:{args.stream_port}", flush=True)
simulation_app = SimulationApp(launch_config)

simulation_app.set_setting("/app/window/drawMouse", True)
simulation_app.set_setting("/app/livestream/allowResize", True)
simulation_app.set_setting("/app/livestream/nvcf/quitOnSessionEnded", False)
simulation_app.set_setting("/app/livestream/publicEndpointAddress", args.stream_ip)
simulation_app.set_setting("/app/livestream/port", args.stream_port)
simulation_app.set_setting("/app/renderer/skipWhileMinimized", False)
simulation_app.set_setting("/app/renderer/sleepMsOutOfFocus", 0)

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
enable_extension("omni.services.livestream.nvcf")
print("[stream] omni.services.livestream.nvcf enabled", flush=True)

# -----------------------------------------------------------------------------
# Isaac Sim API imports (only valid after SimulationApp)
# -----------------------------------------------------------------------------
import math  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: E402

from isaacsim.core.api import World  # noqa: E402

# -----------------------------------------------------------------------------
# Now (after SimulationApp init) peek at the source USD to detect its
# defaultPrim and metersPerUnit. This decides which referencing strategy works.
# -----------------------------------------------------------------------------
_peek = Usd.Stage.Open(args.usd)
_dp = _peek.GetDefaultPrim()
SOURCE_DEFAULT_PRIM = str(_dp.GetPath()) if _dp and _dp.IsValid() else None
SOURCE_MPU = _peek.GetMetadata("metersPerUnit") or 1.0
print(f"[stream] Source USD: {args.usd}", flush=True)
print(f"[stream]   defaultPrim    : {SOURCE_DEFAULT_PRIM}", flush=True)
print(f"[stream]   metersPerUnit  : {SOURCE_MPU}", flush=True)
del _peek

world = World()
stage = omni.usd.get_context().get_stage()

# -----------------------------------------------------------------------------
# Reference the source USD. If it has a defaultPrim, omit primPath so USD
# composition uses it; otherwise fall back to /World/Building (which is the
# sub-prim convention used by InnovationCenter.usd).
# -----------------------------------------------------------------------------
target_prim = stage.DefinePrim("/World/Building", "Xform")
refs = target_prim.GetReferences()
if SOURCE_DEFAULT_PRIM:
    refs.AddReference(args.usd)
    print(f"[stream] Referenced USD via defaultPrim "
          f"({SOURCE_DEFAULT_PRIM})", flush=True)
else:
    refs.AddReference(args.usd, primPath=Sdf.Path("/World/Building"))
    print("[stream] Referenced USD via /World/Building (no defaultPrim)",
          flush=True)

# -----------------------------------------------------------------------------
# Compute world bounding box and frame a corner-up camera around it. The
# bbox is in stage units (whatever metersPerUnit the source USD uses), so
# camera positions stay in stage units too — no manual conversions.
# -----------------------------------------------------------------------------
bbox_cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(),
    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
)
bb = bbox_cache.ComputeWorldBound(target_prim).ComputeAlignedRange()
bmn = (float(bb.GetMin()[0]), float(bb.GetMin()[1]), float(bb.GetMin()[2]))
bmx = (float(bb.GetMax()[0]), float(bb.GetMax()[1]), float(bb.GetMax()[2]))
center = (
    (bmn[0] + bmx[0]) / 2.0,
    (bmn[1] + bmx[1]) / 2.0,
    (bmn[2] + bmx[2]) / 2.0,
)
extent = (bmx[0] - bmn[0], bmx[1] - bmn[1], bmx[2] - bmn[2])
diag = math.sqrt(extent[0] ** 2 + extent[1] ** 2 + extent[2] ** 2)
print(f"[stream] Building bbox (stage units):", flush=True)
print(f"[stream]   min     : ({bmn[0]:+.1f}, {bmn[1]:+.1f}, {bmn[2]:+.1f})",
      flush=True)
print(f"[stream]   max     : ({bmx[0]:+.1f}, {bmx[1]:+.1f}, {bmx[2]:+.1f})",
      flush=True)
print(f"[stream]   extent  : ({extent[0]:.1f}, {extent[1]:.1f}, {extent[2]:.1f})",
      flush=True)
print(f"[stream]   center  : ({center[0]:+.1f}, {center[1]:+.1f}, {center[2]:+.1f})",
      flush=True)

# Corner-up camera position: out beyond +X/+Y corner, lifted above bbox top.
back = max(extent[0], extent[1]) * 0.7
cam_pos = (
    bmx[0] + back,
    bmx[1] + back,
    bmx[2] + extent[2] * 0.3,
)
look_at = center


def look_at_quat(eye, target, up=(0, 0, 1)):
    """Compute (w,x,y,z) quaternion for a -Z forward, +Y up camera."""
    fx, fy, fz = (target[0] - eye[0], target[1] - eye[1], target[2] - eye[2])
    fl = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
    fx, fy, fz = (fx / fl, fy / fl, fz / fl)
    rx = up[1] * fz - up[2] * fy
    ry = up[2] * fx - up[0] * fz
    rz = up[0] * fy - up[1] * fx
    rl = math.sqrt(rx * rx + ry * ry + rz * rz) or 1.0
    rx, ry, rz = (rx / rl, ry / rl, rz / rl)
    ux = fy * rz - fz * ry
    uy = fz * rx - fx * rz
    uz = fx * ry - fy * rx
    m00, m01, m02 = rx, ux, -fx
    m10, m11, m12 = ry, uy, -fy
    m20, m21, m22 = rz, uz, -fz
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return (0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
    if (m00 > m11) and (m00 > m22):
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        return ((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
    if m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        return ((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
    s = math.sqrt(1.0 + m22 - m00 - m11) * 2
    return ((m02 - m20) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)


cam_path = "/World/StreamCam"
cam_prim = UsdGeom.Camera.Define(stage, cam_path)
xform = UsdGeom.Xformable(cam_prim)
xform.ClearXformOpOrder()
t_op = xform.AddTranslateOp()
t_op.Set(Gf.Vec3d(*cam_pos))
qw, qx, qy, qz = look_at_quat(cam_pos, look_at)
o_op = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
o_op.Set(Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz)))
# Wide aperture for a cleaner overview shot of large buildings
cam_prim.GetFocalLengthAttr().Set(24.0)
cam_prim.GetHorizontalApertureAttr().Set(20.955)
# Far clip needs to extend past the building diagonal in stage units.
cam_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, float(diag * 4.0)))

print(f"[stream] Stream cam at "
      f"({cam_pos[0]:+.1f}, {cam_pos[1]:+.1f}, {cam_pos[2]:+.1f}) "
      f"-> ({look_at[0]:+.1f}, {look_at[1]:+.1f}, {look_at[2]:+.1f})",
      flush=True)

try:
    from omni.kit.viewport.utility import get_active_viewport
    vp = get_active_viewport()
    if vp is not None:
        vp.camera_path = cam_path
        print(f"[stream] Active viewport switched to {cam_path}", flush=True)
except Exception as e:
    print(f"[stream] WARN: could not switch active viewport: {e}", flush=True)

# -----------------------------------------------------------------------------
# Keep-alive loop. Do NOT call simulation_app.close() under streaming.
# -----------------------------------------------------------------------------
world.reset()
print(f"[stream] Ready. Connect WebRTC client to "
      f"{args.stream_ip}:{args.stream_port}", flush=True)
print("[stream] Press Ctrl+C to stop the simulation", flush=True)

while simulation_app.is_running():
    world.step(render=True)
