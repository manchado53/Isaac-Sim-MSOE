"""Render the flood animation to MP4 (headless batch job).

Output: videos/flood_render.mp4
Run:    sbatch simulations/crowd/submit_render_flood.sh
"""
import os, math, sys
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["PRIVACY_CONSENT"] = "Y"

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 1280, "height": 720})

import omni.usd
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, Vt

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
IN_USD  = os.path.join(PROJECT_DIR, "assets/environments/msoe_building/floor4_no_plants.usd")
OUT_MP4 = os.path.join(PROJECT_DIR, "videos/flood_render.mp4")

sys.path.insert(0, PROJECT_DIR)
from sim_utils.video_capture import VideoRecorder

FPS = 30
RISE_FRAMES  = 30 * 20   # 20s water rises
HOLD_FRAMES  = 30 * 10   # 10s waves keep flowing at peak
TOTAL_FRAMES = RISE_FRAMES + HOLD_FRAMES  # 900 frames total
FLOOR_Z = 26.9; FLOOD_RISE = 1.5
WAVE_AMP = 0.35; WAVE_FREQ = 0.10; WAVE_SPEED = 1.8

print(f"Opening: {IN_USD}")
omni.usd.get_context().open_stage(IN_USD)
app.update()

stage = omni.usd.get_context().get_stage()
stage.SetStartTimeCode(0)
stage.SetEndTimeCode(TOTAL_FRAMES)
stage.SetTimeCodesPerSecond(FPS)

# ── Water mesh ────────────────────────────────────────────────────────────────
old = stage.GetPrimAtPath("/World/WaterPlane")
if old.IsValid():
    stage.RemovePrim("/World/WaterPlane")

N = 10; half = 60.0
mesh = UsdGeom.Mesh.Define(stage, "/World/WaterPlane")

base_xy = [(-half + 2*half*i/N, -half + 2*half*j/N)
           for j in range(N+1) for i in range(N+1)]

face_counts, face_idx = [], []
for j in range(N):
    for i in range(N):
        v0 = j*(N+1)+i
        face_counts.append(4)
        face_idx.extend([v0, v0+1, v0+(N+1)+1, v0+(N+1)])

mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_counts))
mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(face_idx))
mesh.GetSubdivisionSchemeAttr().Set("none")
mesh.CreateDoubleSidedAttr(True)

xform = UsdGeom.Xformable(mesh.GetPrim())
t_op  = xform.AddTranslateOp()

print(f"Baking {TOTAL_FRAMES+1} frames (rise={RISE_FRAMES} + hold={HOLD_FRAMES})...")
for frame in range(TOTAL_FRAMES + 1):
    # Water rises during RISE_FRAMES then holds at peak
    frac = min(frame / RISE_FRAMES, 1.0)
    t_op.Set(Gf.Vec3d(-20.0, 65.0, FLOOR_Z + frac * FLOOD_RISE), frame)
    # Waves continue for the full duration
    time_rad = frame / FPS * WAVE_SPEED
    pts = []
    for (x, y) in base_xy:
        dz = WAVE_AMP * math.sin(WAVE_FREQ*x + time_rad) * math.cos(WAVE_FREQ*y*0.7 + time_rad*0.8)
        pts.append(Gf.Vec3f(float(x), float(y), float(dz)))
    mesh.GetPointsAttr().Set(Vt.Vec3fArray(pts), frame)

print("Wave baked")

# ── Water material ────────────────────────────────────────────────────────────
mat_path = "/World/Looks/WaterMaterial"
old_mat  = stage.GetPrimAtPath(mat_path)
if old_mat.IsValid():
    stage.RemovePrim(mat_path)

material = UsdShade.Material.Define(stage, mat_path)
shader   = UsdShade.Shader.Define(stage, mat_path + "/Shader")
shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateInput("diffuseColor",       Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.02, 0.18, 0.55))
shader.CreateInput("opacity",            Sdf.ValueTypeNames.Float).Set(0.72)
shader.CreateInput("roughness",          Sdf.ValueTypeNames.Float).Set(0.03)
shader.CreateInput("metallic",           Sdf.ValueTypeNames.Float).Set(0.0)
shader.CreateInput("ior",                Sdf.ValueTypeNames.Float).Set(1.33)
shader.CreateInput("specularColor",      Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.9, 0.95, 1.0))
shader.CreateInput("clearcoat",         Sdf.ValueTypeNames.Float).Set(1.0)
shader.CreateInput("clearcoatRoughness",Sdf.ValueTypeNames.Float).Set(0.02)
# Emissive blue glow keeps water visible when camera is submerged
shader.CreateInput("emissiveColor",      Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.08, 0.30))
material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

# ── Camera ────────────────────────────────────────────────────────────────────
cam_path = "/World/FloodCam"
old_cam  = stage.GetPrimAtPath(cam_path)
if old_cam.IsValid():
    stage.RemovePrim(cam_path)

cam_prim = UsdGeom.Camera.Define(stage, cam_path)
cam_xform = UsdGeom.Xformable(cam_prim.GetPrim())

# Eye-level shot: slightly above floor, looking across the water
cam_xform.AddTranslateOp().Set(Gf.Vec3d(40.0, -10.0, 29.5))
cam_xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 160.0))   # face toward building interior
cam_prim.GetFocalLengthAttr().Set(24.0)
cam_prim.GetHorizontalApertureAttr().Set(20.955)

print("Camera set")
app.update()

# ── Render loop ───────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_MP4), exist_ok=True)

import omni.timeline
timeline = omni.timeline.get_timeline_interface()
timeline.set_current_time(0.0)
timeline.play()

# Switch viewport to flood cam
import omni.kit.viewport.utility as vp_util
viewport = vp_util.get_active_viewport()
viewport.camera_path = cam_path

print(f"Rendering {TOTAL_FRAMES} frames → {OUT_MP4}")
with VideoRecorder(os.path.splitext(os.path.basename(OUT_MP4))[0]) as recorder:
    for frame in range(TOTAL_FRAMES + 1):
        timeline.set_current_time(frame / FPS)
        app.update()
        recorder.capture()
        if frame % 60 == 0:
            print(f"  Frame {frame}/{TOTAL_FRAMES}")

print(f"\nDone → {OUT_MP4}")
app.close()
