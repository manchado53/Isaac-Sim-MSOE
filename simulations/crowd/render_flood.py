"""Render the flood animation to MP4 (headless batch job).

Uses omni.replicator camera with look_at — works in headless mode.

Output: videos/flood_render.mp4
Run:    sbatch simulations/crowd/submit_render_flood.sh
"""
import os, math, sys
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["PRIVACY_CONSENT"] = "Y"

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 1280, "height": 720})

import omni.usd, omni.timeline
import omni.replicator.core as rep
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, Vt

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
IN_USD  = os.path.join(PROJECT_DIR, "assets/environments/msoe_building/floor4_no_plants.usd")

sys.path.insert(0, PROJECT_DIR)
from sim_utils.video_capture import VideoRecorder

FPS          = 30
RISE_FRAMES  = 30 * 20    # 20s rising
HOLD_FRAMES  = 30 * 10    # 10s waves at peak
TOTAL_FRAMES = RISE_FRAMES + HOLD_FRAMES  # 900
FLOOR_Z = 26.9; FLOOD_RISE = 1.5
WAVE_AMP = 0.35; WAVE_FREQ = 0.10; WAVE_SPEED = 1.8

print(f"Opening: {IN_USD}")
omni.usd.get_context().open_stage(IN_USD)
for _ in range(5):
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
    frac = min(frame / RISE_FRAMES, 1.0)
    t_op.Set(Gf.Vec3d(-20.0, 65.0, FLOOR_Z + frac * FLOOD_RISE), frame)
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
shader.CreateInput("emissiveColor",      Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.08, 0.30))
material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

print("Material applied")

# ── Replicator camera (look_at handles orientation automatically) ─────────────
# Position outside flood zone looking across the floor toward building interior
cam = rep.create.camera(
    position=(-20, -50, 31),
    look_at=(-20, 65, 27),
)
rp = rep.create.render_product(cam, resolution=(1280, 720))
rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_annot.attach([rp])

print("Camera + render product created")

# Warmup: let RTX initialize
for _ in range(30):
    app.update()

# ── Render loop ───────────────────────────────────────────────────────────────
timeline = omni.timeline.get_timeline_interface()
timeline.set_start_time(0)
timeline.set_end_time(TOTAL_FRAMES / FPS)

print(f"Rendering {TOTAL_FRAMES} frames → videos/flood_render.mp4")
with VideoRecorder("flood_render") as recorder:
    for frame in range(TOTAL_FRAMES + 1):
        timeline.set_current_time(frame / FPS)
        app.update()
        app.update()  # second update flushes RTX render pipeline
        rgba = rgb_annot.get_data()
        if rgba is not None and rgba.size > 0:
            recorder.capture_rgba(rgba)
        if frame % 60 == 0:
            print(f"  Frame {frame}/{TOTAL_FRAMES}")

print("Done")
app.close()
