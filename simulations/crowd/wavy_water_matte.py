"""Dense wavy water with fully matte material — no lighting flicker.

40x40 grid with 0.12m waves. Matte diffuse blue only.

Run from Isaac Sim Script Editor:
  exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/wavy_water_matte.py').read())
"""
import omni.usd, math
from pxr import UsdGeom, UsdShade, Sdf, Gf, Vt

stage = omni.usd.get_context().get_stage()

FPS=30; RISE_FRAMES=600; HOLD_FRAMES=300; TOTAL_FRAMES=900
FLOOR_Z=26.9; FLOOD_RISE=1.5
WAVE_AMP=0.12; WAVE_FREQ=0.10; WAVE_SPEED=1.4
N=40; half=60.0

# ── Rebuild mesh ──────────────────────────────────────────────────────────────
old = stage.GetPrimAtPath("/World/WaterPlane")
if old.IsValid():
    stage.RemovePrim("/World/WaterPlane")

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

xf   = UsdGeom.Xformable(mesh.GetPrim())
t_op = xf.AddTranslateOp()

print(f"Baking {TOTAL_FRAMES+1} frames ({(N+1)**2} verts)... (~5-10 min)")
for frame in range(TOTAL_FRAMES + 1):
    frac = min(frame / RISE_FRAMES, 1.0)
    t_op.Set(Gf.Vec3d(-20.0, 65.0, FLOOR_Z + frac * FLOOD_RISE), frame)
    time_rad = frame / FPS * WAVE_SPEED
    pts = []
    for (x, y) in base_xy:
        dz = WAVE_AMP * math.sin(WAVE_FREQ*x + time_rad) * math.cos(WAVE_FREQ*y*0.7 + time_rad*0.8)
        pts.append(Gf.Vec3f(float(x), float(y), float(dz)))
    mesh.GetPointsAttr().Set(Vt.Vec3fArray(pts), frame)
    if frame % 150 == 0:
        print(f"  {frame}/{TOTAL_FRAMES}")

print("Waves baked")

# ── Fully matte material — zero specular, zero reflection ────────────────────
old_mat = stage.GetPrimAtPath("/World/Looks/WaterMaterial")
if old_mat.IsValid():
    stage.RemovePrim("/World/Looks/WaterMaterial")

mat = UsdShade.Material.Define(stage, "/World/Looks/WaterMaterial")
sh  = UsdShade.Shader.Define(stage, "/World/Looks/WaterMaterial/Shader")
sh.CreateIdAttr("UsdPreviewSurface")
sh.CreateInput("diffuseColor",  Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.02, 0.15, 0.50))
sh.CreateInput("opacity",       Sdf.ValueTypeNames.Float).Set(0.85)
sh.CreateInput("roughness",     Sdf.ValueTypeNames.Float).Set(1.0)   # fully matte
sh.CreateInput("metallic",      Sdf.ValueTypeNames.Float).Set(0.0)
sh.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)

print("Done — matte wavy water ready. Press Play to preview.")
