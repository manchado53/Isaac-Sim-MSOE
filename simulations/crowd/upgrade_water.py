"""Upgrade water plane: rippling wave mesh + clearcoat material.
Run from Isaac Sim Script Editor:
  exec(open('/home/somed/isaac-sim-msoe/simulations/crowd/upgrade_water.py').read())
"""
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, Vt
import omni.usd, math

stage = omni.usd.get_context().get_stage()

FPS = 30; DURATION_S = 20; FRAMES = FPS * DURATION_S
FLOOR_Z = 26.9; FLOOD_RISE = 1.5
OFFSET_X = -20.0; OFFSET_Y = 65.0

old = stage.GetPrimAtPath("/World/WaterPlane")
if old.IsValid():
    stage.RemovePrim("/World/WaterPlane")
    print("Removed old WaterPlane")

N = 10; half = 60.0
mesh = UsdGeom.Mesh.Define(stage, "/World/WaterPlane")

base_xy = []
for j in range(N + 1):
    for i in range(N + 1):
        base_xy.append((
            -half + 2 * half * i / N,
            -half + 2 * half * j / N
        ))

face_counts, face_idx = [], []
for j in range(N):
    for i in range(N):
        v0 = j * (N + 1) + i
        face_counts.append(4)
        face_idx.extend([v0, v0+1, v0+(N+1)+1, v0+(N+1)])

mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_counts))
mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(face_idx))
mesh.GetSubdivisionSchemeAttr().Set("none")
mesh.CreateDoubleSidedAttr(True)

WAVE_AMP = 0.04
WAVE_FREQ = 0.10
WAVE_SPEED = 1.8

xform = UsdGeom.Xformable(mesh.GetPrim())
t_op = xform.AddTranslateOp()

print(f"Baking {FRAMES+1} frames...")
for frame in range(FRAMES + 1):
    frac = frame / FRAMES
    t_op.Set(Gf.Vec3d(OFFSET_X, OFFSET_Y, FLOOR_Z + frac * FLOOD_RISE), frame)
    time_rad = frame / FPS * WAVE_SPEED
    pts = []
    for (x, y) in base_xy:
        dz = (WAVE_AMP
              * math.sin(WAVE_FREQ * x + time_rad)
              * math.cos(WAVE_FREQ * y * 0.7 + time_rad * 0.8))
        pts.append(Gf.Vec3f(float(x), float(y), float(dz)))
    mesh.GetPointsAttr().Set(Vt.Vec3fArray(pts), frame)

print("Wave animation baked")

mat_path = "/World/Looks/WaterMaterialV2"
old_mat = stage.GetPrimAtPath(mat_path)
if old_mat.IsValid():
    stage.RemovePrim(mat_path)

material = UsdShade.Material.Define(stage, mat_path)
shader   = UsdShade.Shader.Define(stage, mat_path + "/Shader")
shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateInput("diffuseColor",  Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.02, 0.18, 0.55))
shader.CreateInput("opacity",       Sdf.ValueTypeNames.Float).Set(0.65)
shader.CreateInput("roughness",     Sdf.ValueTypeNames.Float).Set(0.03)
shader.CreateInput("metallic",      Sdf.ValueTypeNames.Float).Set(0.0)
shader.CreateInput("ior",           Sdf.ValueTypeNames.Float).Set(1.33)
shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.9, 0.95, 1.0))
shader.CreateInput("clearcoat",     Sdf.ValueTypeNames.Float).Set(1.0)
shader.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(0.02)

material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

print("Done - rippling water with clearcoat applied")
