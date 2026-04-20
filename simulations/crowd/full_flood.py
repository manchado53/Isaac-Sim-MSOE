"""Full flood: wavy water + OmniPBR material, rises ground floor to roof.

Building bounds (from check_bounds.py):
  X: -7165 to 4114, Y: -1922 to 8880, Z: -0.3 to 3860.8

Run from Isaac Sim Script Editor:
  exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/full_flood.py').read())
"""
import omni.usd, math
from pxr import UsdGeom, UsdShade, Sdf, Gf, Vt

stage = omni.usd.get_context().get_stage()

FPS          = 30
RISE_FRAMES  = 600    # rises over 20s
TOTAL_FRAMES = 900    # 30s total

# Building Z: -0.3 to 3860.8 — start just below ground, flood to roof
START_Z    = -20.0
FLOOD_RISE = 3920.0   # ends at ~3900, above the 3860 roof

# Wave parameters — scaled to building unit system
WAVE_AMP   = 15.0    # ~15 units amplitude (reasonable ripple)
WAVE_FREQ  = 0.0015  # low frequency for large-scale waves
WAVE_SPEED = 1.4

# Mesh covers full building footprint + margin
# Building X: -7165 to 4114, center ~-1525; half-width ~6000
# Building Y: -1922 to 8880, center ~3479; half-width ~6000
N        = 40
half     = 6000.0
center_x = -1525.0
center_y =  3479.0

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

print(f"Baking {TOTAL_FRAMES+1} frames ({(N+1)**2} verts)...")
for frame in range(TOTAL_FRAMES + 1):
    frac = min(frame / RISE_FRAMES, 1.0)
    t_op.Set(Gf.Vec3d(center_x, center_y, START_Z + frac * FLOOD_RISE), frame)
    time_rad = frame / FPS * WAVE_SPEED
    pts = []
    for (x, y) in base_xy:
        dz = WAVE_AMP * math.sin(WAVE_FREQ*x + time_rad) * math.cos(WAVE_FREQ*y*0.7 + time_rad*0.8)
        pts.append(Gf.Vec3f(float(x), float(y), float(dz)))
    mesh.GetPointsAttr().Set(Vt.Vec3fArray(pts), frame)
    if frame % 150 == 0:
        print(f"  {frame}/{TOTAL_FRAMES}")

print("Waves baked")

# ── OmniPBR water material ────────────────────────────────────────────────────
old_mat = stage.GetPrimAtPath("/World/Looks/WaterMaterial")
if old_mat.IsValid():
    stage.RemovePrim("/World/Looks/WaterMaterial")

mat = UsdShade.Material.Define(stage, "/World/Looks/WaterMaterial")
sh  = UsdShade.Shader.Define(stage, "/World/Looks/WaterMaterial/Shader")

sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
sh.CreateIdAttr("OmniPBR")

sh.CreateInput("diffuse_color_constant",        Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.01, 0.12, 0.40))
sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.04)
sh.CreateInput("metallic_constant",             Sdf.ValueTypeNames.Float).Set(0.0)
sh.CreateInput("opacity_constant",              Sdf.ValueTypeNames.Float).Set(0.82)
sh.CreateInput("enable_opacity",                Sdf.ValueTypeNames.Bool).Set(True)
sh.CreateInput("ior_constant",                  Sdf.ValueTypeNames.Float).Set(1.33)
sh.CreateInput("specular_level",                Sdf.ValueTypeNames.Float).Set(0.6)

mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
mat.CreateDisplacementOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
mat.CreateVolumeOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")

UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)

print(f"Done — water at X={center_x}, Y={center_y}, rises Z={START_Z} to {START_Z+FLOOD_RISE}")
print("Switch to RTX Interactive, then press Play.")
