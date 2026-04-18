"""Add rising water flood animation to floor4_no_plants.usd.

Creates a semi-transparent water plane at Floor 4 level (Z=26.9m)
that rises 1.5m over 600 frames (20 seconds at 30fps).

Output: assets/environments/msoe_building/floor4_flood.usd
Run: sbatch simulations/crowd/submit_add_flood.sh
"""
import os
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["PRIVACY_CONSENT"] = "Y"

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, VtArray

PROJECT_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
IN_USD  = os.path.join(PROJECT_DIR, "assets/environments/msoe_building/floor4_no_plants.usd")
OUT_USD = os.path.join(PROJECT_DIR, "assets/environments/msoe_building/floor4_flood.usd")

FLOOR_Z    = 26.9   # Floor 4 slab Z (m)
FLOOD_RISE = 1.5    # How high water rises (m)
FPS        = 30
DURATION_S = 20
FRAMES     = FPS * DURATION_S  # 600

print(f"Opening: {IN_USD}")
stage = Usd.Stage.Open(IN_USD)
stage.SetStartTimeCode(0)
stage.SetEndTimeCode(FRAMES)
stage.SetTimeCodesPerSecond(FPS)

# ── Water plane mesh ──────────────────────────────────────────────────────────
water_path = "/World/WaterPlane"
mesh = UsdGeom.Mesh.Define(stage, water_path)

# Large flat quad covering Floor 4 area (120m x 120m)
half = 60.0
pts = VtArray.Vec3fArray([
    Gf.Vec3f(-half, -half, 0),
    Gf.Vec3f( half, -half, 0),
    Gf.Vec3f( half,  half, 0),
    Gf.Vec3f(-half,  half, 0),
])
mesh.GetPointsAttr().Set(pts)
mesh.GetFaceVertexCountsAttr().Set(VtArray.IntArray([4]))
mesh.GetFaceVertexIndicesAttr().Set(VtArray.IntArray([0, 1, 2, 3]))
mesh.GetSubdivisionSchemeAttr().Set("none")

# Double-sided so it's visible from below too
mesh.CreateDoubleSidedAttr(True)

# ── Animate Z position: rises from FLOOR_Z to FLOOR_Z + FLOOD_RISE ───────────
xform = UsdGeom.Xformable(mesh.GetPrim())
translate_op = xform.AddTranslateOp()

for frame in range(FRAMES + 1):
    t = frame / FRAMES  # 0.0 → 1.0
    z = FLOOR_Z + t * FLOOD_RISE
    translate_op.Set(Gf.Vec3d(-20.0, 65.0, z), frame)

print(f"Water plane animated: Z {FLOOR_Z:.1f}m → {FLOOR_Z + FLOOD_RISE:.1f}m over {DURATION_S}s")

# ── Water material ────────────────────────────────────────────────────────────
mat_path = "/World/Looks/WaterMaterial"
material = UsdShade.Material.Define(stage, mat_path)
shader   = UsdShade.Shader.Define(stage, mat_path + "/Shader")

shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateInput("diffuseColor",  Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.05, 0.25, 0.65))
shader.CreateInput("opacity",       Sdf.ValueTypeNames.Float).Set(0.55)
shader.CreateInput("roughness",     Sdf.ValueTypeNames.Float).Set(0.05)
shader.CreateInput("metallic",      Sdf.ValueTypeNames.Float).Set(0.0)
shader.CreateInput("ior",           Sdf.ValueTypeNames.Float).Set(1.33)

material.CreateSurfaceOutput().ConnectToSource(
    shader.ConnectableAPI(), "surface"
)

UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

print("Water material applied (semi-transparent blue)")

# ── Export ────────────────────────────────────────────────────────────────────
stage.Export(OUT_USD)
print(f"\nSaved: {OUT_USD}")

app.close()
