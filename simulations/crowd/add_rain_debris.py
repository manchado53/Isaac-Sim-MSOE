"""Add rain particles and floating debris to the flood scene.

Run from Isaac Sim Script Editor (floor4_no_plants.usd must be loaded
with WaterPlane already created):
  exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/add_rain_debris.py').read())
"""
from pxr import UsdGeom, UsdShade, Sdf, Gf, Vt
import omni.usd, math, random

stage = omni.usd.get_context().get_stage()

FPS         = 30
RISE_FRAMES = 30 * 20
HOLD_FRAMES = 30 * 10
TOTAL_FRAMES = RISE_FRAMES + HOLD_FRAMES
FLOOR_Z     = 26.9
FLOOD_RISE  = 1.5

random.seed(7)

# ── Helpers ───────────────────────────────────────────────────────────────────
def water_z(frame):
    return FLOOR_Z + min(frame / RISE_FRAMES, 1.0) * FLOOD_RISE

def make_material(path, color, opacity, emissive=None):
    old = stage.GetPrimAtPath(path)
    if old.IsValid():
        stage.RemovePrim(path)
    mat = UsdShade.Material.Define(stage, path)
    sh  = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("opacity",      Sdf.ValueTypeNames.Float).Set(opacity)
    sh.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(0.6)
    if emissive:
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat

# ── Rain PointInstancer ───────────────────────────────────────────────────────
rain_root = "/World/Rain"
old = stage.GetPrimAtPath(rain_root)
if old.IsValid():
    stage.RemovePrim(rain_root)

instancer = UsdGeom.PointInstancer.Define(stage, rain_root)

# Raindrop prototype: thin capsule
proto_path = rain_root + "/Proto/Drop"
drop = UsdGeom.Capsule.Define(stage, proto_path)
drop.GetRadiusAttr().Set(0.025)
drop.GetHeightAttr().Set(0.35)

rain_mat = make_material("/World/Looks/RainMat", (0.75, 0.87, 1.0), 0.35,
                          emissive=(0.05, 0.1, 0.2))
UsdShade.MaterialBindingAPI.Apply(drop.GetPrim()).Bind(rain_mat)
instancer.GetPrototypesRel().AddTarget(proto_path)

N   = 220          # number of drops
W   = 55.0         # horizontal spread radius
HI  = 5.5          # height above water where drops spawn
CYC = 20           # frames per drop cycle (~0.67s fall)

# World-space coords centered over the flood corridor
xs     = [random.uniform(-60, 20)   for _ in range(N)]
ys     = [random.uniform(20, 110)   for _ in range(N)]
phases = [random.randint(0, CYC-1)  for _ in range(N)]

# Slight tilt on drops (fixed orientation)
import math
tilt_quat = Gf.Quath(math.cos(math.radians(10)), math.sin(math.radians(10)), 0, 0)
orientations = Vt.QuathArray([tilt_quat] * N)
instancer.GetOrientationsAttr().Set(orientations)
instancer.GetProtoIndicesAttr().Set(Vt.IntArray([0] * N))

print(f"Baking rain ({N} drops, {TOTAL_FRAMES+1} frames)...")
for frame in range(TOTAL_FRAMES + 1):
    wz = water_z(frame)
    pos = []
    for i in range(N):
        t = ((frame + phases[i]) % CYC) / CYC   # 0→1 within cycle
        z = wz + HI - t * (HI + 0.3)            # spawn hi, land at water
        pos.append(Gf.Vec3f(xs[i], ys[i], z))
    instancer.GetPositionsAttr().Set(Vt.Vec3fArray(pos), frame)

print("Rain baked")

# ── Floating debris ───────────────────────────────────────────────────────────
old = stage.GetPrimAtPath("/World/Debris")
if old.IsValid():
    stage.RemovePrim("/World/Debris")

debris_mat   = make_material("/World/Looks/DebrisMat",   (0.55, 0.38, 0.22), 1.0)
barrel_mat   = make_material("/World/Looks/BarrelMat",   (0.25, 0.25, 0.28), 1.0)
plastic_mat  = make_material("/World/Looks/PlasticMat",  (0.80, 0.15, 0.10), 1.0)

# (x_offset, y_offset, shape, scale_xyz, material, bob_amp, drift_deg_per_s)
ITEMS = [
    # chairs (flat seat boxes)
    ( -8,  5, "box",      (0.50, 0.50, 0.08), debris_mat,  0.06,  8.0),
    (  4, -3, "box",      (0.50, 0.50, 0.08), debris_mat,  0.04, -6.0),
    ( 15,  8, "box",      (0.50, 0.50, 0.08), debris_mat,  0.07,  5.0),
    (-18,  2, "box",      (0.50, 0.50, 0.08), debris_mat,  0.05, -9.0),
    # tables
    (  7, 12, "box",      (1.20, 0.60, 0.07), debris_mat,  0.03,  3.0),
    (-12, 15, "box",      (1.10, 0.55, 0.07), debris_mat,  0.04, -4.0),
    # barrels
    ( -5, -8, "cylinder", (0.28, 0.28, 0.55), barrel_mat,  0.08, 12.0),
    ( 10, -5, "cylinder", (0.28, 0.28, 0.55), barrel_mat,  0.06, -10.0),
    # plastic bins / crates
    (  2, 18, "box",      (0.45, 0.35, 0.35), plastic_mat, 0.07,  7.0),
    (-20, -2, "box",      (0.40, 0.30, 0.30), plastic_mat, 0.05,  -5.0),
    # door panel floating flat
    (  8, -12,"box",      (0.90, 0.04, 1.95), debris_mat,  0.04,  2.0),
    (-15,  20,"box",      (0.70, 0.04, 1.60), debris_mat,  0.03, -3.0),
]

bob_phases = [random.uniform(0, 2 * math.pi) for _ in ITEMS]
# start positions scattered around floor 4 center
cx, cy = -20.0, 65.0

print(f"Baking {len(ITEMS)} debris items...")
for idx, (dx, dy, shape, scale, mat, bob_amp, drift) in enumerate(ITEMS):
    item_path = f"/World/Debris/Item_{idx:02d}"

    if shape == "box":
        prim = UsdGeom.Cube.Define(stage, item_path)
    else:
        prim = UsdGeom.Cylinder.Define(stage, item_path)
        prim.GetRadiusAttr().Set(scale[0])
        prim.GetHeightAttr().Set(scale[2])

    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mat)
    prim.CreateDoubleSidedAttr(True)

    xf    = UsdGeom.Xformable(prim.GetPrim())
    t_op  = xf.AddTranslateOp()
    r_op  = xf.AddRotateXYZOp()
    s_op  = xf.AddScaleOp()
    s_op.Set(Gf.Vec3f(*scale))

    bp = bob_phases[idx]
    # Drift slowly in XY as if carried by current
    drift_x = random.uniform(-0.002, 0.002)
    drift_y = random.uniform(-0.001, 0.003)

    for frame in range(TOTAL_FRAMES + 1):
        wz  = water_z(frame)
        t   = frame / FPS
        bob = bob_amp * math.sin(bp + t * 1.8)
        fx  = cx + dx + drift_x * frame
        fy  = cy + dy + drift_y * frame
        t_op.Set(Gf.Vec3d(fx, fy, wz + bob + scale[2] * 0.5), frame)
        r_op.Set(Gf.Vec3f(
            3.0 * math.sin(bp + t * 0.7),     # gentle pitch rock
            3.0 * math.cos(bp + t * 0.5),     # gentle roll rock
            drift * t                          # slow yaw drift
        ), frame)

print("Debris baked")
print("\nDone — rain + floating debris added. Press Play to see it.")
