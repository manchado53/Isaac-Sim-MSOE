"""Add floating debris throughout the flood halls.

Run from Isaac Sim Script Editor:
  exec(open('/home/ad.msoe.edu/somed/isaac-sim-msoe/simulations/crowd/add_rain_debris.py').read())
"""
from pxr import UsdGeom, UsdShade, Sdf, Gf, Vt
import omni.usd, math, random

stage = omni.usd.get_context().get_stage()

FPS          = 30
RISE_FRAMES  = 30 * 20
HOLD_FRAMES  = 30 * 10
TOTAL_FRAMES = RISE_FRAMES + HOLD_FRAMES
FLOOR_Z      = 26.9
FLOOD_RISE   = 1.5

random.seed(42)

def water_z(frame):
    return FLOOR_Z + min(frame / RISE_FRAMES, 1.0) * FLOOD_RISE

def make_mat(path, color, opacity=1.0):
    old = stage.GetPrimAtPath(path)
    if old.IsValid():
        stage.RemovePrim(path)
    mat = UsdShade.Material.Define(stage, path)
    sh  = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("opacity",      Sdf.ValueTypeNames.Float).Set(opacity)
    sh.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(0.65)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat

# ── Remove rain if it exists ──────────────────────────────────────────────────
rain = stage.GetPrimAtPath("/World/Rain")
if rain.IsValid():
    stage.RemovePrim("/World/Rain")
    print("Removed rain")

# ── Remove old debris ─────────────────────────────────────────────────────────
old = stage.GetPrimAtPath("/World/Debris")
if old.IsValid():
    stage.RemovePrim("/World/Debris")

# ── Materials ─────────────────────────────────────────────────────────────────
wood_mat    = make_mat("/World/Looks/Wood",    (0.55, 0.35, 0.18))
metal_mat   = make_mat("/World/Looks/Metal",   (0.40, 0.40, 0.45))
plastic_mat = make_mat("/World/Looks/Plastic", (0.75, 0.12, 0.10))
grey_mat    = make_mat("/World/Looks/Grey",    (0.55, 0.55, 0.58))
white_mat   = make_mat("/World/Looks/White",   (0.90, 0.90, 0.92))
yellow_mat  = make_mat("/World/Looks/Yellow",  (0.85, 0.75, 0.10))
dark_mat    = make_mat("/World/Looks/Dark",    (0.15, 0.15, 0.17))

# ── Debris definitions ────────────────────────────────────────────────────────
# (world_x, world_y, shape, (sx,sy,sz), material, bob_amp, drift_rot_dps)
# Spread throughout the hall corridor (X: -45 to 15, Y: 25 to 105)
ITEMS = [
    # --- chairs scattered down the hall ---
    (-30, 30,  "box", (0.50, 0.50, 0.08), wood_mat,   0.06,  9.0),
    (-22, 35,  "box", (0.50, 0.50, 0.08), wood_mat,   0.05, -7.0),
    (-15, 40,  "box", (0.50, 0.50, 0.08), wood_mat,   0.07,  5.0),
    ( -8, 38,  "box", (0.50, 0.50, 0.08), wood_mat,   0.04, 11.0),
    (  2, 45,  "box", (0.50, 0.50, 0.08), wood_mat,   0.06, -8.0),
    (-35, 50,  "box", (0.50, 0.50, 0.08), wood_mat,   0.05,  6.0),
    (-20, 55,  "box", (0.50, 0.50, 0.08), wood_mat,   0.07, -5.0),
    ( -5, 60,  "box", (0.50, 0.50, 0.08), wood_mat,   0.04, 10.0),
    (-28, 65,  "box", (0.50, 0.50, 0.08), wood_mat,   0.06, -9.0),
    (-12, 70,  "box", (0.50, 0.50, 0.08), wood_mat,   0.05,  7.0),
    (  3, 75,  "box", (0.50, 0.50, 0.08), wood_mat,   0.07, -6.0),
    (-40, 80,  "box", (0.50, 0.50, 0.08), wood_mat,   0.04,  8.0),
    (-18, 85,  "box", (0.50, 0.50, 0.08), wood_mat,   0.06, -4.0),
    ( -2, 90,  "box", (0.50, 0.50, 0.08), wood_mat,   0.05,  5.0),
    (-25, 95,  "box", (0.50, 0.50, 0.08), wood_mat,   0.07, 12.0),
    (-10,100,  "box", (0.50, 0.50, 0.08), wood_mat,   0.04, -7.0),

    # --- tables ---
    (-25, 33,  "box", (1.20, 0.60, 0.07), wood_mat,   0.03,  3.0),
    ( -5, 48,  "box", (1.10, 0.55, 0.07), wood_mat,   0.04, -3.0),
    (-32, 62,  "box", (1.30, 0.65, 0.07), wood_mat,   0.03,  2.5),
    (-15, 77,  "box", (1.20, 0.60, 0.07), wood_mat,   0.04, -2.0),
    (  1, 92,  "box", (1.10, 0.55, 0.07), wood_mat,   0.03,  3.5),

    # --- barrels / cylinders ---
    (-38, 42,  "cyl", (0.28, 0.28, 0.55), metal_mat,  0.08, 14.0),
    (-20, 58,  "cyl", (0.28, 0.28, 0.55), metal_mat,  0.07,-12.0),
    ( -8, 72,  "cyl", (0.28, 0.28, 0.55), metal_mat,  0.09, 10.0),
    (-30, 88,  "cyl", (0.28, 0.28, 0.55), metal_mat,  0.06,-11.0),
    (-42, 68,  "cyl", (0.22, 0.22, 0.40), dark_mat,   0.07,  9.0),

    # --- plastic crates / bins ---
    (-18, 36,  "box", (0.45, 0.35, 0.35), plastic_mat,0.07,  8.0),
    ( -3, 52,  "box", (0.40, 0.30, 0.30), plastic_mat,0.06, -6.0),
    (-35, 67,  "box", (0.45, 0.35, 0.35), plastic_mat,0.08,  7.0),
    (-14, 83,  "box", (0.40, 0.30, 0.30), plastic_mat,0.05, -5.0),
    ( -1, 97,  "box", (0.45, 0.35, 0.35), plastic_mat,0.07,  6.0),

    # --- door panels floating flat ---
    (-22, 44,  "box", (0.90, 0.04, 1.95), grey_mat,   0.03,  2.0),
    ( -9, 63,  "box", (0.85, 0.04, 1.80), grey_mat,   0.04, -1.5),
    (-33, 79,  "box", (0.90, 0.04, 1.95), grey_mat,   0.03,  2.5),

    # --- whiteboards floating sideways ---
    (-27, 41,  "box", (1.80, 0.04, 1.20), white_mat,  0.04,  1.5),
    ( -4, 69,  "box", (1.60, 0.04, 1.10), white_mat,  0.03, -2.0),
    (-38, 94,  "box", (1.80, 0.04, 1.20), white_mat,  0.04,  1.0),

    # --- yellow hazard signs / small boxes ---
    (-16, 47,  "box", (0.30, 0.30, 0.30), yellow_mat, 0.09, 15.0),
    (-29, 73,  "box", (0.25, 0.25, 0.25), yellow_mat, 0.08,-13.0),
    (  0, 56,  "box", (0.30, 0.30, 0.30), yellow_mat, 0.10, 11.0),

    # --- monitor / equipment boxes ---
    (-11, 61,  "box", (0.60, 0.40, 0.50), dark_mat,   0.05,  5.0),
    (-24, 84,  "box", (0.55, 0.38, 0.45), dark_mat,   0.06, -4.0),
    (-37, 51,  "box", (0.50, 0.35, 0.40), metal_mat,  0.05,  6.0),
]

bob_phases = [random.uniform(0, 2 * math.pi) for _ in ITEMS]
drift_x    = [random.uniform(-0.003, 0.003)  for _ in ITEMS]
drift_y    = [random.uniform(-0.002, 0.004)  for _ in ITEMS]

print(f"Baking {len(ITEMS)} debris items x {TOTAL_FRAMES+1} frames...")
for idx, (wx, wy, shape, scale, mat, bob_amp, rot_dps) in enumerate(ITEMS):
    item_path = f"/World/Debris/Item_{idx:02d}"

    if shape == "cyl":
        prim = UsdGeom.Cylinder.Define(stage, item_path)
        prim.GetRadiusAttr().Set(scale[0])
        prim.GetHeightAttr().Set(scale[2])
    else:
        prim = UsdGeom.Cube.Define(stage, item_path)

    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mat)
    prim.CreateDoubleSidedAttr(True)

    xf   = UsdGeom.Xformable(prim.GetPrim())
    t_op = xf.AddTranslateOp()
    r_op = xf.AddRotateXYZOp()
    s_op = xf.AddScaleOp()
    s_op.Set(Gf.Vec3f(*scale))

    bp = bob_phases[idx]
    dx = drift_x[idx]
    dy = drift_y[idx]

    for frame in range(TOTAL_FRAMES + 1):
        wz  = water_z(frame)
        t   = frame / FPS
        bob = bob_amp * math.sin(bp + t * 1.8)
        t_op.Set(Gf.Vec3d(wx + dx*frame, wy + dy*frame, wz + bob + scale[2]*0.5), frame)
        r_op.Set(Gf.Vec3f(
            2.5 * math.sin(bp + t * 0.6),
            2.5 * math.cos(bp + t * 0.4),
            rot_dps * t
        ), frame)

    if idx % 10 == 0:
        print(f"  {idx}/{len(ITEMS)} items...")

print(f"\nDone — {len(ITEMS)} floating objects added throughout the halls.")
