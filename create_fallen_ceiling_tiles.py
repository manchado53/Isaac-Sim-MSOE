"""
create_fallen_ceiling_tiles.py

Adds fallen-ceiling-tile anomalies to the MSOE building USD scene.
Writes a new sublayer USD that references the original — source is never modified.

Each anomaly consists of:
  • 2–5 rectangular tile slabs on the floor (tilted at random angles)
  • Ceiling gap marker (thin dark rectangle at ceiling height showing the hole)
  • Debris scatter — small cube/sphere fragments around fallen tiles
  • Optional dust ring around the impact zone

Usage:
    python3 create_fallen_ceiling_tiles.py
        [--output MSOEBuildingV4_ceiling_tiles.usd]
        [--count N]
        [--seed SEED]
        [--min-severity 0.0] [--max-severity 1.0]

Defaults: 5 incidents, seed=42
"""

import argparse
import math
import random
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SOURCE_USD    = Path(__file__).parent / "MSOEBuildingV4_physics.usd"
BUILDING_BASE = (
    "/World/Building"
    "/_22_129___MSOE_Engineering_Building___A___2_25_2026"
    "/Geometry"
)

TILE_W = 0.610   # standard 2x2 ft ceiling tile (0.610 m ≈ 24 in)
TILE_H = 0.005   # 5 mm thick

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_material(
    stage: Usd.Stage,
    path: str,
    diffuse: tuple,
    opacity: float   = 1.0,
    roughness: float = 0.8,
    metallic: float  = 0.0,
    emissive: tuple  = (0.0, 0.0, 0.0),
) -> UsdShade.Material:
    mat    = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor",  Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    shader.CreateInput("opacity",       Sdf.ValueTypeNames.Float).Set(opacity)
    shader.CreateInput("roughness",     Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic",      Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _bind(prim: Usd.Prim, mat: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def _collect_floor_positions(stage: Usd.Stage, limit: int = 93) -> list:
    root = stage.GetPrimAtPath(f"{BUILDING_BASE}/Floors")
    if not root:
        return []
    positions = []
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() == "Mesh":
            m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            positions.append(m.ExtractTranslation())
            if len(positions) >= limit:
                break
    return positions


# ---------------------------------------------------------------------------
# Fallen ceiling tile builder
# ---------------------------------------------------------------------------

def add_fallen_ceiling_tiles(
    stage:        Usd.Stage,
    anomaly_root: str,
    idx:          int,
    position:     Gf.Vec3d,
    severity:     float,
    ceiling_z:    float = 3.0,
) -> None:
    """
    Place a fallen-ceiling-tile anomaly at world-space *position* (floor level).

    severity (0–1) scales number of tiles, tilt amount, and debris density.
    ceiling_z is the height of the ceiling above the floor local origin.
    """
    inc_path = f"{anomaly_root}/CeilingTileIncident_{idx:03d}"
    root     = UsdGeom.Xform.Define(stage, inc_path)
    UsdGeom.XformCommonAPI(root).SetTranslate(Gf.Vec3d(*position))

    looks = f"{inc_path}/Looks"

    # ---- Counts / scales from severity ------------------------------------
    n_tiles   = max(2, int(2 + severity * 3))          # 2–5 tiles
    n_debris  = max(4, int(4 + severity * 16))          # 4–20 debris chunks
    max_tilt  = 15 + severity * 45                      # 15°–60° from horizontal

    # ---- Tile material (acoustic ceiling tile: off-white, matte) ----------
    tile_mat = _make_material(
        stage, f"{looks}/TileMat",
        diffuse=(0.88, 0.86, 0.82),
        roughness=0.92,
    )
    # Cracked/dirty variant for heavily fallen tiles
    cracked_mat = _make_material(
        stage, f"{looks}/CrackedTileMat",
        diffuse=(0.75, 0.72, 0.68),
        roughness=0.95,
    )

    # ---- Fallen tiles on the floor ----------------------------------------
    tile_spread = 0.3 + severity * 0.9   # how scattered they land

    for t in range(n_tiles):
        tile_path = f"{inc_path}/Tile_{t:02d}"
        tile      = UsdGeom.Mesh.Define(stage, tile_path)

        # Tile dimensions: full tile or half-broken piece
        tw = TILE_W * random.uniform(0.5 if severity > 0.5 else 0.8, 1.0)
        td = TILE_W * random.uniform(0.5 if severity > 0.5 else 0.8, 1.0)

        # Flat quad geometry (two triangles)
        hw, hd, hh = tw / 2, td / 2, TILE_H / 2
        pts = [
            Gf.Vec3f(-hw, -hd, -hh), Gf.Vec3f( hw, -hd, -hh),
            Gf.Vec3f( hw,  hd, -hh), Gf.Vec3f(-hw,  hd, -hh),
            Gf.Vec3f(-hw, -hd,  hh), Gf.Vec3f( hw, -hd,  hh),
            Gf.Vec3f( hw,  hd,  hh), Gf.Vec3f(-hw,  hd,  hh),
        ]
        tile.CreatePointsAttr(pts)
        tile.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])
        tile.CreateFaceVertexIndicesAttr([
            0, 1, 2, 3,   # bottom
            4, 5, 6, 7,   # top
            0, 1, 5, 4,   # front
            2, 3, 7, 6,   # back
            1, 2, 6, 5,   # right
            0, 3, 7, 4,   # left
        ])
        tile.CreateNormalsAttr([
            Gf.Vec3f( 0,  0, -1), Gf.Vec3f( 0,  0, -1),
            Gf.Vec3f( 0,  0, -1), Gf.Vec3f( 0,  0, -1),
            Gf.Vec3f( 0,  0,  1), Gf.Vec3f( 0,  0,  1),
            Gf.Vec3f( 0,  0,  1), Gf.Vec3f( 0,  0,  1),
        ])

        # Scatter position on the floor
        ox = random.uniform(-tile_spread, tile_spread)
        oy = random.uniform(-tile_spread, tile_spread)
        oz = hh + 0.001   # just above floor

        # Random tilt — tiles don't land perfectly flat
        tilt_x = random.uniform(-max_tilt, max_tilt)
        tilt_y = random.uniform(-max_tilt, max_tilt)
        rot_z  = random.uniform(0, 360)

        api = UsdGeom.XformCommonAPI(tile.GetPrim())
        api.SetTranslate(Gf.Vec3d(ox, oy, oz))
        api.SetRotate(Gf.Vec3f(tilt_x, tilt_y, rot_z))

        # Physics (rigid body already at rest on floor)
        UsdPhysics.RigidBodyAPI.Apply(tile.GetPrim()).CreateKinematicEnabledAttr(True)
        UsdPhysics.CollisionAPI.Apply(tile.GetPrim())
        UsdPhysics.MassAPI.Apply(tile.GetPrim()).CreateMassAttr(0.45)

        mat = cracked_mat if (severity > 0.6 and t < 2) else tile_mat
        _bind(tile.GetPrim(), mat)

    # ---- Ceiling gap marker (dark rectangle at ceiling, showing hole) ------
    gap_w = TILE_W * (1 + severity * 0.5)     # gap can span >1 tile for high sev
    gap_d = TILE_W * (0.8 + severity * 0.4)

    gap = UsdGeom.Cube.Define(stage, f"{inc_path}/CeilingGap")
    gap.CreateSizeAttr(1.0)
    UsdGeom.XformCommonAPI(gap).SetScale(Gf.Vec3f(gap_w, gap_d, 0.02))
    UsdGeom.XformCommonAPI(gap).SetTranslate(Gf.Vec3d(0, 0, ceiling_z - 0.01))
    _bind(gap.GetPrim(), _make_material(
        stage, f"{looks}/GapMat",
        diffuse=(0.05, 0.05, 0.06),   # dark void above ceiling
        roughness=1.0,
    ))

    # ---- Exposed ceiling grid bars (T-bar grid) ----------------------------
    # Two thin crossing bars visible at the gap
    for bar_idx, (bw, bd, bx, by) in enumerate([
        (gap_w, 0.025, 0.0, 0.0),     # longitudinal bar
        (0.025, gap_d, 0.0, 0.0),     # transverse bar
    ]):
        bar = UsdGeom.Cube.Define(stage, f"{inc_path}/GridBar_{bar_idx}")
        bar.CreateSizeAttr(1.0)
        UsdGeom.XformCommonAPI(bar).SetScale(Gf.Vec3f(bw, bd, 0.03))
        UsdGeom.XformCommonAPI(bar).SetTranslate(Gf.Vec3d(bx, by, ceiling_z - 0.035))
        _bind(bar.GetPrim(), _make_material(
            stage, f"{looks}/GridBarMat_{bar_idx}",
            diffuse=(0.55, 0.55, 0.55),
            roughness=0.4, metallic=0.6,
        ))

    # ---- Debris scatter (tile fragments, dust chunks) ---------------------
    debris_spread = tile_spread + 0.25 + severity * 0.3
    debris_mat = _make_material(
        stage, f"{looks}/DebrisMat",
        diffuse=(0.82, 0.80, 0.76),
        roughness=0.97,
    )
    dust_mat = _make_material(
        stage, f"{looks}/DustMat",
        diffuse=(0.72, 0.70, 0.67),
        opacity=0.60 + severity * 0.25,
        roughness=0.99,
    )

    for d in range(n_debris):
        dx = random.uniform(-debris_spread, debris_spread)
        dy = random.uniform(-debris_spread, debris_spread)
        dz = random.uniform(0.002, 0.015)
        rot = random.uniform(0, 360)

        # Mix of flat shards and small cubes
        chunk_type = random.random()
        db_path = f"{inc_path}/Debris_{d:02d}"

        if chunk_type < 0.5:
            # Flat shard (thin cube)
            chunk = UsdGeom.Cube.Define(stage, db_path)
            chunk.CreateSizeAttr(1.0)
            cw = random.uniform(0.03, 0.18)
            cd = random.uniform(0.03, 0.15)
            ch = random.uniform(0.003, 0.012)
            UsdGeom.XformCommonAPI(chunk).SetScale(Gf.Vec3f(cw, cd, ch))
            UsdGeom.XformCommonAPI(chunk).SetTranslate(Gf.Vec3d(dx, dy, dz))
            UsdGeom.XformCommonAPI(chunk).SetRotate(
                Gf.Vec3f(random.uniform(-20, 20), random.uniform(-20, 20), rot)
            )
            _bind(chunk.GetPrim(), debris_mat)
        else:
            # Dust/powder blob (sphere, flattened)
            blob = UsdGeom.Sphere.Define(stage, db_path)
            blob.CreateRadiusAttr(random.uniform(0.02, 0.08))
            UsdGeom.XformCommonAPI(blob).SetScale(
                Gf.Vec3f(random.uniform(0.8, 1.2), random.uniform(0.8, 1.2), 0.15)
            )
            UsdGeom.XformCommonAPI(blob).SetTranslate(Gf.Vec3d(dx, dy, 0.003))
            _bind(blob.GetPrim(), dust_mat)

    # ---- Dust ring on floor (impact circle) --------------------------------
    dust_r = tile_spread + 0.4 + severity * 0.3
    ring   = UsdGeom.Cylinder.Define(stage, f"{inc_path}/DustRing")
    ring.CreateRadiusAttr(dust_r)
    ring.CreateHeightAttr(0.002)
    ring.CreateAxisAttr(UsdGeom.Tokens.z)
    UsdGeom.XformCommonAPI(ring).SetTranslate(Gf.Vec3d(0, 0, 0.001))
    _bind(ring.GetPrim(), _make_material(
        stage, f"{looks}/DustRingMat",
        diffuse=(0.78, 0.76, 0.72),
        opacity=0.18 + severity * 0.22,
        roughness=0.98,
    ))

    # ---- Metadata ----------------------------------------------------------
    root.GetPrim().SetCustomDataByKey("anomaly:type",       "fallen_ceiling_tiles")
    root.GetPrim().SetCustomDataByKey("anomaly:severity",   severity)
    root.GetPrim().SetCustomDataByKey("anomaly:tile_count", n_tiles)
    root.GetPrim().SetCustomDataByKey("anomaly:debris_count", n_debris)
    root.GetPrim().SetCustomDataByKey("anomaly:ceiling_z_m",  ceiling_z)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "MSOEBuildingV4_ceiling_tiles.usd"),
    )
    parser.add_argument("--count",        type=int,   default=5,
                        help="Number of ceiling-tile incidents to place")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--min-severity", type=float, default=0.2)
    parser.add_argument("--max-severity", type=float, default=1.0)
    parser.add_argument("--ceiling-z",    type=float, default=3.0,
                        help="Ceiling height above floor level in meters (default 3.0)")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Opening source stage: {SOURCE_USD.name}")
    src_stage = Usd.Stage.Open(str(SOURCE_USD))

    print("Collecting floor positions…")
    positions = _collect_floor_positions(src_stage)
    print(f"  {len(positions)} candidate floor positions found")

    if not positions:
        raise RuntimeError("No floor meshes found — check BUILDING_BASE path.")

    selected = random.sample(positions, min(args.count, len(positions)))

    # Build output stage
    print(f"\nCreating: {Path(args.output).name}")
    out = Usd.Stage.CreateNew(args.output)
    out.GetRootLayer().subLayerPaths.append(str(SOURCE_USD))
    UsdGeom.SetStageUpAxis(out, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(out, 1.0)

    anomaly_root = "/World/Anomalies"
    root_xform   = UsdGeom.Xform.Define(out, anomaly_root)
    root_xform.GetPrim().SetCustomDataByKey(
        "anomaly:description", "Fallen ceiling tile anomalies"
    )

    print(f"\nPlacing {len(selected)} ceiling-tile incident(s):")
    for i, pos in enumerate(selected):
        severity = random.uniform(args.min_severity, args.max_severity)
        add_fallen_ceiling_tiles(out, anomaly_root, i, pos, severity, args.ceiling_z)
        print(
            f"  CeilingTileIncident_{i:03d}  severity={severity:.2f}"
            f"  @ ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
        )

    out.Save()
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
