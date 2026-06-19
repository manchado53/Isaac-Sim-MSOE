"""
Shared geometry + orbit helpers for the panorama scripts.

Extracted so both `simulations/environment/panorama_360.py` (ground-level
building panorama) and `simulations/environment/field_panorama.py`
(elevated drone-style field panorama) can reuse the math without either
script triggering the other's `SimulationApp()` at import time.

All functions are pure Python / pxr / numpy — no Isaac Sim runtime needed.
They can be imported safely after `SimulationApp({"headless": True})` has
been instantiated in the caller.
"""

import math

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdLux


# Revit USD exports contain a `/Geometry/FamilyData/` subtree of unplaced
# family prototypes (stray coords at origin); those meshes back the
# visible instances elsewhere, so we skip them from framing math but do
# NOT hide them (hiding would break the instances).
_BBOX_SKIP_SUBSTRINGS = ("/FamilyData/",)


def hide_matching_prims(stage, patterns, root_path="/World/Building"):
    """Hide any prim under root_path whose path contains one of `patterns`.

    Returns (hidden_paths_set, per_pattern_counts_dict). Hidden paths
    should be passed to compute_building_geometry() so landscape/trees
    don't bias the camera math.
    """
    hidden = set()
    per_pattern = {p: 0 for p in patterns}
    if not patterns:
        return hidden, per_pattern
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(root_path):
            continue
        for pat in patterns:
            if pat and pat in path:
                imageable = UsdGeom.Imageable(prim)
                if imageable:
                    imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                hidden.add(path)
                per_pattern[pat] += 1
                break
    return hidden, per_pattern


def compute_building_geometry(building_prim, bbox_cache, hidden_paths=None):
    """Volume-weighted centroid + bbox + XY corners of every visible mesh.

    Returns (centroid_xyz, bbox_min, bbox_max, mesh_count, corners_xy).
    corners_xy is shape (N*4, 2) — the horizontal bbox corners of each
    included mesh, used downstream to compute per-angle facade distances
    for elliptical (facade-aware) camera orbits on elongated buildings.

    Meshes under `/FamilyData/` subtrees are skipped (Revit prototypes at
    stray coords that inflate the bbox). Hidden prims (from
    hide_matching_prims) are also excluded.
    """
    hidden_paths = hidden_paths or set()
    positions = []
    weights = []
    corners_xy = []
    all_min = np.array([np.inf, np.inf, np.inf])
    all_max = np.array([-np.inf, -np.inf, -np.inf])
    mesh_count = 0
    skipped_family_data = 0

    for prim in Usd.PrimRange(building_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        p = str(prim.GetPath())
        if any(p == h or p.startswith(h + "/") for h in hidden_paths):
            continue
        if any(skip in p for skip in _BBOX_SKIP_SUBSTRINGS):
            skipped_family_data += 1
            continue
        elem_bbox = bbox_cache.ComputeWorldBound(prim)
        elem_range = elem_bbox.ComputeAlignedRange()
        if elem_range.IsEmpty():
            continue
        emin = elem_range.GetMin()
        emax = elem_range.GetMax()
        emin_np = np.array([emin[0], emin[1], emin[2]])
        emax_np = np.array([emax[0], emax[1], emax[2]])

        center = (emin_np + emax_np) / 2.0
        extent = emax_np - emin_np
        vol = max(float(extent[0] * extent[1] * extent[2]), 0.001)
        positions.append(center)
        weights.append(vol)
        corners_xy.append([emin_np[0], emin_np[1]])
        corners_xy.append([emax_np[0], emin_np[1]])
        corners_xy.append([emin_np[0], emax_np[1]])
        corners_xy.append([emax_np[0], emax_np[1]])
        all_min = np.minimum(all_min, emin_np)
        all_max = np.maximum(all_max, emax_np)
        mesh_count += 1

    if mesh_count == 0:
        return None, None, None, 0, None
    print(f"  skipped {skipped_family_data} Revit FamilyData prototype meshes", flush=True)

    positions = np.array(positions)
    weights = np.array(weights)
    centroid = np.average(positions, axis=0, weights=weights)
    return centroid, all_min, all_max, mesh_count, np.array(corners_xy)


def sample_orbit_geometry(centroid_xy, corners_xy, num_samples=72):
    """For each of `num_samples` evenly-spaced view angles around the
    centroid, return:

      facade_along[a]    -- how far the building reaches along +view_dir
                            (camera must be at least this far to clear
                            the building along that direction).
      silhouette_half[a] -- half the apparent building width perpendicular
                            to the view direction (the silhouette that must
                            fit in the camera frame horizontally).

    Returns (sample_angles_rad, facade_along_arr, silhouette_half_arr).
    """
    angles = np.linspace(0.0, 2.0 * math.pi, num_samples, endpoint=False)
    offsets = corners_xy - np.asarray(centroid_xy, dtype=float)

    view_dirs = np.stack([np.cos(angles), np.sin(angles)], axis=0)    # (2, A)
    perp_dirs = np.stack([-np.sin(angles), np.cos(angles)], axis=0)   # (2, A)

    proj_along = offsets @ view_dirs                                  # (N*4, A)
    proj_perp = offsets @ perp_dirs                                   # (N*4, A)

    facade_along = proj_along.max(axis=0)
    silhouette_half = (proj_perp.max(axis=0) - proj_perp.min(axis=0)) / 2.0
    return angles, facade_along, silhouette_half


def radius_at(angle_rad, sample_angles, facade_along, silhouette_half,
              framing_mult, min_clearance):
    """Linear interpolation over the sampled orbit geometry; returns the
    orbit radius required at that angle.

    radius = facade_along + silhouette_half * framing_mult + min_clearance
    """
    two_pi = 2.0 * math.pi
    a = angle_rad % two_pi
    n = len(sample_angles)
    step = two_pi / n
    idx = a / step
    i0 = int(idx) % n
    i1 = (i0 + 1) % n
    frac = idx - int(idx)
    fa = facade_along[i0] * (1.0 - frac) + facade_along[i1] * frac
    sh = silhouette_half[i0] * (1.0 - frac) + silhouette_half[i1] * frac
    return fa + sh * framing_mult + min_clearance


def look_at_orientation(cam_pos, target):
    """Quaternion (as numpy array) so a camera at cam_pos looks at target.

    Uses Isaac Sim's euler-to-quat convention: [roll=0, pitch, yaw],
    where yaw is atan2(dy, dx) and pitch tilts down when target is below.
    """
    # Lazy import to avoid Isaac Sim dependency at module load for callers
    # that don't need look_at_orientation (but all panorama scripts do).
    import isaacsim.core.utils.numpy.rotations as rot_utils

    dx = target[0] - cam_pos[0]
    dy = target[1] - cam_pos[1]
    dz = target[2] - cam_pos[2]
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(-dz, math.sqrt(dx * dx + dy * dy)))
    return rot_utils.euler_angles_to_quats(
        np.array([0.0, pitch, yaw]), degrees=True
    )


def add_exterior_lighting(stage, sun_intensity=3500.0, sky_intensity=1000.0):
    """Standard outdoor sun + sky dome for exterior building shots."""
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(sun_intensity)
    sun.CreateAngleAttr(1.0)
    sun_xf = UsdGeom.Xformable(sun.GetPrim())
    sun_xf.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 30.0, 0.0))

    sky = UsdLux.DomeLight.Define(stage, "/World/Sky")
    sky.CreateIntensityAttr(sky_intensity)
