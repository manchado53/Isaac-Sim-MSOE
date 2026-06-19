#!/usr/bin/env python3
"""
Cracked-glass single-panel preview renderer.

Picks N random glass panels from a building USD and renders each one
multiple times — once with an intact OmniGlass material and once per
crack variant — so you can visually inspect how each variant looks
when bound to real glass through Isaac Sim's RTX path tracer.

Used to generate the Layer 4 images in docs/cracked-glass-pipeline.md.
Bypasses Replicator's BasicWriter (which doesn't flush its async queue
before detach in our pipeline) and instead reads RGB pixels directly
from a `rep` annotator into PIL.

Usage:
    ./python.sh SyntheticDataGen/cracked_glass/cracked_glass_preview.py
    ./python.sh SyntheticDataGen/cracked_glass/cracked_glass_preview.py \\
        --variants v4,v5 --n-panels 5 --distance 3.0
"""

import argparse
import json
import os
import random
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _PROJ)
sys.path.insert(0, _SCRIPT_DIR)


def parse_args():
    p = argparse.ArgumentParser(
        description="Cracked-glass single-panel preview renderer"
    )
    p.add_argument("--usd", default=os.path.join(
        _PROJ, "assets", "environments",
        "InnovationCenter", "InnovationCenter.usd"))
    p.add_argument("--variants", default="v1,v2,v3,v4",
                   help="Comma-separated crack variant names to render. "
                        "Each must have crack_normal_<v>.png in the texture "
                        "dir. (default: v1,v2,v3,v4)")
    p.add_argument("--n-panels", type=int, default=3,
                   help="Number of random glass panels to sample (default 3)")
    p.add_argument("--distance", type=float, default=2.5,
                   help="Camera distance from each panel in stage units "
                        "(default 2.5 — meters for InnovationCenter.usd)")
    p.add_argument("--seed", type=int, default=7,
                   help="Random seed for panel selection (default 7 matches "
                        "the doc's reference renders)")
    p.add_argument("--output", default=None,
                   help="Output dir for PNGs (default: "
                        "datasets/cracked_glass/preview_<variants>)")
    p.add_argument("--resolution", default="1280x720",
                   help="Render resolution as WxH (default 1280x720)")
    return p.parse_args()


args = parse_args()

# -----------------------------------------------------------------------------
# Launch Isaac Sim before any pxr/omni imports
# -----------------------------------------------------------------------------
from isaacsim import SimulationApp  # noqa: E402
simulation_app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import omni.usd  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from pxr import Sdf, Usd, UsdGeom, UsdShade  # noqa: E402

from isaacsim.core.api import World  # noqa: E402
from sim_utils.building_physics import (  # noqa: E402
    setup_physics_scene,
    _find_mesh_descendant,
)
from glass_anomaly import (  # noqa: E402
    create_cracked_glass_materials,
    enumerate_glass_prims,
)
from sim_utils.test_cameras import (  # noqa: E402
    CAMERA_FOCAL_LENGTH,
)

TEXTURE_DIR = os.path.join(_PROJ, "assets", "materials", "cracked_glass")

variant_list = tuple(v.strip() for v in args.variants.split(",") if v.strip())
if not variant_list:
    print("[preview] ERROR: --variants must list at least one variant",
          flush=True)
    sys.exit(1)

w, h = (int(x) for x in args.resolution.lower().split("x"))
RESOLUTION = (w, h)

output_dir = args.output or os.path.join(
    _PROJ, "datasets", "cracked_glass",
    f"preview_{'_'.join(variant_list)}",
)
os.makedirs(output_dir, exist_ok=True)
print(f"[preview] USD: {args.usd}", flush=True)
print(f"[preview] Variants: {variant_list}", flush=True)
print(f"[preview] N panels: {args.n_panels}", flush=True)
print(f"[preview] Output: {output_dir}", flush=True)

random.seed(args.seed)


def bind_visual_material_deep(stage, prim, material_path):
    """Bind at both the Xform wrapper AND any Mesh descendant.

    Some panels in InnovationCenter.usd have a material bound directly on
    the inner Mesh prim; binding 'strongerThanDescendants' at the wrapper
    Xform doesn't override a child Mesh binding, so the swap silently
    no-ops. Walking to the Mesh descendant + unbinding any pre-existing
    default-purpose binding fixes it. Documented as Gotcha 1 in
    docs/cracked-glass-pipeline.md.
    """
    mat = UsdShade.Material(stage.GetPrimAtPath(material_path))
    targets = [prim, _find_mesh_descendant(prim, max_depth=4)]
    for target in targets:
        if target is None or not target.IsValid():
            continue
        api = UsdShade.MaterialBindingAPI.Apply(target)
        try:
            api.UnbindDirectBinding()
        except Exception:
            pass
        api.Bind(mat, UsdShade.Tokens.strongerThanDescendants)


def panel_pose(prim, distance, bbox_cache, building_bbox=None):
    """Compute a camera pose looking at the panel, placed on the INSIDE
    of the building.

    If ``building_bbox`` is given, evaluate the two candidate camera
    positions (panel_center ± panel_normal × distance) and pick the one
    closer to the building centroid — that's the one inside. Without a
    bbox, falls back to the old -axis convention (coin-flip but
    consistent with the legacy behaviour).
    """
    bb = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    mn, mx = bb.GetMin(), bb.GetMax()
    c = (
        (mn[0] + mx[0]) / 2,
        (mn[1] + mx[1]) / 2,
        (mn[2] + mx[2]) / 2,
    )
    dims = [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]]
    axis = min(range(3), key=lambda i: dims[i])  # thinnest = panel normal
    n = [0.0, 0.0, 0.0]
    n[axis] = 1.0
    cam_pos = (c[0] + n[0] * distance, c[1] + n[1] * distance, c[2] + n[2] * distance)
    cam_neg = (c[0] - n[0] * distance, c[1] - n[1] * distance, c[2] - n[2] * distance)

    if building_bbox is not None:
        bmn, bmx = building_bbox.GetMin(), building_bbox.GetMax()
        bc = ((bmn[0] + bmx[0]) / 2, (bmn[1] + bmx[1]) / 2, (bmn[2] + bmx[2]) / 2)
        d_pos = sum((cam_pos[i] - bc[i]) ** 2 for i in range(3))
        d_neg = sum((cam_neg[i] - bc[i]) ** 2 for i in range(3))
        cam = cam_pos if d_pos <= d_neg else cam_neg
    else:
        cam = cam_neg  # legacy: -axis side
    return cam, c, dims


def is_interior_partition(prim, distance, bbox_cache, building_bbox, margin=1.0):
    """Return True if BOTH candidate camera positions (panel ± normal × dist)
    land inside the building bounding box, i.e. the panel partitions two
    interior spaces rather than facing outward onto the exterior.

    ``margin`` shrinks the bbox on each side so panels near the exterior
    wall aren't falsely classified as interior.
    """
    bb = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    mn, mx = bb.GetMin(), bb.GetMax()
    c = ((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2)
    dims = [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]]
    axis = min(range(3), key=lambda i: dims[i])
    bmn = building_bbox.GetMin()
    bmx = building_bbox.GetMax()

    def inside(p):
        return all(bmn[i] + margin <= p[i] <= bmx[i] - margin for i in range(3))

    p_plus = list(c); p_plus[axis] += distance
    p_minus = list(c); p_minus[axis] -= distance
    return inside(p_plus) and inside(p_minus)


def make_emissive_crack_material(stage, mat_path, variant, texture_dir,
                                 rgba_tex_path):
    """OmniEmissive material for the self-lit crack-overlay plane.

    `OmniEmissive.mdl` is a dedicated emissive shader that reads the
    ALPHA channel of `emissive_color_texture` as a cutout mask — so the
    pre-baked crack_emissive_<variant>.png (RGB=white, A=crack pattern)
    gives us "glowing cracks, transparent elsewhere" in one texture slot
    with no opacity-switch acrobatics. This avoids the brittle
    enable_opacity / opacity_mode / opacity_threshold plumbing on
    OmniPBR which didn't fire reliably in our setup.
    """
    UsdGeom.Scope.Define(stage, Sdf.Path(mat_path).GetParentPath())
    mat = UsdShade.Material.Define(stage, mat_path)
    sh = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    sh.CreateIdAttr("mdlMaterial")
    sh.SetSourceAsset("OmniEmissive.mdl", "mdl")
    sh.SetSourceAssetSubIdentifier("OmniEmissive", "mdl")

    sh.CreateInput("emissive_color_texture",
                   Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(rgba_tex_path))
    sh.CreateInput("emission_intensity",
                   Sdf.ValueTypeNames.Float).Set(3000.0)

    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateDisplacementOutput("mdl").ConnectToSource(
        sh.ConnectableAPI(), "out")
    mat.CreateVolumeOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    return mat


def create_crack_overlay_plane(stage, panel_prim, bbox_cache, mat_path,
                               overlay_path, offset=0.02):
    """Build a thin quad mesh in the panel plane, offset toward the
    interior (camera side), and bind the emissive crack material to it.

    The quad is axis-aligned to the panel's thinnest axis (same logic
    `panel_pose` uses to pick the panel normal). UVs 0..1 map the crack
    PNG across the full panel.
    """
    bb = bbox_cache.ComputeWorldBound(panel_prim).ComputeAlignedRange()
    mn, mx = bb.GetMin(), bb.GetMax()
    c = [(mn[i] + mx[i]) / 2.0 for i in range(3)]
    dims = [mx[i] - mn[i] for i in range(3)]

    axis = min(range(3), key=lambda i: dims[i])
    plane_axes = [i for i in range(3) if i != axis]
    hw = dims[plane_axes[0]] / 2.0
    hh = dims[plane_axes[1]] / 2.0

    # `panel_pose` put the camera at c + n*distance with n[axis] = -1,
    # so "toward the interior" means nudging the overlay by -axis.
    oc = list(c)
    oc[axis] -= offset

    u = [0.0, 0.0, 0.0]; u[plane_axes[0]] = 1.0
    v = [0.0, 0.0, 0.0]; v[plane_axes[1]] = 1.0

    def corner(su, sv):
        return (
            oc[0] + u[0] * su * hw + v[0] * sv * hh,
            oc[1] + u[1] * su * hw + v[1] * sv * hh,
            oc[2] + u[2] * su * hw + v[2] * sv * hh,
        )

    mesh = UsdGeom.Mesh.Define(stage, overlay_path)
    mesh.CreatePointsAttr([
        corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1),
    ])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    # Render both faces — avoids the overlay vanishing when the quad's
    # generated normal happens to point away from the camera.
    mesh.CreateDoubleSidedAttr(True)
    # Explicit extent so Hydra doesn't cull the prim as empty-bounds.
    pts = mesh.GetPointsAttr().Get()
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
    mesh.CreateExtentAttr([
        (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))])

    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.Float2Array, UsdGeom.Tokens.varying)
    st.Set([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])

    mat_prim = stage.GetPrimAtPath(mat_path)
    if mat_prim and mat_prim.IsValid():
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
            UsdShade.Material(mat_prim),
            UsdShade.Tokens.strongerThanDescendants)
    return overlay_path


def make_intact_material(stage, path="/World/Looks/Intact"):
    """Author a clean-glass OmniGlass baseline with low frosting."""
    UsdGeom.Scope.Define(stage, Sdf.Path(path).GetParentPath())
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
    sh.CreateIdAttr("mdlMaterial")
    sh.SetSourceAsset("OmniGlass.mdl", "mdl")
    sh.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
    sh.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(
        (0.90, 0.95, 0.97))
    sh.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.491)
    sh.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(0.02)
    sh.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(0.003)
    sh.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateDisplacementOutput("mdl").ConnectToSource(
        sh.ConnectableAPI(), "out")
    mat.CreateVolumeOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    return path


# -----------------------------------------------------------------------------
# Scene setup
# -----------------------------------------------------------------------------
if not os.path.isfile(args.usd):
    print(f"[preview] ERROR: USD not found: {args.usd}", flush=True)
    sys.exit(2)

world = World()
stage = omni.usd.get_context().get_stage()

bp = stage.DefinePrim("/World/Building", "Xform")
bp.GetReferences().AddReference(args.usd, primPath=Sdf.Path("/World/Building"))
setup_physics_scene(stage)

# Lighting: Tanner's `bright_fluorescent` preset (Option B), lifted from
# origin/tanner_vlm:simulations/robot/door_static_collection.py:190-220.
#   * DomeLight @ 25000 intensity / 5500K — ambient daylight fill. 5x
#     our previous dome because path-tracing occlusion by the building
#     ceiling swallows most dome contribution; the high value ensures
#     meaningful light reaches even deep interior spaces.
#   * DistantLight @ 6500K — gentle sun for side-modelling.
#   * Per-panel SphereLight @ 400000 / 4500K added in the render loop
#     below, positioned 1.5 m above each panel (Tanner's fill pattern).
#     Small 0.1 m radius acts like a point source so no floor-spot
#     artifact — the old 2x2 RectLight array was what produced those.
from pxr import UsdLux  # noqa: E402
dome = UsdLux.DomeLight.Define(stage, "/World/PreviewDome")
dome.CreateIntensityAttr(25000.0)
dome.CreateColorTemperatureAttr(5500.0)
dome.CreateEnableColorTemperatureAttr(True)

sun = UsdLux.DistantLight.Define(stage, "/World/PreviewSun")
sun.CreateIntensityAttr(2500.0)
sun.CreateAngleAttr(0.5)
sun.CreateColorTemperatureAttr(6500.0)
sun.CreateEnableColorTemperatureAttr(True)
UsdGeom.XformCommonAPI(sun.GetPrim()).SetRotate((-45.0, 30.0, 0.0))

# Non-occluded dome strategy — lets the DomeLight contribute inside
# closed rooms that the path tracer would otherwise block.
import carb  # noqa: E402
carb.settings.get_settings().set("/rtx/domeLight/upperLowerStrategy", 3)
# Extra path-traced bounces so light from the bright curtain walls
# reaches deep atrium interiors. Default is 4; 8 is the sweet spot for
# interior scenes per NVIDIA's RTX best-practices notes.
carb.settings.get_settings().set("/rtx/pathTracing/maxBounces", 8)

# Author the intact baseline + each requested cracked variant
intact_path = make_intact_material(stage)
cracked_paths = {}
for v in variant_list:
    paths = create_cracked_glass_materials(
        stage,
        texture_dir=TEXTURE_DIR,
        root_path=f"/World/Looks/Crack_{v}",
        variants=(v,),
    )
    if not paths:
        print(f"[preview] WARN: variant '{v}' produced no material "
              f"(missing crack_normal_{v}.png?)", flush=True)
        continue
    cracked_paths[v] = paths[0]

if not cracked_paths:
    print("[preview] ERROR: no valid variants — aborting", flush=True)
    sys.exit(3)

# Pre-create one emissive-overlay material per variant. The per-panel
# overlay Mesh is created on demand during the render loop and reuses
# these shared materials — no material churn per panel.
#
# OmniEmissive.mdl reads the alpha channel of emissive_color_texture as
# the cutout mask, so we bake an RGBA file per variant: RGB = white
# everywhere, alpha = the crack pattern (cracks opaque, intact fully
# transparent). The RGBA file lives in the texture dir alongside the
# other crack PNGs.
emissive_mat_paths = {}
for v in variant_list:
    rough_file = os.path.join(TEXTURE_DIR, f"crack_roughness_{v}.png")
    if not os.path.isfile(rough_file):
        print(f"[preview] WARN: no crack_roughness_{v}.png — skipping emissive "
              f"overlay for {v}", flush=True)
        continue
    rgba_file = os.path.join(TEXTURE_DIR, f"crack_emissive_{v}.png")
    # Always re-bake so tweaks to the binarise/dilate logic take effect
    # without the old cached RGBA file masking the change.
    mask = Image.open(rough_file).convert("L")
    arr = np.asarray(mask)
    # Binarise: any pixel in a crack becomes fully opaque in the mask.
    # Fine crack lines (v4, tempered) were losing emission because their
    # sub-threshold gray values were dimming the emission below visibility.
    binary = (arr > 30).astype(np.uint8)
    # Dilate a few pixels so thin cracks read more clearly on a 2048²
    # texture mapped to a 1.5m panel (each px is ~0.75mm; +2px widens
    # the emission band by ~3mm which is enough to be visible at the
    # panel-framing viewing distance).
    try:
        from scipy.ndimage import binary_dilation
        binary = binary_dilation(binary, iterations=2).astype(np.uint8)
    except Exception:
        pass
    alpha = Image.fromarray((binary * 255).astype(np.uint8), mode="L")
    rgba = Image.new("RGBA", mask.size, (255, 255, 255, 0))
    rgba.putalpha(alpha)
    rgba.save(rgba_file)
    print(f"[preview] wrote {os.path.basename(rgba_file)} (RGBA mask, "
          f"{int(binary.sum())} opaque px)", flush=True)
    emissive_mat_paths[v] = f"/World/Looks/CrackOverlayMat_{v}"
    make_emissive_crack_material(
        stage, emissive_mat_paths[v], v, TEXTURE_DIR, rgba_file)

glass = enumerate_glass_prims(stage)
print(f"[preview] Glass prims discovered: {len(glass)}", flush=True)
world.reset()

bbox_cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(),
    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
)

# Compute the building's overall bbox — used both to normalise the camera
# to the INSIDE of the building (consistent school-day interior POV) and
# to filter out panels that face outward (exterior curtain walls). This
# is more reliable than trusting the parent prim name, because many
# exterior panels share the same `Curtain_Panels_*` naming as interior
# partitions.
building_prim = stage.GetPrimAtPath("/World/Building")
building_bbox = bbox_cache.ComputeWorldBound(building_prim).ComputeAlignedRange()

# Filter: (1) area ≥ 1 m², (2) inner mesh not a shadowbox/railing,
# (3) panel is an interior partition (both sides inside the building bbox),
# (4) GL_1 double-pane only — excludes GL_2 pattern-film (etched/frosted
#     from factory, different optical behaviour) and GL_1_1 laminated
#     safety glass, which render differently from our validated GL_1
#     double-pane reference (panel 5 in earlier runs).
_SHADOWBOX_MESH_PATTERNS = ("Spandrel", "Shadowbox", "GL_3", "_railing_")
_KEEP_MESH_PREFIX = "Glass_GL_1_"    # excludes "Glass___Laminated___GL_1_1_*"
                                      # and "Glass___Pattern_Film__GL_2_*"
candidates = []
rejected_exterior = 0
rejected_shadowbox = 0
rejected_small = 0
rejected_wrong_glass = 0
for p in glass:
    _, _, d = panel_pose(p, args.distance, bbox_cache)
    if sorted(d)[1] * sorted(d)[2] <= 1.0:
        rejected_small += 1
        continue
    mesh = _find_mesh_descendant(p, max_depth=4)
    if mesh is not None:
        mname = mesh.GetName()
        if any(tag in mname for tag in _SHADOWBOX_MESH_PATTERNS):
            rejected_shadowbox += 1
            continue
        if not mname.startswith(_KEEP_MESH_PREFIX):
            rejected_wrong_glass += 1
            continue
    if not is_interior_partition(p, args.distance, bbox_cache, building_bbox):
        rejected_exterior += 1
        continue
    candidates.append(p)
random.shuffle(candidates)
panels = candidates[: args.n_panels]
print(f"[preview] Candidate pool: {len(candidates)} GL_1 double-pane interior "
      f"partitions (rejected {rejected_exterior} exterior / "
      f"{rejected_shadowbox} shadowbox / {rejected_wrong_glass} non-GL_1 glass "
      f"/ {rejected_small} fragments)", flush=True)
print(f"[preview] Selected {len(panels)} panels", flush=True)

# Build the variant list in render order: intact first, then each cracked
render_variants = [("intact", intact_path)]
for v in variant_list:
    if v in cracked_paths:
        render_variants.append((v, cracked_paths[v]))

# -----------------------------------------------------------------------------
# Per-panel render loop
# -----------------------------------------------------------------------------
for idx, panel in enumerate(panels):
    cam_pos, look_at, dims = panel_pose(panel, args.distance, bbox_cache,
                                        building_bbox=building_bbox)
    mesh = _find_mesh_descendant(panel, max_depth=4)
    print(f"[preview] panel {idx}: {panel.GetName()} "
          f"dims=({dims[0]:.2f}, {dims[1]:.2f}, {dims[2]:.2f}) "
          f"mesh={mesh.GetName() if mesh else None}", flush=True)

    # Per-panel fill SphereLight (Tanner's bright_fluorescent formula,
    # matching its exact position: camera XY, camera Z + 1.5m). Placing
    # it above the camera — NOT above the panel — keeps the light on
    # the camera-side of the glass so it actually hits the face we're
    # photographing, instead of the back of the panel. Tiny 0.1 m radius
    # = near-point source, so no floor-spot disc artifact.
    fill = UsdLux.SphereLight.Define(
        stage, Sdf.Path(f"/World/PreviewFill_{idx}"))
    fill.CreateIntensityAttr(400000.0)
    fill.CreateRadiusAttr(0.1)
    fill.CreateColorTemperatureAttr(4500.0)
    fill.CreateEnableColorTemperatureAttr(True)
    UsdGeom.XformCommonAPI(fill.GetPrim()).SetTranslate(
        (cam_pos[0], cam_pos[1], cam_pos[2] + 1.5))

    cam_node = rep.create.camera(
        position=cam_pos, look_at=look_at,
        focal_length=CAMERA_FOCAL_LENGTH, horizontal_aperture=20.955,
        name=f"PreviewCam_{idx}",
    )
    rp = rep.create.render_product(cam_node, RESOLUTION)
    rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_ann.attach([rp])

    for vname, mpath in render_variants:
        bind_visual_material_deep(stage, panel, mpath)

        # Emissive overlay: create a thin self-lit plane in front of the
        # panel whenever we're rendering a cracked variant. The overlay
        # makes cracks visible regardless of dome/exterior brightness by
        # giving them their own light source. Removed after each render
        # so it doesn't leak into the next variant.
        overlay_path = None
        if vname in emissive_mat_paths:
            overlay_path = (
                f"/World/CrackOverlay_p{idx:02d}_{vname}"
            )
            create_crack_overlay_plane(
                stage, panel, bbox_cache,
                mat_path=emissive_mat_paths[vname],
                overlay_path=overlay_path,
            )

        # Step a few times to let materials + shaders settle
        for _ in range(4):
            rep.orchestrator.step(rt_subframes=4)
        data = rgb_ann.get_data()
        if data is None or not hasattr(data, "shape"):
            print(f"  [WARN] no RGB data for {vname}", flush=True)
            continue
        img = np.asarray(data)
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        dst = os.path.join(output_dir, f"panel{idx:02d}_{vname}.png")
        Image.fromarray(img.astype(np.uint8)).save(dst)
        print(f"  saved {os.path.basename(dst)}", flush=True)

        # Tear down the overlay so the next variant starts clean.
        if overlay_path is not None:
            stage.RemovePrim(Sdf.Path(overlay_path))

    # Detaching the annotator throws a benign HydraTexture warning on
    # Isaac Sim 5.1; swallow it so the loop continues.
    try:
        rgb_ann.detach([rp])
    except Exception as e:
        print(f"  [detach warn] {e}", flush=True)

# -----------------------------------------------------------------------------
# Sidecar manifest
# -----------------------------------------------------------------------------
meta = {
    "usd": args.usd,
    "panels": [str(p.GetPath()) for p in panels],
    "variants": [v[0] for v in render_variants],
    "binding": "bind_visual_material_deep (Xform + Mesh descendant)",
    "seed": args.seed,
    "distance": args.distance,
    "resolution": RESOLUTION,
}
with open(os.path.join(output_dir, "preview_manifest.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(f"[preview] DONE. Manifest: {output_dir}/preview_manifest.json",
      flush=True)
simulation_app.close()
