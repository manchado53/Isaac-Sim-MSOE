"""
Glass-anomaly utilities for building visual anomaly detection datasets.

Enumerates glass prims in a Revit-exported building, builds OmniPBR-style
"cracked glass" materials from authored normal + opacity PNGs, and applies
them per-prim along with semantic labels so Replicator's segmentation
annotator can emit labeled cracked/intact masks.
"""

import os
from typing import Iterable, List, Optional, Sequence, Tuple

from pxr import Sdf, Usd, UsdGeom, UsdShade, Gf

from sim_utils.building_physics import (
    PrimCategory,
    _find_mesh_descendant,
    categorize_prim,
    decode_tn_name,
    iter_building_elements,
)


# ---------------------------------------------------------------------------
# Glass prim enumeration
# ---------------------------------------------------------------------------

# Name substrings that identify the glass panels we want to crack.
# GL_1 = standard double-pane glazing (the main interior/exterior windows).
# CurtainPanels = curtain wall subpanels (large exterior glass).
DEFAULT_GLASS_NAME_FILTERS: Tuple[str, ...] = (
    "GL_1",
    "Glass_GL_1",
    "Curtain",
    "System_Panel_Glazed",
)


def enumerate_glass_prims(
    stage: Usd.Stage,
    building_path: str = "/World/Building",
    name_filters: Sequence[str] = DEFAULT_GLASS_NAME_FILTERS,
    exclude_filters: Sequence[str] = ("GL_3", "GL_1_1", "Mullion"),
) -> List[Usd.Prim]:
    """Return Xform prims that look like crackable glass panels.

    Walks the building via ``iter_building_elements`` (so it works with both
    the category-folder Innovation Center USD and the hierarchical
    engineering building export), keeps only elements classified as
    ``STATIC_GLAZING``, and then applies name-substring filters.

    Args:
        stage: USD stage with the building reference already added.
        building_path: Prim path of the building root. Default matches the
            ``AddReference`` pattern used across demos.
        name_filters: Substrings that must appear in the decoded prim or
            category name for the prim to be kept.
        exclude_filters: Substrings that force exclusion (applied after
            ``name_filters``). Defaults exclude shadowbox/laminated glass
            and mullions so we don't accidentally crack frames.

    Returns:
        List of Xform prims ready for material binding.
    """
    building_prim = stage.GetPrimAtPath(building_path)
    if not building_prim or not building_prim.IsValid():
        return []

    result: List[Usd.Prim] = []
    for _level, category_name, element in iter_building_elements(stage, building_prim):
        category = categorize_prim(
            element.GetName(), category_hint=category_name
        )
        if category is not PrimCategory.STATIC_GLAZING:
            continue

        decoded_name = decode_tn_name(element.GetName())
        decoded_cat = decode_tn_name(category_name) if category_name else ""
        haystack = f"{decoded_cat}/{decoded_name}"

        if not any(f in haystack for f in name_filters):
            continue
        if any(f in haystack for f in exclude_filters):
            continue
        result.append(element)

    return result


# ---------------------------------------------------------------------------
# Cracked-glass OmniPBR material authoring
# ---------------------------------------------------------------------------

# OmniGlass.mdl is Omniverse's physically-based glass shader; it supports
# true transparency with index of refraction, unlike OmniPBR's flat
# alpha-blend. See
# https://docs.omniverse.nvidia.com/materials-and-rendering/latest/templates/OmniGlass.html
OMNIGLASS_MDL_URL = "OmniGlass.mdl"
OMNIGLASS_MDL_NAME = "OmniGlass"


def create_cracked_glass_material(
    stage: Usd.Stage,
    material_path: str,
    normal_texture: str,
    opacity_texture: Optional[str] = None,  # unused for OmniGlass but kept for API stability
    roughness_texture: Optional[str] = None,
    glass_color: Tuple[float, float, float] = (0.82, 0.90, 0.94),
    glass_ior: float = 1.491,
    frosting_roughness: float = 0.03,
    depth: float = 0.005,
    thin_walled: bool = True,
    crack_hits: int = 0,
) -> UsdShade.Material:
    """Define a single OmniGlass "cracked glass" material.

    Uses OmniGlass.mdl so the panel reads as real transparent glass; the
    normal map from the crack pattern drives visible crack shading on
    what is otherwise a clear surface.

    Args:
        stage: USD stage.
        material_path: Prim path under which the Material is defined.
        normal_texture: Absolute path to the crack normal map (PNG).
        opacity_texture: Unused (signature kept for backward compat).
        glass_color: Tint applied to refracted light. Slight blue-green
            cast matches architectural glass.
        glass_ior: Index of refraction. 1.491 = soda-lime glass.
        frosting_roughness: 0 = perfect mirror, 1 = fully frosted. Bumped
            slightly on the cracked variant so cracks scatter light.
        depth: Path-length through the glass in stage units; used for
            the color absorption calculation. Small values keep the tint
            subtle.
        thin_walled: Treat the panel as a thin sheet (no refraction through
            a volume). Correct for window glass which is modeled as a
            single mesh face.
        crack_hits: Controls how many impact points appear on the panel.
            0 = default UV tiling (whatever the panel's UVs give, typically
                2-4 impacts — produces the dense shatter look).
            1 = single impact (projected, no tiling).
            4 = 2x2 grid of impacts.
            9 = 3x3 grid of impacts.
            N = ceil(sqrt(N)) x ceil(sqrt(N)) grid.
            For dataset variety, randomly pick from [0, 1, 4, 9].
    """
    import math

    material = UsdShade.Material.Define(stage, material_path)

    shader_path = f"{material_path}/Shader"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("mdlMaterial")
    shader.SetSourceAsset(OMNIGLASS_MDL_URL, "mdl")
    shader.SetSourceAssetSubIdentifier(OMNIGLASS_MDL_NAME, "mdl")

    shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*glass_color))
    shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(glass_ior)
    shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(frosting_roughness)
    # Crack tint — a "tint" PNG auto-baked next to the opacity file where
    # cracks are mid-grey (~0.35) instead of pure black (0). Using the raw
    # opacity as glass_color_texture made cracks fully absorb transmitted
    # light and read as pitch black; the mid-grey version preserves some
    # transmission so cracks appear as a dark frosted smudge, not a hole.
    if opacity_texture and os.path.isfile(opacity_texture):
        tint_path = opacity_texture.replace("crack_opacity_",
                                            "crack_tint_")
        if not os.path.isfile(tint_path):
            try:
                import numpy as _np
                from PIL import Image as _Image
                arr = _np.asarray(
                    _Image.open(opacity_texture).convert("L"))
                # remap: crack pixels (0) -> ~180 (lighter grey, ~70%
                #        transmission) so cracks read as a perceptible
                #        tint without blocking most of the light,
                #        intact pixels (255) -> 255 (clear)
                floor = 180
                tint = _np.maximum(arr, floor).astype(_np.uint8)
                _Image.fromarray(tint, mode="L").save(tint_path)
            except Exception:
                tint_path = opacity_texture
        shader.CreateInput("glass_color_texture",
                           Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(tint_path))
    shader.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(depth)
    shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(thin_walled)

    if crack_hits > 0:
        # Use object-space projection so we control tiling from a known
        # baseline (one copy of the texture = one impact point).
        shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
        shader.CreateInput("world_or_object", Sdf.ValueTypeNames.Bool).Set(True)
        if crack_hits > 1:
            # Scale UP the texture coordinates so the texture repeats.
            # In OmniGlass, texture_scale multiplies the UV lookup:
            #   scale > 1 = UVs stretched = texture repeats more.
            # n copies per axis = ceil(sqrt(hits)).
            n = math.ceil(math.sqrt(crack_hits))
            shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(
                Gf.Vec2f(float(n), float(n))
            )

    if normal_texture and os.path.isfile(normal_texture):
        shader.CreateInput("normal_map_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(normal_texture)
        )
        shader.CreateInput("flip_tangent_v", Sdf.ValueTypeNames.Bool).Set(True)

    # Per-pixel roughness: the crack roughness PNG is white (1.0) where cracks
    # are and black (0.0) where glass is intact. With full influence, OmniGlass
    # scatters light only along the crack — the fracture reads as a visibly
    # frosted band, not a faint normal-map bend. Intact areas stay clear.
    if roughness_texture and os.path.isfile(roughness_texture):
        shader.CreateInput("roughness_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(roughness_texture)
        )
        # influence=1.0 → texture fully overrides the base scalar, so cracks
        # go fully frosted (scatter light, read as opaque white) while intact
        # glass falls back to the very low frosting_roughness (~mirror clear).
        # Maximum contrast between cracked and clear pixels.
        shader.CreateInput("roughness_texture_influence",
                           Sdf.ValueTypeNames.Float).Set(1.0)

    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    material.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    material.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    return material


def create_cracked_glass_materials(
    stage: Usd.Stage,
    texture_dir: str,
    root_path: str = "/World/Looks/CrackedGlass",
    variants: Sequence[str] = ("v1", "v2", "v3"),
) -> List[str]:
    """Create one OmniPBR material per texture variant.

    Looks for ``crack_normal_<variant>.png`` and ``crack_opacity_<variant>.png``
    inside ``texture_dir``. Missing textures are tolerated — the material
    still renders, just with constant opacity and no normal detail, so the
    pipeline keeps running while artists finish authoring.

    Returns:
        List of material prim paths (strings) that were created. Empty list
        if no variant textures were found.
    """
    UsdGeom.Scope.Define(stage, Sdf.Path(root_path).GetParentPath())
    UsdGeom.Scope.Define(stage, Sdf.Path(root_path))

    created: List[str] = []
    for variant in variants:
        normal_path = os.path.join(texture_dir, f"crack_normal_{variant}.png")
        opacity_path = os.path.join(texture_dir, f"crack_opacity_{variant}.png")
        roughness_path = os.path.join(texture_dir, f"crack_roughness_{variant}.png")
        material_path = f"{root_path}/{variant}"
        create_cracked_glass_material(
            stage,
            material_path=material_path,
            normal_texture=normal_path,
            opacity_texture=opacity_path if os.path.isfile(opacity_path) else None,
            roughness_texture=roughness_path if os.path.isfile(roughness_path) else None,
        )
        created.append(material_path)
    return created


# ---------------------------------------------------------------------------
# Binding + semantic labels
# ---------------------------------------------------------------------------

def bind_visual_material(
    stage: Usd.Stage,
    prim: Usd.Prim,
    material_path: str,
) -> None:
    """Bind a visual material to a prim using the default purpose binding.

    Mirrors ``bind_physics_material`` in ``building_physics.py`` but
    targets the default (visual) binding purpose instead of ``physics``.
    """
    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    binding_api.Bind(material, UsdShade.Tokens.strongerThanDescendants)


def apply_semantic_label(
    prim: Usd.Prim,
    label: str,
    semantic_type: str = "class",
) -> bool:
    """Tag a prim's underlying Mesh with a UsdSemantics label.

    Replicator's semantic segmentation annotator reads labels from Mesh
    prims. The Revit-exported building wraps each glass panel as an
    Xform with a nested Mesh behind an instance boundary, so we:
      1. Walk down (de-instancing as we go) to find the actual Mesh
         via the shared ``_find_mesh_descendant`` helper.
      2. Apply labels on both the Xform wrapper AND the Mesh — the
         wrapper for bbox annotators, the Mesh for segmentation.

    Returns True if a Mesh descendant was successfully labeled.
    """
    from isaacsim.core.utils.semantics import add_labels

    try:
        add_labels(prim, [label], instance_name=semantic_type, overwrite=True)
    except Exception:
        pass

    mesh = _find_mesh_descendant(prim, max_depth=4)
    if mesh is None:
        return False
    try:
        add_labels(mesh, [label], instance_name=semantic_type, overwrite=True)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------

def label_glass_prims(
    prims: Iterable[Usd.Prim],
    intact_label: str = "intact_glass",
) -> int:
    """Tag every glass prim with the intact label up-front.

    The Replicator randomization graph overrides selected prims each frame
    with ``cracked_glass``; anything not overridden keeps this baseline
    label. Returns the count of prims tagged.
    """
    count = 0
    for prim in prims:
        apply_semantic_label(prim, intact_label)
        count += 1
    return count
