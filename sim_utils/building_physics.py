"""
Physics utilities for applying collision, rigid bodies, materials, and joints
to a Revit-exported MSOE building USD.

Design principle:
  - The building itself is entirely static colliders (walls, floors, roofs, etc.)
  - Doors are the ONLY dynamic bodies (revolute joints for open/close)
  - Loose objects (chairs, robots, etc.) are added by simulation scripts, not here

Key constraint:
  - Triangle mesh collision (approx="none") only works on static objects
  - Dynamic doors must use convex hull approximation
"""

from enum import Enum, auto
from typing import List, Optional

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf, Sdf

try:
    from pxr import PhysxSchema
    _HAS_PHYSX_SCHEMA = True
except ImportError:
    PhysxSchema = None  # type: ignore
    _HAS_PHYSX_SCHEMA = False


# ---------------------------------------------------------------------------
# Prim Categories
# ---------------------------------------------------------------------------

class PrimCategory(Enum):
    STATIC_STRUCTURAL = auto()
    STATIC_GLAZING = auto()
    STATIC_STAIR = auto()
    DYNAMIC_DOOR = auto()
    SKIP = auto()


# Prefix -> category mapping (checked in order, first match wins)
_CATEGORY_RULES = [
    # Dynamic
    ("Door_Passage", PrimCategory.DYNAMIC_DOOR),
    ("Door_Curtain_Wall", PrimCategory.DYNAMIC_DOOR),
    ("Elevator_Door", PrimCategory.DYNAMIC_DOOR),

    # Skip (before structural to avoid false matches)
    ("Rectangular_Mullion", PrimCategory.SKIP),
    ("node_D_View", PrimCategory.SKIP),          # Camera views
    ("Strip_Lighting", PrimCategory.SKIP),        # MEP / lighting
    ("Lighting_Fixture", PrimCategory.SKIP),
    ("Troffer", PrimCategory.SKIP),
    ("Pendant", PrimCategory.SKIP),
    ("M_Round_Duct", PrimCategory.SKIP),
    ("Round_Duct", PrimCategory.SKIP),
    ("Rectangular_Duct", PrimCategory.SKIP),
    ("Duct_Fitting", PrimCategory.SKIP),
    ("Pipe_Type", PrimCategory.SKIP),
    ("Piping", PrimCategory.SKIP),
    ("Elbow", PrimCategory.SKIP),
    ("Tee_", PrimCategory.SKIP),
    ("Sprinkler", PrimCategory.SKIP),
    ("Diffuser", PrimCategory.SKIP),
    ("Air_Terminal", PrimCategory.SKIP),
    ("Mechanical_Equipment", PrimCategory.SKIP),
    ("Electrical_Equipment", PrimCategory.SKIP),
    ("Junction_Box", PrimCategory.SKIP),
    ("Receptacle", PrimCategory.SKIP),
    ("Switch_", PrimCategory.SKIP),
    ("Fire_Alarm", PrimCategory.SKIP),
    ("Smoke_Detector", PrimCategory.SKIP),
    ("cable_tray", PrimCategory.SKIP),
    ("Conduit", PrimCategory.SKIP),
    ("Panel_Board", PrimCategory.SKIP),
    ("Transformer", PrimCategory.SKIP),
    ("Plumbing", PrimCategory.SKIP),
    ("Toilet", PrimCategory.SKIP),
    ("Urinal", PrimCategory.SKIP),
    ("Sink", PrimCategory.SKIP),
    ("Drinking_Fountain", PrimCategory.SKIP),
    ("camera_D_View", PrimCategory.SKIP),         # Camera prims

    # Glazing
    ("System_Panel_Glazed", PrimCategory.STATIC_GLAZING),

    # Stairs
    ("Assembled_Stair", PrimCategory.STATIC_STAIR),
    ("Railing", PrimCategory.STATIC_STAIR),
    ("Top_Rail", PrimCategory.STATIC_STAIR),

    # Structural (catch-all for building elements)
    ("Basic_Wall", PrimCategory.STATIC_STRUCTURAL),
    ("Floor_", PrimCategory.STATIC_STRUCTURAL),
    ("Compound_Ceiling", PrimCategory.STATIC_STRUCTURAL),
    ("Basic_Roof", PrimCategory.STATIC_STRUCTURAL),
    ("Wall_Foundation", PrimCategory.STATIC_STRUCTURAL),
    ("Footing", PrimCategory.STATIC_STRUCTURAL),
    ("Column", PrimCategory.STATIC_STRUCTURAL),
    ("W_Shapes", PrimCategory.STATIC_STRUCTURAL),
    ("System_Panel_Solid", PrimCategory.STATIC_STRUCTURAL),
    ("HSS_Shapes", PrimCategory.STATIC_STRUCTURAL),
    ("Structural_Framing", PrimCategory.STATIC_STRUCTURAL),
    ("Concrete", PrimCategory.STATIC_STRUCTURAL),
    ("Slab", PrimCategory.STATIC_STRUCTURAL),
    ("Beam", PrimCategory.STATIC_STRUCTURAL),
    ("Brace", PrimCategory.STATIC_STRUCTURAL),
    ("Foundation", PrimCategory.STATIC_STRUCTURAL),
]


def decode_tn_name(name: str) -> str:
    """Strip the ``tn__`` prefix added by the Revit-to-USD exporter.

    Revit-exported prims in the engineering building carry a ``tn__`` prefix
    (e.g. ``tn__Basic_Wall_Exterior_123``).  This helper strips it so that
    downstream rules can match on the human-readable portion.  Names without
    the prefix pass through unchanged for backward compatibility with
    simpler USD files like TestBuilding.
    """
    if name.startswith("tn__"):
        return name[4:]
    return name


# Category-folder name -> PrimCategory mapping.
# Used when the building has a hierarchical structure with explicit category
# folders (Doors, Walls, Floors, …) under each level.  The folder name
# (after stripping ``tn__`` and any hash suffix) is matched with startswith.
_FOLDER_CATEGORY_MAP = {
    # Structural
    "Walls":              PrimCategory.STATIC_STRUCTURAL,
    "Wall_Sweeps":        PrimCategory.STATIC_STRUCTURAL,
    "Floors":             PrimCategory.STATIC_STRUCTURAL,
    "Ceilings":           PrimCategory.STATIC_STRUCTURAL,
    "Roofs":              PrimCategory.STATIC_STRUCTURAL,
    "Ramps":              PrimCategory.STATIC_STRUCTURAL,
    "Supports":           PrimCategory.STATIC_STRUCTURAL,
    "StructuralColumns":  PrimCategory.STATIC_STRUCTURAL,
    "StructuralFraming":  PrimCategory.STATIC_STRUCTURAL,
    "Columns":            PrimCategory.STATIC_STRUCTURAL,
    # Stairs / railings
    "Stairs":             PrimCategory.STATIC_STAIR,
    "Railings":           PrimCategory.STATIC_STAIR,
    "Runs":               PrimCategory.STATIC_STAIR,
    "Landings":           PrimCategory.STATIC_STAIR,
    "Handrails":          PrimCategory.STATIC_STAIR,
    "Top_Rails":          PrimCategory.STATIC_STAIR,
    # Doors (dynamic)
    "Doors":              PrimCategory.DYNAMIC_DOOR,
    # Glazing
    "Windows":            PrimCategory.STATIC_GLAZING,
    "CurtainPanels":      PrimCategory.STATIC_GLAZING,
    "Curtain_Panels":     PrimCategory.STATIC_GLAZING,
    "CurtainWallMullions": PrimCategory.STATIC_GLAZING,
    "Curtain_Wall_Mullions": PrimCategory.STATIC_GLAZING,
    # Everything not listed here falls through to SKIP.
}


def categorize_prim(name: str, category_hint: Optional[str] = None) -> PrimCategory:
    """Categorize a building prim by category-folder hint or name prefix.

    Args:
        name: The prim name (may carry a ``tn__`` prefix).
        category_hint: Optional category-folder name such as ``"Doors"``,
            ``"Walls"``, or ``"tn__StructuralFraming_xH"``.  When provided
            the folder-based ``_FOLDER_CATEGORY_MAP`` is consulted first
            (primary approach for the engineering building).  Falls back to
            prefix-based ``_CATEGORY_RULES`` when *category_hint* is ``None``
            (backward-compatible path for TestBuilding).
    """
    if category_hint is not None:
        decoded_hint = decode_tn_name(category_hint)
        for folder_name, category in _FOLDER_CATEGORY_MAP.items():
            if decoded_hint.startswith(folder_name):
                return category
        return PrimCategory.SKIP

    # Fallback: prefix-based matching (TestBuilding compatibility)
    decoded = decode_tn_name(name)
    for prefix, category in _CATEGORY_RULES:
        if decoded.startswith(prefix):
            return category
    return PrimCategory.SKIP


# ---------------------------------------------------------------------------
# Mesh Discovery
# ---------------------------------------------------------------------------

def _find_mesh_child(prim):
    """
    Find the first Mesh child of an Xform prim.

    Handles both typed Mesh prims and untyped (instanced) prims that
    resolve to a Mesh via USD composition (references/inherits).
    Returns the mesh prim or None.
    """
    for child in prim.GetChildren():
        # Check composed type — works for both typed and untyped/instanced prims
        if child.IsA(UsdGeom.Mesh):
            return child
    return None


def _find_mesh_descendant(prim, max_depth: int = 3) -> Optional[Usd.Prim]:
    """Find the first Mesh descendant, searching up to *max_depth* levels.

    For the engineering building most element prims are **instanceable**.
    Meshes are hidden behind the instance boundary and invisible to normal
    traversal.  This function calls ``SetInstanceable(False)`` when needed
    so that the Mesh children become accessible.  After de-instancing the
    Mesh is typically a direct child (depth 1), but a higher *max_depth*
    is kept for safety.

    Args:
        prim: Starting Xform prim to search under.
        max_depth: Maximum nesting levels to recurse into.

    Returns:
        The first ``UsdGeom.Mesh`` prim found, or ``None``.
    """
    # De-instance if the prim hides its children behind an instance boundary.
    if prim.IsInstanceable():
        prim.SetInstanceable(False)

    if max_depth <= 0:
        return None

    for child in prim.GetChildren():
        if child.IsA(UsdGeom.Mesh):
            return child
        result = _find_mesh_descendant(child, max_depth - 1)
        if result is not None:
            return result
    return None


def _find_all_mesh_descendants(prim, max_depth: int = 4):
    """Yield every Mesh descendant of *prim* up to *max_depth* levels.

    Unlike ``_find_mesh_descendant`` (first match), this walks the whole subtree.
    Needed for multi-mesh assemblies like non-hinged door families where a single
    element prim contains several separate meshes (glass panel, frame, hardware)
    that all need colliders for navigation raycasts to see them.
    """
    if prim.IsInstanceable():
        prim.SetInstanceable(False)
    if max_depth <= 0:
        return
    for child in prim.GetChildren():
        if child.IsA(UsdGeom.Mesh):
            yield child
        yield from _find_all_mesh_descendants(child, max_depth - 1)


# ---------------------------------------------------------------------------
# Physics Application
# ---------------------------------------------------------------------------

def _apply_mesh_collider(mesh_prim, approx="none"):
    """Apply CollisionAPI + MeshCollisionAPI (+ PhysX variants) to a Mesh prim."""
    UsdPhysics.CollisionAPI.Apply(mesh_prim)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
    mesh_collision.CreateApproximationAttr(approx)
    if _HAS_PHYSX_SCHEMA:
        PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)
        if approx == "none":
            PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(mesh_prim)


def apply_static_collider(stage, prim, approx="none", deep_search=False, all_meshes=False):
    """
    Apply static collision to a prim (no RigidBodyAPI = infinite mass, immovable).

    Applies CollisionAPI to the Xform and MeshCollisionAPI to its Mesh child.
    Triangle mesh (approx="none") gives exact collision for static geometry.

    Args:
        stage: USD stage
        prim: Xform prim containing a Mesh child
        approx: Collision approximation — "none" for triangle mesh (static only),
                 "convexHull", "convexDecomposition", "boundingCube", "boundingSphere"
        deep_search: If True, use _find_mesh_descendant() to search nested/instanced
                     hierarchies. Default False preserves backward compatibility.
        all_meshes: If True, apply collider to EVERY Mesh descendant (needed for
                    multi-mesh assemblies like door families with separate glass,
                    frame, and hardware meshes). Requires deep_search=True.
    Returns:
        True if collision was applied, False if no mesh child found
    """
    if all_meshes:
        found = False
        for mesh_prim in _find_all_mesh_descendants(prim):
            _apply_mesh_collider(mesh_prim, approx=approx)
            found = True
        return found

    if deep_search:
        mesh_prim = _find_mesh_descendant(prim)
    else:
        mesh_prim = _find_mesh_child(prim)
    if mesh_prim is None:
        return False

    _apply_mesh_collider(mesh_prim, approx=approx)
    return True


def apply_dynamic_rigid_body(stage, prim, mass=15.0, deep_search=False, all_meshes=False):
    """
    Apply dynamic rigid body physics to a prim (e.g. a door).

    Adds RigidBodyAPI + MassAPI on the Xform, CollisionAPI + MeshCollisionAPI
    on the Mesh child. Convex hull required for dynamic bodies.

    Args:
        stage: USD stage
        prim: Xform prim containing a Mesh child
        mass: Mass in kg
        deep_search: If True, use _find_mesh_descendant() to search nested/instanced
                     hierarchies. Default False preserves backward compatibility.
        all_meshes: If True, apply a convex-hull collider to EVERY mesh descendant
                    (frame, glass, hardware, …) so they all participate as shapes
                    under the rigid body. Without this, multi-mesh doors like
                    Curtain_Wall_Type_FG have only the first mesh (the frame)
                    blocking raycasts — the glass panel stays invisible.
    Returns:
        True if physics was applied, False if no mesh child found
    """
    if all_meshes:
        mesh_prims = list(_find_all_mesh_descendants(prim))
        if not mesh_prims:
            return False
        mesh_prim = mesh_prims[0]  # for material binding / primary shape reference
    else:
        if deep_search:
            found = _find_mesh_descendant(prim)
        else:
            found = _find_mesh_child(prim)
        if found is None:
            return False
        mesh_prims = [found]
        mesh_prim = found

    # RigidBodyAPI on the Xform (makes it a dynamic body)
    UsdPhysics.RigidBodyAPI.Apply(prim)

    # PhysxRigidBodyAPI for CCD and depenetration control (NVIDIA setRigidBody pattern)
    if _HAS_PHYSX_SCHEMA:
        physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        physx_rb.CreateMaxDepenetrationVelocityAttr(3.0)
        physx_rb.CreateSolverPositionIterationCountAttr(8)
        physx_rb.CreateSolverVelocityIterationCountAttr(4)
        physx_rb.CreateEnableCCDAttr(True)

    # CollisionAPI + MeshCollisionAPI on every mesh that should act as a shape.
    # convexHull is fast to cook; convexDecomposition is more accurate but
    # takes hours on T4 GPUs with large buildings — not worth the tradeoff.
    for m in mesh_prims:
        UsdPhysics.CollisionAPI.Apply(m)
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(m)
        mesh_collision.CreateApproximationAttr("convexHull")
        if _HAS_PHYSX_SCHEMA:
            PhysxSchema.PhysxCollisionAPI.Apply(m)

    # MassAPI on the Xform
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass)

    # Merge child meshes into a single collision shape for multi-mesh doors
    if _HAS_PHYSX_SCHEMA:
        mesh_children = [c for c in prim.GetChildren() if c.IsA(UsdGeom.Mesh)]
        if not mesh_children:
            for c in prim.GetChildren():
                mesh_children.extend(
                    gc for gc in c.GetChildren() if gc.IsA(UsdGeom.Mesh)
                )
        if len(mesh_children) > 1:
            PhysxSchema.PhysxMeshMergeCollisionAPI.Apply(prim)

    return True


def create_placeholder_door_mesh(stage, door_prim, panel_root="/World/DoorPanels"):
    """Create a visible rectangular mesh for doors with missing visual geometry.

    TEMPORARY — remove when Revit export includes proper door panel geometry.

    Creates a box mesh at /World/DoorPanels/<door_name> in world-space
    coordinates (feet). The mesh is defined directly in world space so it
    doesn't depend on the door prim's Revit-scale transforms.

    Note: The panel is static visual geometry only — it does NOT swing with
    the door's rigid body. For a proper demo of door physics, the door's
    collision shape (invisible convex hull) is what actually rotates.
    """
    xf_cache = UsdGeom.XformCache()
    world_points = _collect_mesh_world_points(door_prim, xf_cache)
    if len(world_points) < 4:
        return False

    xs = [p[0] for p in world_points]
    ys = [p[1] for p in world_points]
    zs = [p[2] for p in world_points]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)

    dx = x_max - x_min
    dy = y_max - y_min
    height = z_max - z_min

    if height < 0.1 or max(dx, dy) < 0.1:
        return False

    # Build a thin box in world space at the door's position
    cy = (y_min + y_max) / 2.0
    cx = (x_min + x_max) / 2.0
    thickness = 0.2  # 0.2 feet (~2.4 inches)

    if dx >= dy:
        # Door extends along X
        pts = [
            Gf.Vec3f(x_min, cy - thickness, z_min),
            Gf.Vec3f(x_max, cy - thickness, z_min),
            Gf.Vec3f(x_max, cy - thickness, z_max),
            Gf.Vec3f(x_min, cy - thickness, z_max),
            Gf.Vec3f(x_min, cy + thickness, z_min),
            Gf.Vec3f(x_max, cy + thickness, z_min),
            Gf.Vec3f(x_max, cy + thickness, z_max),
            Gf.Vec3f(x_min, cy + thickness, z_max),
        ]
    else:
        # Door extends along Y
        pts = [
            Gf.Vec3f(cx - thickness, y_min, z_min),
            Gf.Vec3f(cx - thickness, y_max, z_min),
            Gf.Vec3f(cx - thickness, y_max, z_max),
            Gf.Vec3f(cx - thickness, y_min, z_max),
            Gf.Vec3f(cx + thickness, y_min, z_min),
            Gf.Vec3f(cx + thickness, y_max, z_min),
            Gf.Vec3f(cx + thickness, y_max, z_max),
            Gf.Vec3f(cx + thickness, y_min, z_max),
        ]

    # Ensure root prim exists
    if not stage.GetPrimAtPath(panel_root):
        UsdGeom.Xform.Define(stage, panel_root)

    door_name = door_prim.GetName()
    mesh_path = f"{panel_root}/{door_name}"
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])
    mesh.CreateFaceVertexIndicesAttr([
        0, 1, 2, 3,  # front
        5, 4, 7, 6,  # back
        0, 3, 7, 4,  # left
        1, 5, 6, 2,  # right
        3, 2, 6, 7,  # top
        0, 4, 5, 1,  # bottom
    ])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr([(0.55, 0.35, 0.17)])

    return True


# ---------------------------------------------------------------------------
# Physics Materials
# ---------------------------------------------------------------------------

# Material name -> (static_friction, dynamic_friction, restitution)
MATERIAL_PROPERTIES = {
    "Concrete": (0.6, 0.5, 0.2),
    "Wood":     (0.5, 0.4, 0.3),
    "Metal":    (0.4, 0.3, 0.3),
    "Glass":    (0.3, 0.2, 0.5),
    "Tile":     (0.5, 0.4, 0.2),
}


def create_physics_materials(stage, root_path="/World/PhysicsMaterials"):
    """
    Create physics materials under the given root path.

    Returns:
        dict mapping material name to its USD path string
    """
    paths = {}
    for name, (sf, df, rest) in MATERIAL_PROPERTIES.items():
        mat_path = f"{root_path}/{name}"
        UsdShade.Material.Define(stage, mat_path)
        mat_prim = stage.GetPrimAtPath(mat_path)
        physics_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
        physics_mat.CreateStaticFrictionAttr(sf)
        physics_mat.CreateDynamicFrictionAttr(df)
        physics_mat.CreateRestitutionAttr(rest)
        paths[name] = mat_path

    return paths


def bind_physics_material(stage, prim, material_path):
    """Bind a physics material to a prim using MaterialBindingAPI."""
    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    binding_api.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")


# Category -> material name mapping
CATEGORY_MATERIALS = {
    PrimCategory.STATIC_STRUCTURAL: "Concrete",
    PrimCategory.STATIC_GLAZING:    "Glass",
    PrimCategory.STATIC_STAIR:      "Concrete",
    PrimCategory.DYNAMIC_DOOR:      "Wood",
}


# ---------------------------------------------------------------------------
# Revolute Joints (Door Hinges)
# ---------------------------------------------------------------------------

def create_revolute_joint(stage, joint_path, body0_path, body1_path,
                          axis="Z", lower_limit=-90.0, upper_limit=90.0,
                          local_pos0=None, local_pos1=None,
                          local_rot0=None, local_rot1=None,
                          break_force=3.4028235e+38,
                          break_torque=3.4028235e+38):
    """
    Create a revolute joint (hinge) between two bodies.

    For doors: body0 is the building frame (static world), body1 is the door.
    The joint axis is typically Z (vertical) for standard hinged doors.

    Args:
        stage: USD stage
        joint_path: Where to define the joint prim
        body0_path: Path to static body (or None/"" for world frame)
        body1_path: Path to dynamic body (the door)
        axis: Rotation axis — "X", "Y", or "Z"
        lower_limit: Lower angle limit in degrees
        upper_limit: Upper angle limit in degrees
        local_pos0: Local position offset on body0 (Gf.Vec3f)
        local_pos1: Local position offset on body1 (Gf.Vec3f)
        local_rot0: Local rotation on body0 (Gf.Quatf)
        local_rot1: Local rotation on body1 (Gf.Quatf)
        break_force: Force threshold to break joint (MAX_FLOAT = unbreakable)
        break_torque: Torque threshold to break joint (MAX_FLOAT = unbreakable)
    """
    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)

    # Set axis
    joint.CreateAxisAttr(axis)

    # Set angular limits
    joint.CreateLowerLimitAttr(lower_limit)
    joint.CreateUpperLimitAttr(upper_limit)

    # Connect bodies
    if body0_path:
        joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])

    # Set local frame offsets if provided
    if local_pos0 is not None:
        joint.CreateLocalPos0Attr(local_pos0)
    if local_pos1 is not None:
        joint.CreateLocalPos1Attr(local_pos1)
    if local_rot0 is not None:
        joint.CreateLocalRot0Attr(local_rot0)
    if local_rot1 is not None:
        joint.CreateLocalRot1Attr(local_rot1)

    # Prevent joint breakage under stress
    joint.CreateBreakForceAttr(break_force)
    joint.CreateBreakTorqueAttr(break_torque)

    # PhysX joint friction and soft angular limits
    if _HAS_PHYSX_SCHEMA:
        physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
        physx_joint.CreateJointFrictionAttr(0.1)

        limit_api = PhysxSchema.PhysxLimitAPI.Apply(joint.GetPrim(), "angular")
        limit_api.CreateStiffnessAttr(500.0)
        limit_api.CreateDampingAttr(100.0)
        limit_api.CreateRestitutionAttr(0.3)
        limit_api.CreateBounceThresholdAttr(0.5)

    return joint


# ---------------------------------------------------------------------------
# Door Mechanism Classification & Drive
# ---------------------------------------------------------------------------
#
# Classification System
# ---------------------
# Doors are classified by DoorMechanism based on substrings found in their
# USD prim path (which encodes the Revit family type name).
#
# Only HINGED_SINGLE and HINGED_DOUBLE get physics applied (rigid body +
# revolute joint + DriveAPI).  OVERHEAD, SLIDING, and UNKNOWN are skipped
# by the setup pipeline — they require joint types not yet implemented.
#
# Mechanism types and their Revit type patterns:
#   HINGED_SINGLE  — Type_F, Type_G, Type_N, Type_FG, Curtain_Wall (single leaf)
#   HINGED_DOUBLE  — Type_2F, Type_2G, Type_2FG, Type_2FU, Type_2F___Dual
#   OVERHEAD       — AV300_RAYNOR, Overhead_Rolling, Overhead_Sectional
#   SLIDING        — Sliding_Glass, Dormakaba
#   UNKNOWN        — No pattern matched (e.g. Elevator_Door)
#
# Double Door Geometry (Revit Export Limitation)
# -----------------------------------------------
# Revit exports double-leaf doors as a SINGLE unified prim with combined
# geometry (e.g. AL frame mesh + Glass mesh, both spanning the full width).
# The two leaves are NOT separate sub-prims and cannot be split at runtime.
#
# Consequence: double doors are treated as a single swinging unit with:
#   - One rigid body (30 kg total, vs 15 kg for single doors)
#   - One revolute joint at the min-edge of the bounding box
#   - Stronger drive parameters (stiffness=500, damping=250, max_force=5000)
#   - Reduced angular limits (±70° vs ±100° for single doors)
#
# This means both leaves swing together as one piece.  Independent leaf
# mechanics would require either a different Revit export format or
# runtime mesh splitting.
# ---------------------------------------------------------------------------


class DoorMechanism(Enum):
    """Mechanical type of a door, determines drive parameters."""
    HINGED_SINGLE = auto()
    HINGED_DOUBLE = auto()
    OVERHEAD = auto()
    SLIDING = auto()
    UNKNOWN = auto()


# Prim-path substrings -> mechanism mapping (checked in order, first match wins)
_MECHANISM_RULES = [
    # Overhead / garage
    ("AV300_RAYNOR", DoorMechanism.OVERHEAD),
    ("Overhead_Rolling", DoorMechanism.OVERHEAD),
    ("Overhead_Sectional", DoorMechanism.OVERHEAD),
    # Sliding
    ("Sliding_Glass", DoorMechanism.SLIDING),
    ("Dormakaba", DoorMechanism.SLIDING),
    # Double-leaf hinged (must check before single)
    ("Type_2F___Dual", DoorMechanism.HINGED_DOUBLE),  # RS - Type 2F - Dual Door
    ("Type_2F", DoorMechanism.HINGED_DOUBLE),
    ("Type_2G", DoorMechanism.HINGED_DOUBLE),
    ("2FG", DoorMechanism.HINGED_DOUBLE),
    ("2FU", DoorMechanism.HINGED_DOUBLE),
    ("2F_", DoorMechanism.HINGED_DOUBLE),
    # Sidelight door assemblies — door+fixed glass as one mesh, skip
    ("SIDELITE", DoorMechanism.UNKNOWN),
    ("Sidelite", DoorMechanism.UNKNOWN),
    # Clearwall / glass wall panels — not hinged doors
    ("Clearwall", DoorMechanism.UNKNOWN),
    ("clearwall", DoorMechanism.UNKNOWN),
    # Single-leaf hinged
    ("Type_F", DoorMechanism.HINGED_SINGLE),
    ("Type_G", DoorMechanism.HINGED_SINGLE),
    ("Type_N", DoorMechanism.HINGED_SINGLE),
    ("Type_FG", DoorMechanism.HINGED_SINGLE),
    ("Curtain_Wall", DoorMechanism.HINGED_SINGLE),
    # HMFrame (Hollow Metal) doors from combined Revit exports —
    # width determines single vs double (48"+ = double leaf)
    ("HMFrame96x", DoorMechanism.HINGED_DOUBLE),
    ("HMFrame90x", DoorMechanism.HINGED_DOUBLE),
    ("HMFrame84x", DoorMechanism.HINGED_DOUBLE),
    ("HMFrame72x", DoorMechanism.HINGED_DOUBLE),
    ("HMFrame68x", DoorMechanism.HINGED_DOUBLE),
    ("HMFrame66x", DoorMechanism.HINGED_DOUBLE),
    ("HMFrame60x", DoorMechanism.HINGED_DOUBLE),
    ("HMFrame48x", DoorMechanism.HINGED_DOUBLE),
    ("HMFrame42x", DoorMechanism.HINGED_SINGLE),
    ("HMFrame40x", DoorMechanism.HINGED_SINGLE),
    ("HMFrame36x", DoorMechanism.HINGED_SINGLE),
    ("HMFrame32x", DoorMechanism.HINGED_SINGLE),
    ("HMFrame30x", DoorMechanism.HINGED_SINGLE),
    ("HMFrame24x", DoorMechanism.HINGED_SINGLE),
    # ALFrame / CLFrame (Aluminum / Clear frame) — same width logic
    ("ALFrame36x", DoorMechanism.HINGED_SINGLE),
    ("CLFrame72x", DoorMechanism.HINGED_DOUBLE),
    # Bare size patterns (WxH) from combined Revit exports
    ("72x84", DoorMechanism.HINGED_DOUBLE),
    ("92HX84W", DoorMechanism.HINGED_DOUBLE),
    ("36x80", DoorMechanism.HINGED_SINGLE),
    ("36x84", DoorMechanism.HINGED_SINGLE),
]


def classify_door_mechanism(door_prim) -> DoorMechanism:
    """Classify a door prim's mechanical type by inspecting its path.

    Looks for known substrings in the prim path (which includes the
    type-group folder name from the Revit export).
    """
    path_str = str(door_prim.GetPath())
    for substring, mechanism in _MECHANISM_RULES:
        if substring in path_str:
            return mechanism
    return DoorMechanism.UNKNOWN


# Default drive parameters per mechanism: (stiffness, damping, max_force)
DOOR_DRIVE_PARAMS = {
    DoorMechanism.HINGED_SINGLE: (300.0, 150.0, 3000.0),
    DoorMechanism.HINGED_DOUBLE: (500.0, 250.0, 5000.0),  # stronger for heavier 2-leaf door
}


def setup_door_drive(joint, stiffness=300.0, damping=150.0,
                     max_force=3000.0, target_position=0.0):
    """Apply UsdPhysics.DriveAPI (angular position drive) to a revolute joint.

    Args:
        joint: UsdPhysics.RevoluteJoint or its prim
        stiffness: Spring constant toward target angle
        damping: Damping coefficient to prevent oscillation
        max_force: Maximum torque the drive can apply
        target_position: Initial target angle in degrees (0 = closed)

    Returns:
        The UsdPhysics.DriveAPI instance
    """
    joint_prim = joint.GetPrim() if hasattr(joint, "GetPrim") else joint
    drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
    drive.CreateStiffnessAttr(stiffness)
    drive.CreateDampingAttr(damping)
    drive.CreateMaxForceAttr(max_force)
    drive.CreateTargetPositionAttr(target_position)
    return drive


def set_drive_target(drive, angle_deg):
    """Update the drive's target angle (degrees) at runtime."""
    drive.GetTargetPositionAttr().Set(angle_deg)


def _collect_mesh_world_points(prim, xf_cache, max_depth=4):
    """Gather all mesh vertex positions in world space under a prim."""
    points = []
    if prim.IsInstanceable():
        prim.SetInstanceable(False)
    if prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        if pts:
            local_to_world = xf_cache.GetLocalToWorldTransform(prim)
            for p in pts:
                wp = local_to_world.Transform(Gf.Vec3d(p[0], p[1], p[2]))
                points.append(wp)
    if max_depth > 0:
        for child in prim.GetChildren():
            points.extend(_collect_mesh_world_points(child, xf_cache, max_depth - 1))
    return points


def estimate_door_hinge_offset(stage, door_prim):
    """
    Estimate hinge position using mesh vertex analysis.

    Instead of using the prim bounding box (which includes frame/mullion
    geometry for curtain wall doors), this analyzes the actual mesh vertices
    to find the densest vertical edge — that's where the hinge goes.

    For each of the 4 vertical edges of the door's XY footprint (min-X,
    max-X, min-Y, max-Y), counts how many vertices are near that edge.
    The edge with the most vertices is the frame/hinge side.

    Returns (local_pos0, local_pos1, local_rot0, local_rot1) where positions
    are Gf.Vec3f and rotations are Gf.Quatf, suitable for create_revolute_joint.
    Returns (None, None, None, None) if geometry can't be analyzed.
    """
    xf_cache = UsdGeom.XformCache()

    # Collect all mesh vertices in world space
    world_points = _collect_mesh_world_points(door_prim, xf_cache)
    if len(world_points) < 4:
        return None, None, None, None

    # Compute tight bounds from actual vertices
    xs = [p[0] for p in world_points]
    ys = [p[1] for p in world_points]
    zs = [p[2] for p in world_points]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)

    dx = x_max - x_min
    dy = y_max - y_min

    if dx < 0.01 and dy < 0.01:
        return None, None, None, None

    # Determine door orientation from vertex spread
    if dx >= dy:
        # Door extends along X — hinge is at min-X or max-X
        # Count vertices near each X edge to find the frame/hinge side.
        # The hinge side typically has more vertices (frame geometry).
        edge_tolerance = dx * 0.15  # 15% of door width
        near_min_x = sum(1 for x in xs if x - x_min < edge_tolerance)
        near_max_x = sum(1 for x in xs if x_max - x < edge_tolerance)
        # Pick the edge with MORE vertices (that's the frame/hinge side)
        if near_min_x >= near_max_x:
            hinge_x = x_min
        else:
            hinge_x = x_max
        hinge_y = (y_min + y_max) / 2.0
    else:
        # Door extends along Y — hinge is at min-Y or max-Y
        edge_tolerance = dy * 0.15
        near_min_y = sum(1 for y in ys if y - y_min < edge_tolerance)
        near_max_y = sum(1 for y in ys if y_max - y < edge_tolerance)
        if near_min_y >= near_max_y:
            hinge_y = y_min
        else:
            hinge_y = y_max
        hinge_x = (x_min + x_max) / 2.0

    hinge_z = (z_min + z_max) / 2.0

    # local_pos0: hinge in world frame
    hinge_world = Gf.Vec3d(hinge_x, hinge_y, hinge_z)
    local_pos0 = Gf.Vec3f(float(hinge_world[0]),
                           float(hinge_world[1]),
                           float(hinge_world[2]))

    # local_pos1: hinge in door's local frame
    # Use the FULL inverse transform (including Revit's ~100x scale) so that
    # local_pos1 is in the door prim's actual local coordinate system.
    # RemoveScaleShear() would strip the scale, producing values ~100x too large.
    world_xform = xf_cache.GetLocalToWorldTransform(door_prim)
    inv_xform = world_xform.GetInverse()
    hinge_local = inv_xform.Transform(hinge_world)
    local_pos1 = Gf.Vec3f(float(hinge_local[0]),
                           float(hinge_local[1]),
                           float(hinge_local[2]))

    # Rotation quaternions for joint frames
    local_rot0 = Gf.Quatf(1.0)  # body0 is world frame — identity
    local_rot1 = Gf.Quatf(1.0)  # Revit doors have no rotation in parent chain

    return local_pos0, local_pos1, local_rot0, local_rot1


# ---------------------------------------------------------------------------
# Collision Filtering for Doors
# ---------------------------------------------------------------------------

def _build_collision_index(stage, exclude_prefix=None):
    """Build a spatial index of all static collision prims.

    Returns list of (prim, bbox_min, bbox_max) tuples.
    Only includes prims that have CollisionAPI but NOT RigidBodyAPI
    (i.e., static colliders only).
    """
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    )
    index = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        if exclude_prefix and str(prim.GetPath()).startswith(exclude_prefix):
            continue
        bbox = bbox_cache.ComputeWorldBound(prim)
        rng = bbox.ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        index.append((prim, rng.GetMin(), rng.GetMax()))
    return index


def _boxes_overlap(a_min, a_max, b_min, b_max):
    """Check if two 3D AABBs overlap."""
    for i in range(3):
        if a_min[i] >= b_max[i] or b_min[i] >= a_max[i]:
            return False
    return True


def add_door_collision_filters(stage, door_prims, collision_index=None,
                               margin=0.0):
    """Disable collision between each door and nearby static geometry.

    Curtain wall doors from Revit sit flush inside wall/mullion assemblies.
    Without filtering, the door's convex hull collision immediately jams
    against the surrounding static colliders, preventing any rotation.

    This function finds all static colliders whose bounding box overlaps
    each door's (optionally inflated) bounding box and applies
    ``PhysxFilteredPairsAPI`` to prevent collision between them.

    Args:
        stage: USD stage.
        door_prims: List of door UsdPrim objects (must have RigidBodyAPI).
        collision_index: Pre-built index from ``_build_collision_index()``.
            If None, one is built automatically.
        margin: Distance (in stage units, typically meters) to inflate each
            door's bounding box before the overlap test.  A positive margin
            catches nearby geometry that the door sweeps into during the
            first degrees of rotation — essential for curtain wall doors
            where mullions sit just outside the resting AABB.

    Returns:
        Total number of filtered pairs added.
    """
    if collision_index is None:
        collision_index = _build_collision_index(stage)

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    )

    total_filtered = 0

    for door_prim in door_prims:
        door_path = str(door_prim.GetPath())

        # Get door's world bounding box
        door_bbox = bbox_cache.ComputeWorldBound(door_prim)
        door_range = door_bbox.ComputeAlignedRange()
        if door_range.IsEmpty():
            continue
        d_min = door_range.GetMin()
        d_max = door_range.GetMax()

        # Inflate bounding box by margin to catch swept-arc geometry
        if margin > 0:
            d_min = Gf.Vec3d(d_min[0] - margin, d_min[1] - margin,
                             d_min[2] - margin)
            d_max = Gf.Vec3d(d_max[0] + margin, d_max[1] + margin,
                             d_max[2] + margin)

        # Find overlapping static colliders
        overlapping = []
        for col_prim, c_min, c_max in collision_index:
            if str(col_prim.GetPath()).startswith(door_path):
                continue  # skip self/children
            if _boxes_overlap(d_min, d_max, c_min, c_max):
                overlapping.append(col_prim)

        if not overlapping:
            continue

        # Apply FilteredPairsAPI on each collision child of the door.
        # The filter must be on the prim with CollisionAPI, and targets
        # must also be collision prims, for PhysX to respect it.
        door_col_prims = []
        for child in door_prim.GetChildren():
            if child.HasAPI(UsdPhysics.CollisionAPI):
                door_col_prims.append(child)

        # Also apply on the door prim itself (covers both levels)
        filter_prims = door_col_prims + [door_prim]

        targets = [col.GetPath() for col in overlapping]
        for fp in filter_prims:
            api = UsdPhysics.FilteredPairsAPI.Apply(fp)
            rel = api.GetFilteredPairsRel()
            rel.SetTargets(targets)
        total_filtered += len(targets)

    return total_filtered


def setup_collision_groups(stage, door_prims, group_root="/World/CollisionGroups"):
    """Create collision groups for doors and static building elements.

    Replaces pairwise FilteredPairsAPI with group-based filtering:
    - "doors" group: all door collision meshes
    - "static_building" group: all static colliders

    Doors won't collide with walls/floors/etc. but WILL collide with
    other dynamic objects (robots, dropped items).

    Args:
        stage: USD stage
        door_prims: list of door Xform prims (with RigidBodyAPI)
        group_root: path prefix for collision group definitions

    Returns:
        (doors_group_path, static_group_path) tuple
    """
    doors_path = f"{group_root}/doors"
    static_path = f"{group_root}/static_building"

    doors_group = UsdPhysics.CollisionGroup.Define(stage, doors_path)
    static_group = UsdPhysics.CollisionGroup.Define(stage, static_path)

    # Doors group filters against static_building group
    doors_group.CreateFilteredGroupsRel().SetTargets([Sdf.Path(static_path)])

    # Add door collision meshes to doors group
    for door_prim in door_prims:
        for child in door_prim.GetChildren():
            if child.HasAPI(UsdPhysics.CollisionAPI):
                doors_group.GetCollidersCollectionAPI().GetIncludesRel().AddTarget(
                    child.GetPath()
                )
        if door_prim.HasAPI(UsdPhysics.CollisionAPI):
            doors_group.GetCollidersCollectionAPI().GetIncludesRel().AddTarget(
                door_prim.GetPath()
            )

    # Add static colliders to static group (auto-discover)
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        # Skip collision prims that are children of rigid bodies (door meshes)
        parent = prim.GetParent()
        is_rb_child = False
        while parent and parent != stage.GetPseudoRoot():
            if parent.HasAPI(UsdPhysics.RigidBodyAPI):
                is_rb_child = True
                break
            parent = parent.GetParent()
        if is_rb_child:
            continue
        static_group.GetCollidersCollectionAPI().GetIncludesRel().AddTarget(
            prim.GetPath()
        )

    return doors_path, static_path


# ---------------------------------------------------------------------------
# Physics Scene Setup
# ---------------------------------------------------------------------------

def setup_physics_scene(stage, gravity=-9.81, time_steps_per_second=60,
                        gpu_dynamics=False, scene_path="/World/PhysicsScene",
                        solver_type="TGS", broadphase_type="MBP",
                        enable_ccd=True, min_position_iterations=4,
                        max_position_iterations=16):
    """
    Create and configure a PhysicsScene with gravity and solver settings.

    Args:
        stage: USD stage
        gravity: Gravitational acceleration (negative = downward)
        time_steps_per_second: Physics solver frequency
        gpu_dynamics: Enable GPU-accelerated dynamics (False recommended for <100 bodies)
        scene_path: Where to define the physics scene prim
        solver_type: "TGS" (better convergence) or "PGS"
        broadphase_type: "MBP" (Multi-Box Pruning) or "GPU"
        enable_ccd: Continuous collision detection (prevents tunneling)
        min_position_iterations: Minimum solver position iterations
        max_position_iterations: Maximum solver position iterations
    """
    scene = UsdPhysics.Scene.Define(stage, scene_path)
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(abs(gravity))

    # Configure PhysX-specific scene properties
    if _HAS_PHYSX_SCHEMA:
        physx_prim = stage.GetPrimAtPath(scene_path)
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physx_prim)
        physx_scene.CreateTimeStepsPerSecondAttr(time_steps_per_second)
        physx_scene.CreateSolverTypeAttr(solver_type)
        physx_scene.CreateBroadphaseTypeAttr(broadphase_type)
        physx_scene.CreateEnableCCDAttr(enable_ccd)
        physx_scene.CreateMinPositionIterationCountAttr(min_position_iterations)
        physx_scene.CreateMaxPositionIterationCountAttr(max_position_iterations)
        physx_scene.CreateEnableStabilizationAttr(True)
        physx_scene.CreateEnableGPUDynamicsAttr(gpu_dynamics)

    return scene


# ---------------------------------------------------------------------------
# Building Traversal
# ---------------------------------------------------------------------------

def _find_wrapper(prim):
    """Return the Xform child of *prim* that has the most children.

    In the engineering building the model root sits inside one or two
    wrapper Xforms.  The wrapper is the Xform child with the most
    sub-prims (level prims, category folders, etc.).  Scope prims are
    skipped because they hold materials, not geometry.
    """
    best = None
    best_count = -1
    for child in prim.GetChildren():
        if child.GetTypeName() == "Scope":
            continue
        if not child.IsA(UsdGeom.Xform):
            continue
        count = len(child.GetChildren())
        if count > best_count:
            best = child
            best_count = count
    return best


def _has_level_children(prim):
    """Return True if *prim* has any children with ``LEVEL`` in their decoded name."""
    for child in prim.GetChildren():
        if "LEVEL" in decode_tn_name(child.GetName()).upper():
            return True
    return False


def _iter_leaf_elements(prim, max_depth: int = 6):
    """Yield leaf-element Xform prims under *prim*.

    Recurses through type sub-folders and extra wrappers until it finds
    actual building elements.  An element is detected by either:

    * ``IsInstanceable()`` — meshes hidden behind an instance boundary
      (engineering building with tn__ prefix export).
    * Having a direct Mesh child — non-instanced elements where the Mesh
      is a regular child prim (MSOEBuildingTake2 / newer exports).

    Non-leaf Xforms (folders/wrappers with only Xform children) are
    recursed into up to *max_depth* levels.
    """
    if max_depth <= 0:
        return
    for child in prim.GetChildren():
        type_name = child.GetTypeName()
        # Skip only non-container types. Untyped prims (empty typeName) act as
        # Xform containers in Revit exports — some door families wrap instances
        # in an untyped family-template prim (e.g. `_4000_x_3000_4_Panel`,
        # `NEW___HM_Frame___36_x96____18__Sidelite`). If we required
        # IsA(UsdGeom.Xform) those whole subtrees would be skipped, leaving
        # glass/frame meshes without any physics.
        if type_name in ("Scope", "Mesh", "Camera", "Light",
                         "DistantLight", "SphereLight", "DomeLight",
                         "RectLight", "DiskLight", "CylinderLight",
                         "Material", "Shader"):
            continue
        if child.IsInstanceable():
            yield child
        elif _find_mesh_child(child) is not None:
            yield child
        else:
            yield from _iter_leaf_elements(child, max_depth - 1)


def iter_building_elements(stage, building_prim, level_filter=None):
    """Yield ``(level_name, category_name, element_prim)`` tuples.

    Automatically detects the building structure:

    * **Flat** (TestBuilding) — children of *building_prim* are element
      Xforms directly.  Yields ``("default", None, prim)``.
    * **Category-based** (MSOEBuildingTake2) — single wrapper with a
      ``Geometry`` child containing category folders (``Walls``, ``Doors``,
      …).  No floor-level grouping.  Yields ``("default", cat_name, prim)``.
    * **Hierarchical** (older engineering building export) — double-wrapped
      with level prims containing category folders.

    Args:
        stage: The USD stage.
        building_prim: Root prim of the building (e.g. ``/World/Building``).
        level_filter: Optional list of substrings.  Only levels whose
            decoded name contains one of these substrings (case-insensitive)
            are included.  ``None`` means all levels.

    Yields:
        ``(level_name, category_name, element_prim)`` where *level_name*
        is the decoded floor-level name, *category_name* is the decoded
        category-folder name (or ``None`` for flat buildings), and
        *element_prim* is the Xform to apply physics to.
    """
    outer = _find_wrapper(building_prim)

    # --- detect category-based structure (Geometry folder) ---
    geometry_prim = None
    if outer is not None:
        for child in outer.GetChildren():
            if child.GetName() == "Geometry" and child.IsA(UsdGeom.Xform):
                geometry_prim = child
                break

    if geometry_prim is not None:
        # -- category-based mode (MSOEBuildingTake2) --
        for cat_prim in geometry_prim.GetChildren():
            if cat_prim.GetTypeName() == "Scope":
                continue
            if not cat_prim.IsA(UsdGeom.Xform):
                continue
            category_name = cat_prim.GetName()
            for element in _iter_leaf_elements(cat_prim):
                yield ("default", category_name, element)
        return

    # --- detect hierarchical structure (level-based) ---
    inner = None
    if outer is not None:
        if _has_level_children(outer):
            inner = outer
        else:
            candidate = _find_wrapper(outer)
            if candidate is not None and _has_level_children(candidate):
                inner = candidate

    if inner is not None:
        # -- hierarchical mode (older engineering building export) --
        for level_prim in inner.GetChildren():
            level_name = decode_tn_name(level_prim.GetName())

            if "LEVEL" not in level_name.upper():
                continue

            if level_filter is not None:
                if not any(f.upper() in level_name.upper() for f in level_filter):
                    continue

            # Each child of the level prim is a category folder.
            for cat_prim in level_prim.GetChildren():
                category_name = decode_tn_name(cat_prim.GetName())
                for element in _iter_leaf_elements(cat_prim):
                    yield (level_name, category_name, element)
    else:
        # -- flat mode (TestBuilding) --
        for child in building_prim.GetChildren():
            if child.IsA(UsdGeom.Xform) and child.GetTypeName() != "Scope":
                yield ("default", None, child)


# ---------------------------------------------------------------------------
# Floor Discovery
# ---------------------------------------------------------------------------

def discover_floor_levels(stage, building_path="/World/Building"):
    """Discover distinct floor surfaces by clustering floor element bboxes.

    Iterates floor elements via ``iter_building_elements``, groups them by
    Z-height (1.5-unit tolerance), and returns an area-weighted centroid per
    cluster.

    Returns:
        list[dict]: Sorted by ``z`` ascending.  Each dict has keys
        ``name``, ``z``, ``center_x``, ``center_y``, ``count``.
    """
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    )
    building_prim = stage.GetPrimAtPath(building_path)

    floor_data = []
    for _level, category_name, element in iter_building_elements(stage, building_prim):
        if category_name != "Floors":
            continue
        bbox = bbox_cache.ComputeWorldBound(element)
        box_range = bbox.ComputeAlignedRange()
        bb_min = box_range.GetMin()
        bb_max = box_range.GetMax()
        if bb_min == bb_max:
            continue
        dx = bb_max[0] - bb_min[0]
        dy = bb_max[1] - bb_min[1]
        if dx < 0.5 and dy < 0.5:
            continue
        floor_data.append({
            "z_top": bb_max[2],
            "x_center": (bb_min[0] + bb_max[0]) / 2.0,
            "y_center": (bb_min[1] + bb_max[1]) / 2.0,
            "area": dx * dy,
        })

    if not floor_data:
        return []

    # Cluster by Z height (1.5-unit tolerance)
    floor_data.sort(key=lambda f: f["z_top"])
    clusters = []
    current = [floor_data[0]]
    for fd in floor_data[1:]:
        if fd["z_top"] - current[0]["z_top"] > 1.5:
            clusters.append(current)
            current = [fd]
        else:
            current.append(fd)
    clusters.append(current)

    # Aggregate each cluster with area-weighted center
    levels = []
    for i, cluster in enumerate(clusters):
        total_area = sum(f["area"] for f in cluster)
        if total_area < 0.01:
            continue
        avg_x = sum(f["x_center"] * f["area"] for f in cluster) / total_area
        avg_y = sum(f["y_center"] * f["area"] for f in cluster) / total_area
        avg_z = sum(f["z_top"] * f["area"] for f in cluster) / total_area
        levels.append({
            "name": f"floor_{i + 1}",
            "z": avg_z,
            "center_x": avg_x,
            "center_y": avg_y,
            "count": len(cluster),
        })

    return levels


# ---------------------------------------------------------------------------
# Asset Path Portability
# ---------------------------------------------------------------------------

def fix_absolute_asset_paths(stage):
    """Convert absolute asset paths to relative ones for portability.

    Revit-to-USD exporters often bake in absolute paths to MDL material files
    (e.g. /home/user/project/assets/.../Materials/X.mdl).  These break when
    another user opens the file.

    This function rewrites every SdfAssetPath-typed attribute that contains
    an absolute ``/home/`` path, converting it to a path relative to the USD
    file's directory (e.g. ``./Materials/X.mdl``).

    Returns the number of attributes fixed.
    """
    REPO_MARKER = "assets/environments/msoe_building/"

    fixed = 0
    for prim in stage.TraverseAll():
        for attr in prim.GetAttributes():
            try:
                if "SdfAssetPath" not in attr.GetTypeName().cppTypeName:
                    continue
                val = attr.Get()
                if val is None:
                    continue
                path_str = val.path if hasattr(val, "path") else str(val)
                if not path_str or "/home/" not in path_str:
                    continue
                idx = path_str.find(REPO_MARKER)
                if idx < 0:
                    continue
                relative = "./" + path_str[idx + len(REPO_MARKER):]
                attr.Set(Sdf.AssetPath(relative))
                fixed += 1
            except Exception:
                pass
    return fixed


# ---------------------------------------------------------------------------
# Native PhysX Runtime Metrics Helpers
# ---------------------------------------------------------------------------
# These functions query the PhysX simulation engine for data that is only
# available AFTER world.reset() has been called and at least one world.step()
# has executed.  All are wrapped in try/except so they degrade gracefully
# when the underlying APIs are unavailable.
# ---------------------------------------------------------------------------

def encode_prim_path(prim_path_str):
    """Encode a USD prim path string to the uint64 used by PhysX runtime APIs.

    Returns the encoded int, or None if encoding is unavailable.
    """
    try:
        from pxr import PhysicsSchemaTools
        return PhysicsSchemaTools.encodeSdfPath(Sdf.Path(prim_path_str))
    except (ImportError, AttributeError):
        return None


def get_physics_scene_stats(stage, scene_path="/World/PhysicsScene"):
    """Snapshot PhysicsSceneStats into a plain dict.

    Returns a dict with selected PhysX scene statistics, or an empty dict
    if the API is unavailable.
    """
    try:
        import omni.usd
        from omni.physx import get_physx_statistics_interface
        from omni.physx.bindings._physx import PhysicsSceneStats
        from pxr import PhysicsSchemaTools

        stage_id = omni.usd.get_context().get_stage_id()
        encoded_path = PhysicsSchemaTools.encodeSdfPath(Sdf.Path(scene_path))

        stats = PhysicsSceneStats()
        get_physx_statistics_interface().get_physx_scene_statistics(
            stage_id, encoded_path, stats
        )

        return {
            # Body counts
            "nb_static_rigids": stats.nb_static_rigids,
            "nb_dynamic_rigids": stats.nb_dynamic_rigids,
            "nb_active_dynamic_rigids": stats.nb_active_dynamic_rigids,
            "nb_kinematic_rigids": stats.nb_kinematic_rigids,
            "nb_active_kinematic_rigids": stats.nb_active_kinematic_rigids,
            "nb_articulations": stats.nb_articulations,
            # Constraints
            "nb_active_constraints": stats.nb_active_constraints,
            "nb_axis_solver_constraints": stats.nb_axis_solver_constaints,
            # Shape counts
            "nb_box_shapes": stats.nb_box_shapes,
            "nb_sphere_shapes": stats.nb_sphere_shapes,
            "nb_capsule_shapes": stats.nb_capsule_shapes,
            "nb_convex_shapes": stats.nb_convex_shapes,
            "nb_trimesh_shapes": stats.nb_trimesh_shapes,
            "nb_plane_shapes": stats.nb_plane_shapes,
            # Broadphase / narrowphase
            "nb_new_pairs": stats.nb_new_pairs,
            "nb_lost_pairs": stats.nb_lost_pairs,
            "nb_new_touches": stats.nb_new_touches,
            "nb_lost_touches": stats.nb_lost_touches,
            "nb_discrete_contact_pairs_total": stats.nb_discrete_contact_pairs_total,
            "nb_discrete_contact_pairs_with_contacts": stats.nb_discrete_contact_pairs_with_contacts,
            # Solver
            "nb_partitions": stats.nb_partitions,
            # Memory
            "compressed_contact_size": stats.compressed_contact_size,
            "required_contact_constraint_memory": stats.required_contact_constraint_memory,
            "peak_constraint_memory": stats.peak_constraint_memory,
            # GPU memory
            "gpu_mem_heap": stats.gpu_mem_heap,
            "gpu_mem_heap_broadphase": stats.gpu_mem_heap_broadphase,
            "gpu_mem_heap_narrowphase": stats.gpu_mem_heap_narrowphase,
            "gpu_mem_heap_solver": stats.gpu_mem_heap_solver,
            "gpu_mem_heap_articulation": stats.gpu_mem_heap_articulation,
            "gpu_mem_heap_simulation": stats.gpu_mem_heap_simulation,
            "gpu_mem_heap_other": stats.gpu_mem_heap_other,
        }
    except Exception as e:
        print(f"[building_physics] Warning: could not collect scene stats: {e}")
        return {}


def get_solver_residuals(stage, scene_path="/World/PhysicsScene"):
    """Read solver residual values from the physics scene prim.

    PhysxResidualReportingAPI must be applied and the carb setting
    /physics/updateResidualsToUsd must be True BEFORE world.reset().
    Call this AFTER at least one world.step().

    Returns dict with rms/max position/velocity residuals, or empty dict.
    """
    try:
        if not _HAS_PHYSX_SCHEMA:
            return {}

        scene_prim = stage.GetPrimAtPath(scene_path)
        if not scene_prim.IsValid():
            return {}

        residual_api = PhysxSchema.PhysxResidualReportingAPI(scene_prim)

        rms_pos = residual_api.GetPhysxResidualReportingRmsResidualPositionIterationAttr().Get()
        max_pos = residual_api.GetPhysxResidualReportingMaxResidualPositionIterationAttr().Get()
        rms_vel = residual_api.GetPhysxResidualReportingRmsResidualVelocityIterationAttr().Get()
        max_vel = residual_api.GetPhysxResidualReportingMaxResidualVelocityIterationAttr().Get()

        return {
            "rms_position": float(rms_pos) if rms_pos is not None else None,
            "max_position": float(max_pos) if max_pos is not None else None,
            "rms_velocity": float(rms_vel) if rms_vel is not None else None,
            "max_velocity": float(max_vel) if max_vel is not None else None,
        }
    except Exception as e:
        print(f"[building_physics] Warning: could not read solver residuals: {e}")
        return {}


def get_compute_cost(stage, scene_path="/World/PhysicsScene"):
    """Extract GPU memory breakdown and physics step timing.

    GPU memory comes from PhysicsSceneStats (bytes → MB).
    Physics step timing comes from PhysicsProfileStats if available.

    Returns dict with cost metrics, or empty dict.
    """
    result = {}
    try:
        scene_stats = get_physics_scene_stats(stage, scene_path)
        if scene_stats:
            to_mb = 1.0 / (1024 * 1024)
            result["gpu_mem_total_mb"] = round(scene_stats.get("gpu_mem_heap", 0) * to_mb, 2)
            result["gpu_mem_solver_mb"] = round(scene_stats.get("gpu_mem_heap_solver", 0) * to_mb, 2)
            result["gpu_mem_broadphase_mb"] = round(scene_stats.get("gpu_mem_heap_broadphase", 0) * to_mb, 2)
            result["gpu_mem_narrowphase_mb"] = round(scene_stats.get("gpu_mem_heap_narrowphase", 0) * to_mb, 2)
            result["gpu_mem_articulation_mb"] = round(scene_stats.get("gpu_mem_heap_articulation", 0) * to_mb, 2)
            result["peak_constraint_memory_mb"] = round(scene_stats.get("peak_constraint_memory", 0) * to_mb, 2)
    except Exception as e:
        print(f"[building_physics] Warning: could not compute GPU cost: {e}")

    # Physics step timing from profiler data
    try:
        import carb
        carb.settings.get_settings().set("/physics/exposeProfilerData", True)
        from omni.physx.bindings._physx import PhysicsProfileStats
        # Profile stats may need a frame to populate after enabling
    except Exception:
        pass

    return result


def check_body_sleeping(stage, prim_path_str):
    """Check if a rigid body is sleeping (has settled).

    Returns True if sleeping, False if awake, None if query failed.
    """
    try:
        import omni.usd
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools

        stage_id = omni.usd.get_context().get_stage_id()
        encoded = PhysicsSchemaTools.encodeSdfPath(Sdf.Path(prim_path_str))
        return get_physx_simulation_interface().is_sleeping(stage_id, encoded)
    except Exception:
        return None


def raycast_down(origin, max_distance=100.0):
    """Cast a ray straight down from origin and return hit info.

    Args:
        origin: (x, y, z) tuple or list in stage units
        max_distance: maximum ray distance

    Returns:
        (hit: bool, distance: float or None, position: tuple or None)
    """
    try:
        import carb
        from omni.physx import get_physx_scene_query_interface

        sq = get_physx_scene_query_interface()
        hit = sq.raycast_closest(
            carb.Float3(float(origin[0]), float(origin[1]), float(origin[2])),
            carb.Float3(0.0, 0.0, -1.0),
            float(max_distance),
        )
        if hit and "distance" in hit:
            pos = hit.get("position", (0, 0, 0))
            return (True, float(hit["distance"]), (float(pos[0]), float(pos[1]), float(pos[2])))
        return (False, None, None)
    except Exception:
        return (False, None, None)


def enable_residual_reporting(stage, scene_path="/World/PhysicsScene"):
    """Apply PhysxResidualReportingAPI to the physics scene prim.

    MUST be called BEFORE world.reset().
    """
    try:
        import carb

        if not _HAS_PHYSX_SCHEMA:
            return False

        scene_prim = stage.GetPrimAtPath(scene_path)
        if scene_prim.IsValid():
            PhysxSchema.PhysxResidualReportingAPI.Apply(scene_prim)
            carb.settings.get_settings().set("/physics/updateResidualsToUsd", True)
            return True
    except (ImportError, AttributeError) as e:
        print(f"[building_physics] Warning: could not enable residual reporting: {e}")
    return False
