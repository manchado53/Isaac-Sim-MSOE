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


# ---------------------------------------------------------------------------
# Physics Application
# ---------------------------------------------------------------------------

def apply_static_collider(stage, prim, approx="none", deep_search=False):
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
    Returns:
        True if collision was applied, False if no mesh child found
    """
    if deep_search:
        mesh_prim = _find_mesh_descendant(prim)
    else:
        mesh_prim = _find_mesh_child(prim)
    if mesh_prim is None:
        return False

    # CollisionAPI + MeshCollisionAPI both on the Mesh child (where geometry lives).
    # The physics engine needs CollisionAPI on the prim with actual geometry data.
    UsdPhysics.CollisionAPI.Apply(mesh_prim)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
    mesh_collision.CreateApproximationAttr(approx)

    return True


def apply_dynamic_rigid_body(stage, prim, mass=15.0, deep_search=False):
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
    Returns:
        True if physics was applied, False if no mesh child found
    """
    if deep_search:
        mesh_prim = _find_mesh_descendant(prim)
    else:
        mesh_prim = _find_mesh_child(prim)
    if mesh_prim is None:
        return False

    # RigidBodyAPI on the Xform (makes it a dynamic body)
    UsdPhysics.RigidBodyAPI.Apply(prim)

    # CollisionAPI + MeshCollisionAPI on the Mesh child (where geometry lives)
    UsdPhysics.CollisionAPI.Apply(mesh_prim)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
    mesh_collision.CreateApproximationAttr("convexHull")

    # MassAPI on the Xform
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass)

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
                          local_pos0=None, local_pos1=None):
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

    return joint


# ---------------------------------------------------------------------------
# Door Mechanism Classification & Drive
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
    ("Type_2F", DoorMechanism.HINGED_DOUBLE),
    ("Type_2G", DoorMechanism.HINGED_DOUBLE),
    ("2FG", DoorMechanism.HINGED_DOUBLE),
    ("2FU", DoorMechanism.HINGED_DOUBLE),
    ("2F_", DoorMechanism.HINGED_DOUBLE),
    # Single-leaf hinged
    ("Type_F", DoorMechanism.HINGED_SINGLE),
    ("Type_G", DoorMechanism.HINGED_SINGLE),
    ("Type_N", DoorMechanism.HINGED_SINGLE),
    ("Type_FG", DoorMechanism.HINGED_SINGLE),
    ("Curtain_Wall", DoorMechanism.HINGED_SINGLE),
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
    DoorMechanism.HINGED_DOUBLE: (300.0, 150.0, 3000.0),
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


def estimate_door_hinge_offset(stage, door_prim):
    """
    Estimate a hinge position at one vertical edge of a door's bounding box.

    Uses **world-space** bounding box to avoid unit-scale mismatches that
    occur with Revit-exported geometry (local coords may be in centimeters
    or other non-stage units).  The world-space hinge position is then
    inverse-transformed back into body1's local frame for ``local_pos1``.

    The hinge edge is chosen along the door's **narrow horizontal axis**
    (the door width), picking the min edge.  Doors oriented along X get
    a min-X hinge; doors oriented along Y get a min-Y hinge.

    Returns (local_pos0, local_pos1) as Gf.Vec3f tuples suitable for
    create_revolute_joint, or (None, None) if bounding box can't be computed.
    """
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    # --- World-space bounding box (in stage units = meters) ---
    world_bbox = bbox_cache.ComputeWorldBound(door_prim)
    world_range = world_bbox.ComputeAlignedRange()
    w_min = world_range.GetMin()
    w_max = world_range.GetMax()

    if w_min == w_max:
        return None, None

    dx = w_max[0] - w_min[0]
    dy = w_max[1] - w_min[1]

    # Determine door orientation: the wider horizontal axis is the door
    # width, the thinner one is the door depth (thickness).
    # Hinge goes on the min edge of the wider axis.
    if dx >= dy:
        # Door extends along X — hinge at min-X edge
        hinge_x = w_min[0]
        hinge_y = (w_min[1] + w_max[1]) / 2.0
    else:
        # Door extends along Y — hinge at min-Y edge
        hinge_x = (w_min[0] + w_max[0]) / 2.0
        hinge_y = w_min[1]

    hinge_z = (w_min[2] + w_max[2]) / 2.0

    # local_pos0: hinge in body0's frame (body0 = world, so world coords)
    hinge_world = Gf.Vec3d(hinge_x, hinge_y, hinge_z)
    local_pos0 = Gf.Vec3f(float(hinge_world[0]),
                           float(hinge_world[1]),
                           float(hinge_world[2]))

    # local_pos1: hinge in body1's (door's) local frame.
    # Inverse-transform the world hinge position into the door's local space
    # so that units match regardless of internal scale factors.
    xformable = UsdGeom.Xformable(door_prim)
    world_xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    inv_xform = world_xform.GetInverse()
    hinge_local = inv_xform.Transform(hinge_world)
    local_pos1 = Gf.Vec3f(float(hinge_local[0]),
                           float(hinge_local[1]),
                           float(hinge_local[2]))

    return local_pos0, local_pos1


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


def add_door_collision_filters(stage, door_prims, collision_index=None):
    """Disable collision between each door and overlapping static geometry.

    Curtain wall doors from Revit sit flush inside wall/mullion assemblies.
    Without filtering, the door's convex hull collision immediately jams
    against the surrounding static colliders, preventing any rotation.

    This function finds all static colliders whose bounding box overlaps
    each door's initial position and applies ``PhysxFilteredPairsAPI``
    to prevent collision between them.

    Args:
        stage: USD stage.
        door_prims: List of door UsdPrim objects (must have RigidBodyAPI).
        collision_index: Pre-built index from ``_build_collision_index()``.
            If None, one is built automatically.

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


# ---------------------------------------------------------------------------
# Physics Scene Setup
# ---------------------------------------------------------------------------

def setup_physics_scene(stage, gravity=-9.81, time_steps_per_second=60,
                        gpu_dynamics=True, scene_path="/World/PhysicsScene"):
    """
    Create and configure a PhysicsScene with gravity and solver settings.

    Args:
        stage: USD stage
        gravity: Gravitational acceleration (negative = downward)
        time_steps_per_second: Physics solver frequency
        gpu_dynamics: Enable GPU-accelerated dynamics
        scene_path: Where to define the physics scene prim
    """
    scene = UsdPhysics.Scene.Define(stage, scene_path)
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(abs(gravity))

    # Set timestep via physxScene API if available
    physx_prim = stage.GetPrimAtPath(scene_path)
    try:
        from pxr import PhysxSchema
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physx_prim)
        physx_scene.CreateTimeStepsPerSecondAttr(time_steps_per_second)
        if gpu_dynamics:
            physx_scene.CreateEnableGPUDynamicsAttr(True)
            physx_scene.CreateBroadphaseTypeAttr("GPU")
    except (ImportError, AttributeError):
        # PhysxSchema not available — base physics still works
        pass

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
        if child.GetTypeName() == "Scope":
            continue
        if not child.IsA(UsdGeom.Xform):
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
