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
from typing import Optional

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


def categorize_prim(name: str) -> PrimCategory:
    """Categorize a building prim by its name prefix."""
    for prefix, category in _CATEGORY_RULES:
        if name.startswith(prefix):
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


# ---------------------------------------------------------------------------
# Physics Application
# ---------------------------------------------------------------------------

def apply_static_collider(stage, prim, approx="none"):
    """
    Apply static collision to a prim (no RigidBodyAPI = infinite mass, immovable).

    Applies CollisionAPI to the Xform and MeshCollisionAPI to its Mesh child.
    Triangle mesh (approx="none") gives exact collision for static geometry.

    Args:
        stage: USD stage
        prim: Xform prim containing a Mesh child
        approx: Collision approximation — "none" for triangle mesh (static only),
                 "convexHull", "convexDecomposition", "boundingCube", "boundingSphere"
    Returns:
        True if collision was applied, False if no mesh child found
    """
    mesh_prim = _find_mesh_child(prim)
    if mesh_prim is None:
        return False

    # CollisionAPI + MeshCollisionAPI both on the Mesh child (where geometry lives).
    # The physics engine needs CollisionAPI on the prim with actual geometry data.
    UsdPhysics.CollisionAPI.Apply(mesh_prim)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
    mesh_collision.CreateApproximationAttr(approx)

    return True


def apply_dynamic_rigid_body(stage, prim, mass=15.0):
    """
    Apply dynamic rigid body physics to a prim (e.g. a door).

    Adds RigidBodyAPI + MassAPI on the Xform, CollisionAPI + MeshCollisionAPI
    on the Mesh child. Convex hull required for dynamic bodies.

    Args:
        stage: USD stage
        prim: Xform prim containing a Mesh child
        mass: Mass in kg
    Returns:
        True if physics was applied, False if no mesh child found
    """
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


def estimate_door_hinge_offset(stage, door_prim):
    """
    Estimate a hinge position at one vertical edge of a door's bounding box.

    Returns (local_pos0, local_pos1) as Gf.Vec3f tuples suitable for
    create_revolute_joint, or (None, None) if bounding box can't be computed.
    """
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeLocalBound(door_prim)
    box_range = bbox.ComputeAlignedRange()
    bb_min = box_range.GetMin()
    bb_max = box_range.GetMax()

    if bb_min == bb_max:
        return None, None

    # Hinge at the min-X edge of the door (left side when facing it)
    # The joint pivot is at the hinge line — vertical axis through one edge
    hinge_x = bb_min[0]
    hinge_y = (bb_min[1] + bb_max[1]) / 2.0
    hinge_z = (bb_min[2] + bb_max[2]) / 2.0

    # local_pos1 is the offset from the door's origin to the hinge point
    local_pos1 = Gf.Vec3f(float(hinge_x), float(hinge_y), float(hinge_z))

    # local_pos0 is the world-space position of the hinge
    # For a door attached to the world frame, we need the door's world transform
    xformable = UsdGeom.Xformable(door_prim)
    world_xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    hinge_world = world_xform.Transform(Gf.Vec3d(float(hinge_x), float(hinge_y), float(hinge_z)))
    local_pos0 = Gf.Vec3f(float(hinge_world[0]), float(hinge_world[1]), float(hinge_world[2]))

    return local_pos0, local_pos1


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
