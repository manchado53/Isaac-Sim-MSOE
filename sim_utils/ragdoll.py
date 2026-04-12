"""
Character ragdoll builder — 11 rigid-body capsules + 10 joints.

Assembles a humanoid ragdoll from primitive capsules connected by
spherical and revolute joints, sized and positioned to approximate a
1.73 m adult. Intended use: drop a ragdoll into a scene under gravity
for anomaly-detection training data ("person fallen on the floor").

Usage::

    from sim_utils.ragdoll import build_character_ragdoll, set_ragdoll_kinematic

    bodies = build_character_ragdoll(
        stage,
        root_prim_path="/World/Ragdoll",
        world_position=(-37.0, 74.0, 13.56),
        kinematic=True,
    )
    # ... run 30 frames so the character stands still ...
    set_ragdoll_kinematic(bodies, False)
    # ... gravity now pulls the ragdoll to the floor ...

The builder does NOT attach a skinned mesh — the rigid capsules ARE the
visual. Adding mesh skinning on top is a deferred follow-up (requires
per-frame SkelAnimation updates driven by rigid body transforms).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from pxr import Gf, Sdf, Usd, UsdPhysics

try:
    from pxr import PhysxSchema
    _HAS_PHYSX_SCHEMA = True
except ImportError:
    _HAS_PHYSX_SCHEMA = False

from sim_utils.building_physics import (
    apply_capsule_rigid_body,
    create_revolute_joint,
    create_spherical_joint,
    set_rigid_body_kinematic,
)


# ---------------------------------------------------------------------------
# Body + joint definitions (in character-local frame, Z-up, feet at Z=0)
# ---------------------------------------------------------------------------

# Each entry: name → dict with capsule params + local pose.
# "position" is the capsule CENTER in character-local frame.
# "axis" is the local axis along which the capsule extends (its "length" dir).
# Capsule.AxisAttr supports only "X"/"Y"/"Z" — we use "Z" for all bones
# since they extend vertically in the standing pose; post-pose rotation
# via joints/gravity will reorient them.
RAGDOLL_BODIES: Dict[str, dict] = {
    # name             center (x, y, z)  length  radius  mass  color
    "Pelvis":    dict(position=(0.0, 0.0, 0.92),  length=0.15, radius=0.13, mass=10.0, color=(0.4, 0.4, 0.8)),
    "Torso":     dict(position=(0.0, 0.0, 1.25),  length=0.40, radius=0.15, mass=20.0, color=(0.6, 0.3, 0.3)),
    "Head":      dict(position=(0.0, 0.0, 1.67),  length=0.10, radius=0.11, mass=4.0,  color=(0.9, 0.7, 0.5)),
    "L_UpperArm":dict(position=(0.22, 0.0, 1.30), length=0.27, radius=0.06, mass=3.0,  color=(0.6, 0.3, 0.3)),
    "L_ForeArm": dict(position=(0.22, 0.0, 1.00), length=0.27, radius=0.05, mass=2.0,  color=(0.5, 0.3, 0.3)),
    "R_UpperArm":dict(position=(-0.22, 0.0, 1.30),length=0.27, radius=0.06, mass=3.0,  color=(0.6, 0.3, 0.3)),
    "R_ForeArm": dict(position=(-0.22, 0.0, 1.00),length=0.27, radius=0.05, mass=2.0,  color=(0.5, 0.3, 0.3)),
    "L_Thigh":   dict(position=(0.10, 0.0, 0.60), length=0.45, radius=0.08, mass=8.0,  color=(0.3, 0.3, 0.6)),
    "L_Calf":    dict(position=(0.10, 0.0, 0.18), length=0.40, radius=0.06, mass=5.0,  color=(0.3, 0.3, 0.5)),
    "R_Thigh":   dict(position=(-0.10, 0.0, 0.60),length=0.45, radius=0.08, mass=8.0,  color=(0.3, 0.3, 0.6)),
    "R_Calf":    dict(position=(-0.10, 0.0, 0.18),length=0.40, radius=0.06, mass=5.0,  color=(0.3, 0.3, 0.5)),
}


# Each joint: (body0, body1, kind, pivot_world_local, limits)
# pivot is the point in the *character-local* frame where the bodies connect.
# For each body, we compute local_pos = pivot - body_center.
RAGDOLL_JOINTS: List[dict] = [
    # Waist: pelvis top → torso bottom, spherical, small cone
    dict(name="Waist",     body0="Pelvis",    body1="Torso",     kind="spherical", pivot=(0.0, 0.0, 1.00),  cone=(25.0, 25.0)),
    # Neck: torso top → head bottom
    dict(name="Neck",      body0="Torso",     body1="Head",      kind="spherical", pivot=(0.0, 0.0, 1.56),  cone=(40.0, 40.0)),
    # Shoulders
    dict(name="L_Shoulder",body0="Torso",     body1="L_UpperArm",kind="spherical", pivot=(0.20, 0.0, 1.45),  cone=(85.0, 85.0)),
    dict(name="R_Shoulder",body0="Torso",     body1="R_UpperArm",kind="spherical", pivot=(-0.20, 0.0, 1.45), cone=(85.0, 85.0)),
    # Elbows (revolute around local X of the upperarm)
    dict(name="L_Elbow",   body0="L_UpperArm",body1="L_ForeArm", kind="revolute",  pivot=(0.22, 0.0, 1.15),  axis="X", lower=-140.0, upper=0.0),
    dict(name="R_Elbow",   body0="R_UpperArm",body1="R_ForeArm", kind="revolute",  pivot=(-0.22, 0.0, 1.15), axis="X", lower=-140.0, upper=0.0),
    # Hips
    dict(name="L_Hip",     body0="Pelvis",    body1="L_Thigh",   kind="spherical", pivot=(0.10, 0.0, 0.85),  cone=(55.0, 55.0)),
    dict(name="R_Hip",     body0="Pelvis",    body1="R_Thigh",   kind="spherical", pivot=(-0.10, 0.0, 0.85), cone=(55.0, 55.0)),
    # Knees (revolute around local X of the thigh)
    dict(name="L_Knee",    body0="L_Thigh",   body1="L_Calf",    kind="revolute",  pivot=(0.10, 0.0, 0.38),  axis="X", lower=0.0,    upper=145.0),
    dict(name="R_Knee",    body0="R_Thigh",   body1="R_Calf",    kind="revolute",  pivot=(-0.10, 0.0, 0.38), axis="X", lower=0.0,    upper=145.0),
]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_character_ragdoll(stage, root_prim_path: str,
                             world_position: Tuple[float, float, float],
                             kinematic: bool = True
                             ) -> Dict[str, object]:
    """Build 11 capsule rigid bodies + 10 joints for a humanoid ragdoll.

    Args:
        stage: USD stage
        root_prim_path: Parent prim under which to create bodies & joints
                        (e.g. "/World/Ragdoll")
        world_position: (x, y, z) where the character's feet-origin lands
        kinematic: Start with all bodies kinematic (standing still).
                   Call set_ragdoll_kinematic(bodies, False) to drop.

    Returns:
        dict[str, Usd.Prim] mapping body name → capsule prim
    """
    # Ensure root Xform exists, mark it as an articulation root.
    # This is THE critical step — PhysX now treats all descendant rigid
    # bodies + joints as a single coupled articulated system, solved with
    # a reduced-coordinate solver that guarantees joints cannot separate.
    # Without this, each joint is an independent constraint pair and the
    # system can explode under stress. See:
    #   extscache/omni.physx.demos-*/scenes/ArticulationDemo.py  line 28
    root_prim = stage.DefinePrim(root_prim_path, "Xform")
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    if _HAS_PHYSX_SCHEMA:
        phys_art = PhysxSchema.PhysxArticulationAPI.Apply(root_prim)
        # Higher solver iterations = more stable under high forces / contacts
        phys_art.CreateSolverPositionIterationCountAttr(32)
        phys_art.CreateSolverVelocityIterationCountAttr(8)
        # No self-collisions — adjacent links in the articulation tree never
        # collide by default; this also skips non-adjacent self-collision
        # (cheaper, and fine for a humanoid ragdoll where the main risk is
        # limbs interpenetrating briefly during the fall)
        phys_art.CreateEnabledSelfCollisionsAttr(False)

    bodies = {}
    base_x, base_y, base_z = (float(world_position[0]),
                               float(world_position[1]),
                               float(world_position[2]))

    # --- Create bodies ---
    for name, spec in RAGDOLL_BODIES.items():
        body_prim_path = f"{root_prim_path}/{name}"
        local_x, local_y, local_z = spec["position"]
        world_pos = (base_x + local_x, base_y + local_y, base_z + local_z)

        prim = apply_capsule_rigid_body(
            stage,
            prim_path=body_prim_path,
            length=spec["length"],
            radius=spec["radius"],
            mass=spec["mass"],
            world_position=world_pos,
            axis="Z",  # capsule extends along local Z (vertical in standing)
            kinematic=kinematic,
            display_color=spec.get("color"),
        )
        bodies[name] = prim

    # --- Create joints ---
    joints_root = f"{root_prim_path}/Joints"
    stage.DefinePrim(joints_root, "Xform")

    for j in RAGDOLL_JOINTS:
        jpath = f"{joints_root}/{j['name']}"
        body0_name = j["body0"]
        body1_name = j["body1"]
        body0_path = str(bodies[body0_name].GetPath())
        body1_path = str(bodies[body1_name].GetPath())

        # Compute joint pivot in each body's LOCAL frame.
        # Body center in character-local space:
        b0_center = RAGDOLL_BODIES[body0_name]["position"]
        b1_center = RAGDOLL_BODIES[body1_name]["position"]
        pivot = j["pivot"]

        local_pos0 = Gf.Vec3f(float(pivot[0] - b0_center[0]),
                               float(pivot[1] - b0_center[1]),
                               float(pivot[2] - b0_center[2]))
        local_pos1 = Gf.Vec3f(float(pivot[0] - b1_center[0]),
                               float(pivot[1] - b1_center[1]),
                               float(pivot[2] - b1_center[2]))

        if j["kind"] == "spherical":
            cone0, cone1 = j["cone"]
            create_spherical_joint(
                stage, jpath, body0_path, body1_path,
                cone_angle0=cone0, cone_angle1=cone1,
                local_pos0=local_pos0, local_pos1=local_pos1,
                axis="Y",
            )
        elif j["kind"] == "revolute":
            create_revolute_joint(
                stage, jpath, body0_path, body1_path,
                axis=j.get("axis", "X"),
                lower_limit=float(j["lower"]),
                upper_limit=float(j["upper"]),
                local_pos0=local_pos0, local_pos1=local_pos1,
            )
        else:
            raise ValueError(f"Unknown joint kind: {j['kind']}")

    # --- Collision filtering between joint-connected bodies ---
    # Without this, adjacent capsules that touch at the joint pivot push each
    # other apart when physics activates, overwhelming the joint and causing
    # the ragdoll to "explode" into disconnected pieces.
    # Pattern: apply UsdPhysics.FilteredPairsAPI to body0, add body1 as
    # filtered target. PhysX ignores collision between listed pairs.
    for j in RAGDOLL_JOINTS:
        body0 = bodies[j["body0"]]
        body1 = bodies[j["body1"]]
        filter_api = UsdPhysics.FilteredPairsAPI.Apply(body0)
        rel = filter_api.CreateFilteredPairsRel()
        # Merge with any existing targets
        existing = list(rel.GetTargets() or [])
        if body1.GetPath() not in existing:
            existing.append(body1.GetPath())
            rel.SetTargets(existing)

    return bodies


def set_ragdoll_kinematic(bodies: Dict[str, object], kinematic: bool) -> None:
    """Flip all ragdoll bodies between kinematic (held still) and dynamic.

    Args:
        bodies: dict returned by build_character_ragdoll
        kinematic: True = held in place by script; False = physics-driven
    """
    for name, prim in bodies.items():
        set_rigid_body_kinematic(prim, kinematic)
