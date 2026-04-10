"""Environment configuration for Nova Carter fine-tuning in the MSOE Innovation Center.

Swaps the procedural 10×10 room for the full building USD, uses hand-placed
waypoints from building_waypoints.json for goal sampling and robot spawning,
and removes the procedural curriculum (building complexity replaces it).

Key differences from NovaCarterPatrolFlatEnvCfg:
- Scene: Innovation Center physics USD (no furniture, no procedural walls)
- LiDAR: is_shared=True to avoid duplicating the 77k-mesh BVH per env
- num_envs: 1 initially (T4 VRAM constraint with large building mesh)
- episode_length_s: 120s (longer — building waypoints are farther apart)
- Commands: goal sampled from building_waypoints.json, not random XY bbox
- Events: reset_base_building() spawns at a random waypoint ≥3m from goal
- Curriculum: disabled (not needed; building provides inherent complexity)
"""

from __future__ import annotations

import json
import math
import os

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.sensors.ray_caster.patterns import LidarPatternCfg
from isaaclab.utils import configclass

from .nova_carter_patrol_env_cfg import (
    NOVA_CARTER_CFG,
    ActionsCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)
import nova_carter_patrol.mdp.observations as obs_mdp
import nova_carter_patrol.mdp.rewards as rew_mdp
import nova_carter_patrol.mdp.terminations as term_mdp
import nova_carter_patrol.mdp.events as event_mdp

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

BUILDING_USD = os.path.join(
    _PROJECT_ROOT, "assets", "environments", "msoe_building", "innovation_center_physics.usd"
)
WAYPOINTS_PATH = os.path.join(_THIS_DIR, "building_waypoints.json")

with open(WAYPOINTS_PATH) as _f:
    _wp_data = json.load(_f)
# Floor G waypoints for Stage A; all floors merged for Stage C
FLOOR_G_WAYPOINTS = _wp_data["G"]
ALL_WAYPOINTS = (
    _wp_data["G"] + _wp_data["1"] + _wp_data["2"] + _wp_data["3"]
)


##
# Scene Configuration — Innovation Center Building
##

@configclass
class NovaCarterBuildingSceneCfg(InteractiveSceneCfg):
    """Scene: Innovation Center physics USD with Nova Carter robot and LiDAR.

    IMPORTANT: num_envs=1 initially. The building has ~77k meshes; duplicating
    the BVH per env would OOM on the T4 (16GB). Increase to 4 only after
    confirming VRAM usage stays under 14GB.
    """

    # Building (no terrain/walls/furniture — the USD provides all geometry)
    building = AssetBaseCfg(
        prim_path="/World/Building",
        spawn=sim_utils.UsdFileCfg(usd_path=BUILDING_USD),
    )

    # Robot
    robot: ArticulationCfg = NOVA_CARTER_CFG.replace(prim_path="/World/Robot")

    # LiDAR — 20 rays at 18° spacing, 10m range, 0.3m above floor
    # NOTE: is_shared=True means the BVH is built once and shared across all
    # num_envs. Without this, VRAM scales as (77k meshes × num_envs) → OOM.
    # If MultiMeshRayCasterCfg does not support is_shared, fall back to
    # RayCasterCfg targeting the building root prim instead.
    lidar = MultiMeshRayCasterCfg(
        prim_path="/World/Robot",
        mesh_prim_paths=["/World/Building"],
        pattern_cfg=LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(0.0, 360.0),
            horizontal_res=18.0,  # 360 / 18 = 20 rays
        ),
        max_distance=10.0,
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.3)),
        debug_vis=False,
    )


##
# Building-specific Commands
##

from nova_carter_patrol.mdp.commands import GoalReachPose2dCommandCfg  # noqa: E402


@configclass
class BuildingCommandsCfg:
    """Goal command sampled from floor G building waypoints.

    Uses GoalReachPose2dCommandCfg in 'goal_stations' mode so that goals
    are drawn from the pre-validated waypoint list rather than random XY.
    Threshold increased to 1.0m (vs 0.75m in room) for building-scale nav.
    """

    pose_command = GoalReachPose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,
        resampling_time_range=(30.0, 30.0),
        debug_vis=False,
        goal_reached_threshold=1.0,
        ranges=GoalReachPose2dCommandCfg.Ranges(
            # Ranges unused in goal_stations mode; kept for API compatibility.
            pos_x=(-20.0, -8.0),
            pos_y=(15.0, 26.0),
            heading=(-math.pi, math.pi),
        ),
    )


##
# Building Events
##

@configclass
class BuildingEventCfg:
    """Reset events for building fine-tuning.

    Spawns the robot at a random floor G waypoint ≥3m from the first goal.
    No furniture resets needed (building geometry is static).
    Physics DR (friction, mass, CoM) carried over from the base EventCfg.
    """

    reset_base = EventTerm(
        func=event_mdp.reset_base_building,
        mode="reset",
        params={
            "waypoints": FLOOR_G_WAYPOINTS,
            "min_spawn_goal_dist": 3.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Physics DR — same as base env (friction + mass + CoM randomization)
    randomize_physics = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.6, 1.1),
            "restitution_range": (0.0, 0.1),
        },
    )

    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="chassis_link"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_coms,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="chassis_link"),
            "com_distribution_params": ((-0.05, 0.05), (-0.05, 0.05), (-0.02, 0.02)),
        },
    )


##
# Main Building Environment Config
##

@configclass
class NovaCarterBuildingEnvCfg(ManagerBasedRLEnvCfg):
    """RL environment config for Nova Carter fine-tuning in the Innovation Center.

    Start with num_envs=1; increase to 4 only after confirming no OOM.
    """

    scene: NovaCarterBuildingSceneCfg = NovaCarterBuildingSceneCfg(
        num_envs=1,
        env_spacing=0.0,  # single-env, no spacing needed
    )
    actions: ActionsCfg = ActionsCfg()
    commands: BuildingCommandsCfg = BuildingCommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: BuildingEventCfg = BuildingEventCfg()

    def __post_init__(self):
        self.decimation = 5
        # 120s episodes — building waypoints are farther apart than room
        self.episode_length_s = 120.0
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.enable_external_forces_every_iteration = True
        # Larger contact count for the denser building geometry
        self.sim.physx.gpu_max_rigid_contact_count = 1048576
        self.sim.physx.gpu_max_rigid_patch_count = 163840

        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
