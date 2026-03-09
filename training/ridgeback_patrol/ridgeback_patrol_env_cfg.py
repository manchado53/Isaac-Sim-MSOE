"""Environment configuration for Ridgeback Patrol indoor room training.

Uses a Ridgeback robot with dummy base joints (NonHolonomicAction),
2D LiDAR via MultiMeshRayCaster, a walled room with furniture obstacles,
and UniformPose2dCommand for goal-directed navigation.
"""

from __future__ import annotations

import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
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
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from ridgeback_patrol.mdp import observations as obs_mdp
from ridgeback_patrol.mdp import rewards as rew_mdp
from ridgeback_patrol.mdp import terminations as term_mdp


# Shared rigid body props for static objects (walls + furniture)
_STATIC_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
_STATIC_MASS_PROPS = sim_utils.MassPropertiesCfg(mass=100.0)
_STATIC_COLLISION_PROPS = sim_utils.CollisionPropertiesCfg()


##
# Robot Configuration
##

RIDGEBACK_BASE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "dummy_base_prismatic_x_joint": 0.0,
            "dummy_base_prismatic_y_joint": 0.0,
            "dummy_base_revolute_z_joint": 0.0,
            # Lock arm joints in a neutral pose
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 2.0,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.035,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "base": ImplicitActuatorCfg(
            joint_names_expr=["dummy_base_.*"],
            effort_limit_sim=1000.0,
            stiffness=0.0,
            damping=1e5,
        ),
    },
)
"""Ridgeback base config with only dummy base joints actuated (arm ignored)."""


##
# Scene Configuration — Indoor Room
##

@configclass
class RidgebackPatrolSceneCfg(InteractiveSceneCfg):
    """Scene: walled 10m x 10m room with furniture, Ridgeback robot, and LiDAR."""

    # Terrain (floor)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Robot
    robot: ArticulationCfg = RIDGEBACK_BASE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # ── Walls (10m x 10m room, 2.5m tall, 0.15m thick) ──
    # Walls are static and NOT randomized at reset
    wall_north = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Wall_north",
        spawn=sim_utils.CuboidCfg(
            size=(10.3, 0.15, 2.5),
            rigid_props=_STATIC_RIGID_PROPS,
            mass_props=_STATIC_MASS_PROPS,
            collision_props=_STATIC_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.83, 0.78)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 5.0, 1.25)),
    )
    wall_south = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Wall_south",
        spawn=sim_utils.CuboidCfg(
            size=(10.3, 0.15, 2.5),
            rigid_props=_STATIC_RIGID_PROPS,
            mass_props=_STATIC_MASS_PROPS,
            collision_props=_STATIC_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.83, 0.78)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -5.0, 1.25)),
    )
    wall_east = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Wall_east",
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 10.0, 2.5),
            rigid_props=_STATIC_RIGID_PROPS,
            mass_props=_STATIC_MASS_PROPS,
            collision_props=_STATIC_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.83, 0.78)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5.0, 0.0, 1.25)),
    )
    wall_west = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Wall_west",
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 10.0, 2.5),
            rigid_props=_STATIC_RIGID_PROPS,
            mass_props=_STATIC_MASS_PROPS,
            collision_props=_STATIC_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.83, 0.78)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-5.0, 0.0, 1.25)),
    )

    # ── Furniture (randomized positions at reset) ──

    # Table — wide, low
    furniture_table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Furniture_table",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.6, 0.75),
            rigid_props=_STATIC_RIGID_PROPS,
            mass_props=_STATIC_MASS_PROPS,
            collision_props=_STATIC_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.35, 0.17)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, 1.0, 0.375)),
    )

    # Bookshelf — tall, narrow
    furniture_bookshelf = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Furniture_bookshelf",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 1.2, 1.8),
            rigid_props=_STATIC_RIGID_PROPS,
            mass_props=_STATIC_MASS_PROPS,
            collision_props=_STATIC_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.25, 0.12)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-3.0, 3.0, 0.9)),
    )

    # Chair — small, medium height
    furniture_chair = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Furniture_chair",
        spawn=sim_utils.CuboidCfg(
            size=(0.45, 0.45, 0.9),
            rigid_props=_STATIC_RIGID_PROPS,
            mass_props=_STATIC_MASS_PROPS,
            collision_props=_STATIC_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, -2.0, 0.45)),
    )

    # Desk — wide, standard height
    furniture_desk = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Furniture_desk",
        spawn=sim_utils.CuboidCfg(
            size=(1.5, 0.7, 0.75),
            rigid_props=_STATIC_RIGID_PROPS,
            mass_props=_STATIC_MASS_PROPS,
            collision_props=_STATIC_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.5, 0.35)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.0, -1.0, 0.375)),
    )

    # Filing cabinet — tall, compact
    furniture_cabinet = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Furniture_cabinet",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.4, 1.4),
            rigid_props=_STATIC_RIGID_PROPS,
            mass_props=_STATIC_MASS_PROPS,
            collision_props=_STATIC_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.5, -3.5, 0.7)),
    )

    # Couch — long, low
    furniture_couch = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Furniture_couch",
        spawn=sim_utils.CuboidCfg(
            size=(1.8, 0.7, 0.6),
            rigid_props=_STATIC_RIGID_PROPS,
            mass_props=_STATIC_MASS_PROPS,
            collision_props=_STATIC_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.3, 0.55)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.0, 3.5, 0.3)),
    )

    # Trash bin — small
    furniture_bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Furniture_bin",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, 0.5),
            rigid_props=_STATIC_RIGID_PROPS,
            mass_props=_STATIC_MASS_PROPS,
            collision_props=_STATIC_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.25, 0.15)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.5, -1.5, 0.25)),
    )

    # LiDAR sensor (2D, 20 rays, 360-degree, 10m range)
    lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0,
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.3)),
        ray_alignment="yaw",
        pattern_cfg=LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(0.0, 360.0),
            horizontal_res=18.0,  # 360/18 = 20 rays
        ),
        max_distance=10.0,
        mesh_prim_paths=[
            "/World/ground",
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/Wall_.*",
                is_shared=False,
                track_mesh_transforms=False,  # walls don't move
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/Furniture_.*",
                is_shared=False,
                track_mesh_transforms=True,
            ),
        ],
        debug_vis=False,
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# Actions
##

@configclass
class ActionsCfg:
    """2D velocity commands -> dummy base joints via NonHolonomicAction."""

    base_velocity = mdp.NonHolonomicActionCfg(
        asset_name="robot",
        body_name="base_link",
        x_joint_name="dummy_base_prismatic_x_joint",
        y_joint_name="dummy_base_prismatic_y_joint",
        yaw_joint_name="dummy_base_revolute_z_joint",
        scale=(0.5, 1.0),  # 0.5 m/s max linear, 1.0 rad/s max angular
        offset=(0.0, 0.0),
    )


##
# Commands
##

@configclass
class CommandsCfg:
    """Random 2D goal pose sampling (within the room)."""

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,
        resampling_time_range=(83.0, 83.0),
        debug_vis=False,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-4.0, 4.0),  # stay inside the 10m room walls
            pos_y=(-4.0, 4.0),
            heading=(-math.pi, math.pi),
        ),
    )


##
# Observations (26-dim total)
##

@configclass
class ObservationsCfg:
    """Observation specification for the policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        """26-dim observation: LiDAR(20) + goal_pos(2) + goal_heading(2) + fwd_vel(1) + yaw_rate(1)."""

        lidar_scan = ObsTerm(
            func=obs_mdp.lidar_scan_normalized,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
        )
        goal_position = ObsTerm(
            func=obs_mdp.relative_goal_position,
            params={"command_name": "pose_command"},
        )
        goal_heading = ObsTerm(
            func=obs_mdp.relative_goal_heading,
            params={"command_name": "pose_command"},
        )
        forward_velocity = ObsTerm(func=obs_mdp.forward_velocity)
        yaw_rate = ObsTerm(func=obs_mdp.yaw_rate)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##
# Rewards
##

@configclass
class RewardsCfg:
    """Reward terms for navigation with obstacle avoidance."""

    goal_progress = RewTerm(
        func=rew_mdp.GoalProgressReward,
        weight=5.0,
        params={"command_name": "pose_command"},
    )
    goal_reached = RewTerm(
        func=rew_mdp.goal_reached,
        weight=10.0,
        params={"command_name": "pose_command", "threshold": 0.5},
    )
    collision = RewTerm(
        func=rew_mdp.collision_penalty,
        weight=-2.0,  # reduced from -10.0 to prevent reward spikes that cause NaN
        params={"sensor_cfg": SceneEntityCfg("lidar"), "threshold": 0.15},
    )
    obstacle_proximity = RewTerm(
        func=rew_mdp.obstacle_proximity,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "safe_dist": 0.5},
    )
    action_smoothness = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    # Survival bonus: positive reward each step to prevent NaN from
    # all-negative reward distributions crushing VecNormalize variance
    survival = RewTerm(func=mdp.is_alive, weight=0.1)


##
# Terminations
##

@configclass
class TerminationsCfg:
    """Termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    goal_reached = DoneTerm(
        func=term_mdp.goal_reached,
        params={"command_name": "pose_command", "threshold": 0.5},
    )
    # Collision is handled as a reward penalty only (weight=-10.0 in RewardsCfg).
    # Terminating on collision causes a death spiral: robot spawns near furniture,
    # dies in ~9 steps, PPO gets no useful signal, gradients explode to NaN.
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.52},
    )


##
# Events (reset randomization)
##

@configclass
class EventCfg:
    """Reset events for robot and furniture. Walls are NOT randomized."""

    # Robot resets near room center (±1.0m — reduced from ±2.0m to avoid
    # spawning on top of furniture, which caused instant collisions and NaN)
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
        },
    )

    # Furniture randomized in specific zones AWAY from the robot's spawn area.
    # Each piece is assigned a quadrant or edge zone (min 1.5m from center)
    # to prevent overlap with the robot at ±1.0m.
    reset_furniture_table = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("furniture_table"),
            "pose_range": {"x": (1.5, 3.5), "y": (1.5, 3.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
        },
    )
    reset_furniture_bookshelf = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("furniture_bookshelf"),
            "pose_range": {"x": (-3.5, -1.5), "y": (1.5, 3.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
        },
    )
    reset_furniture_chair = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("furniture_chair"),
            "pose_range": {"x": (1.5, 3.5), "y": (-3.5, -1.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
        },
    )
    reset_furniture_desk = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("furniture_desk"),
            "pose_range": {"x": (-3.5, -1.5), "y": (-3.5, -1.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
        },
    )
    reset_furniture_cabinet = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("furniture_cabinet"),
            "pose_range": {"x": (2.0, 4.0), "y": (-1.5, 1.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
        },
    )
    reset_furniture_couch = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("furniture_couch"),
            "pose_range": {"x": (-4.0, -2.0), "y": (-1.5, 1.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
        },
    )
    reset_furniture_bin = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("furniture_bin"),
            "pose_range": {"x": (-1.5, 1.5), "y": (2.0, 4.0), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
        },
    )


##
# Main Environment Config
##

@configclass
class RidgebackPatrolFlatEnvCfg(ManagerBasedRLEnvCfg):
    """Full RL environment config for Ridgeback Patrol in an indoor room."""

    scene: RidgebackPatrolSceneCfg = RidgebackPatrolSceneCfg(
        num_envs=256,
        env_spacing=12.0,
    )
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 5
        self.episode_length_s = 83.0

        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.enable_external_forces_every_iteration = True
        self.sim.physx.gpu_max_rigid_contact_count = 524288
        self.sim.physx.gpu_max_rigid_patch_count = 81920

        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
