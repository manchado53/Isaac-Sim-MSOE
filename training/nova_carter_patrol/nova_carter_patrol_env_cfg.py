"""Environment configuration for Nova Carter Patrol indoor room training.

Uses a Nova Carter robot with differential drive wheels,
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
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.sensors.ray_caster.patterns import LidarPatternCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from nova_carter_patrol.mdp import observations as obs_mdp
from nova_carter_patrol.mdp import rewards as rew_mdp
from nova_carter_patrol.mdp.actions import DifferentialDriveActionCfg
from nova_carter_patrol.mdp.commands import GoalReachPose2dCommandCfg
from nova_carter_patrol.mdp import curriculums as curr_mdp
from nova_carter_patrol.mdp import terminations as term_mdp
from nova_carter_patrol.mdp import events as event_mdp
from nova_carter_patrol.mdp import curriculum_utils as cu


# Shared rigid body props for static objects (walls + furniture)
_STATIC_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
_STATIC_MASS_PROPS = sim_utils.MassPropertiesCfg(mass=100.0)
_STATIC_COLLISION_PROPS = sim_utils.CollisionPropertiesCfg(collision_enabled=True)


##
# Robot Configuration
##

NOVA_CARTER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/NovaCarter/nova_carter.usd",
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1),  # slight Z offset to settle onto ground
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["joint_wheel_left", "joint_wheel_right"],
            effort_limit_sim=1000.0,
            stiffness=0.0,       # pure velocity control
            damping=1e5,
        ),
        "casters": ImplicitActuatorCfg(
            joint_names_expr=["joint_caster_.*"],
            effort_limit_sim=0.0,   # passive, free-spinning
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""Nova Carter with differential drive wheels and passive casters."""


##
# Scene Configuration — Indoor Room
##

@configclass
class NovaCarterPatrolSceneCfg(InteractiveSceneCfg):
    """Scene: walled 10m x 10m room with furniture, Nova Carter robot, and LiDAR."""

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
    robot: ArticulationCfg = NOVA_CARTER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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

    # Table — wide, low (max dim 1.2 → 0.6m wall buffer)
    furniture_table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Furniture_table",
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 0.8, 0.75),
            rigid_props=_STATIC_RIGID_PROPS,
            mass_props=_STATIC_MASS_PROPS,
            collision_props=_STATIC_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.35, 0.17)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.375)),
    )

    # Bookshelf — tall, narrow (max dim 1.2 → 0.6m wall buffer)
    furniture_bookshelf = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Furniture_bookshelf",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 1.2, 1.8),
            rigid_props=_STATIC_RIGID_PROPS,
            mass_props=_STATIC_MASS_PROPS,
            collision_props=_STATIC_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.25, 0.12)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.9)),
    )

    # Chair — small, medium height (max dim 0.45 → 0.25m wall buffer)
    furniture_chair = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Furniture_chair",
        spawn=sim_utils.CuboidCfg(
            size=(0.45, 0.45, 0.9),
            rigid_props=_STATIC_RIGID_PROPS,
            mass_props=_STATIC_MASS_PROPS,
            collision_props=_STATIC_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.45)),
    )

    # Desk — wide, standard height (max dim 1.5 → 0.75m wall buffer)
    furniture_desk = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Furniture_desk",
        spawn=sim_utils.CuboidCfg(
            size=(1.5, 0.7, 0.75),
            rigid_props=_STATIC_RIGID_PROPS,
            mass_props=_STATIC_MASS_PROPS,
            collision_props=_STATIC_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.5, 0.35)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.375)),
    )

    # Filing cabinet — tall, compact (max dim 0.5 → 0.25m wall buffer)
    furniture_cabinet = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Furniture_cabinet",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.4, 1.4),
            rigid_props=_STATIC_RIGID_PROPS,
            mass_props=_STATIC_MASS_PROPS,
            collision_props=_STATIC_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.7)),
    )

    # Couch — long, low (max dim 1.8 → 0.9m wall buffer)
    furniture_couch = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Furniture_couch",
        spawn=sim_utils.CuboidCfg(
            size=(1.8, 0.7, 0.6),
            rigid_props=_STATIC_RIGID_PROPS,
            mass_props=_STATIC_MASS_PROPS,
            collision_props=_STATIC_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.3, 0.55)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.3)),
    )

    # Trash bin — small (max dim 0.3 → 0.15m wall buffer)
    furniture_bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Furniture_bin",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, 0.5),
            rigid_props=_STATIC_RIGID_PROPS,
            mass_props=_STATIC_MASS_PROPS,
            collision_props=_STATIC_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.25, 0.15)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25)),
    )

    # LiDAR sensor (2D, 20 rays, 360-degree, 10m range)
    # Mounted on Nova Carter's chassis_link (the main body)
    lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/chassis_link",
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

    # Contact sensor — detects collisions on the chassis only (excludes wheel/ground contact)
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/chassis_link",
        update_period=0,
        history_length=5,   # matches decimation=5 — captures all physics sub-steps in one RL step
        track_air_time=False,
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
    """2D velocity commands -> wheel velocities via DifferentialDriveAction."""

    base_velocity = DifferentialDriveActionCfg(
        asset_name="robot",
        left_wheel_joint="joint_wheel_left",
        right_wheel_joint="joint_wheel_right",
        wheel_radius=0.14,
        wheel_base=0.4,
        scale=(0.5, 1.0),  # max 0.5 m/s linear, 1.0 rad/s angular
        offset=(0.0, 0.0),
    )


##
# Commands
##

@configclass
class CommandsCfg:
    """Random 2D goal pose sampling with goal-reached resampling."""

    pose_command = GoalReachPose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,
        resampling_time_range=(15.0, 15.0),  # fallback timer for unreachable goals
        debug_vis=True,
        goal_reached_threshold=0.75,  # resample immediately when within 0.75m
        ranges=GoalReachPose2dCommandCfg.Ranges(
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
    """7-term reward: delta progress + distance magnet + time penalty + arrival bonus + collision + backward motion + obstacle proximity.

    delta_progress (+3.0): rewards net distance closed per step (d_prev - d_current).
        Cannot be gamed by oscillating; 10x stronger than time penalty per step of good movement.
    distance_to_goal (+1.0, std=0.3): steep tanh magnet active only inside 0.75m — the gradient
        collapses inside the goal zone, pulling the robot through the final gap.
    time_penalty (-0.05): constant cost per step so hovering and circling are always losing.
        Robot still gains net reward by moving at ≥0.017 m/s toward goal (3.0×0.017 > 0.05).
    goal_reached (+100.0): dominant arrival bonus — large enough that reaching is always
        the best strategy over any finite horizon of hovering.
    collision (-20.0): strong deterrent without freezing behavior — -100.0 equaled goal reward per step, causing the robot to stand still.
    obstacle_proximity (-0.5, safe_dist=1.0): continuous LiDAR-based penalty within 1m of any
        obstacle. Bridges LiDAR observations → reward so PPO learns anticipatory avoidance
        rather than crash-then-learn. At 0.5m: -0.021/step; goal_reached still 200× stronger
        so the robot still threads narrow gaps when required.
    """

    # Net distance closed per step — anti-oscillation progress signal
    delta_progress = RewTerm(
        func=rew_mdp.delta_progress,
        weight=3.0,
        params={"command_name": "pose_command"},
    )
    # Steep proximity magnet — pulls robot through the final 0.75m gap
    distance_to_goal = RewTerm(
        func=rew_mdp.position_tracking_tanh,
        weight=1.0,
        params={"command_name": "pose_command", "std": 0.3},
    )
    # Per-step time cost — makes hovering always net-negative
    time_penalty = RewTerm(
        func=rew_mdp.time_penalty,
        weight=-0.05,
    )
    # Penalty for driving casters-first — encourages sensor end to lead
    backward_motion = RewTerm(
        func=rew_mdp.backward_motion_penalty,
        weight=-2.0,
    )
    # Dominant arrival bonus
    goal_reached = RewTerm(
        func=rew_mdp.goal_reached,
        weight=100.0,
        params={"command_name": "pose_command", "threshold": 0.75},
    )
    # Collision penalty via contact sensor
    collision = RewTerm(
        func=mdp.undesired_contacts,
        weight=-20.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="chassis_link"), "threshold": 1.0},
    )
    # Continuous LiDAR proximity penalty — fires before contact, bridges LiDAR obs → reward gap.
    # Returns max(0, 1 - min_lidar/safe_dist): 0 outside 1m, rising to ~1 at contact.
    # Teaches anticipatory avoidance (steer away at 1m) rather than reactive (crash then learn).
    obstacle_proximity = RewTerm(
        func=rew_mdp.obstacle_proximity,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "safe_dist": 1.0},
    )


##
# Terminations
##

@configclass
class TerminationsCfg:
    """Termination conditions.

    Uses patrol_complete to end episodes after 6 successful goal reaches,
    teaching patrol behavior (goal -> goal -> goal). time_out remains as a
    safety fallback.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.52},
    )
    patrol_complete = DoneTerm(
        func=term_mdp.patrol_complete,
        params={"command_name": "pose_command", "num_goals": 6},
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
        func=event_mdp.reset_base_outer_ring,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "ring_mode": False,
            "inner_radius": 2.5,
            "outer_radius": 4.5,
            "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
        },
    )

    # Reset ALL joint positions to defaults on every episode reset.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Furniture randomized in specific zones AWAY from the robot's spawn area.
    # init_state is at origin, so pose_range = absolute room coordinates.
    # Each zone avoids robot spawn (±1.0m center) and stays inside walls
    # with margin for furniture size under any rotation (max_dim/2).
    # Room interior: ~±4.9m (walls at ±5.0, 0.15m thick)

    # Table (1.2x0.6) → 0.6m buffer → NE quadrant
    reset_furniture_table = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("furniture_table"),
            "pose_range": {"x": (1.5, 4.3), "y": (1.5, 4.3), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
        },
    )
    # Bookshelf (0.4x1.2) → 0.6m buffer → NW quadrant
    reset_furniture_bookshelf = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("furniture_bookshelf"),
            "pose_range": {"x": (-4.3, -1.5), "y": (1.5, 4.3), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
        },
    )
    # Chair (0.45x0.45) → 0.25m buffer → SE quadrant
    reset_furniture_chair = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("furniture_chair"),
            "pose_range": {"x": (1.5, 4.5), "y": (-4.5, -1.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
        },
    )
    # Desk (1.5x0.7) → 0.75m buffer → SW quadrant
    reset_furniture_desk = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("furniture_desk"),
            "pose_range": {"x": (-4.0, -1.5), "y": (-4.0, -1.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
        },
    )
    # Cabinet (0.5x0.4) → 0.25m buffer → E edge
    reset_furniture_cabinet = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("furniture_cabinet"),
            "pose_range": {"x": (1.5, 4.5), "y": (-1.5, 1.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
        },
    )
    # Couch (1.8x0.7) → 0.9m buffer → W edge
    reset_furniture_couch = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("furniture_couch"),
            "pose_range": {"x": (-4.0, -1.5), "y": (-1.5, 1.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
        },
    )
    # Bin (0.3x0.3) → 0.15m buffer → N edge
    reset_furniture_bin = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("furniture_bin"),
            "pose_range": {"x": (-1.5, 1.5), "y": (1.5, 4.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
        },
    )

    # Post-placement separation enforcement for the 3 scatter pieces (Stages 3+4).
    # Runs last — after individual furniture resets — and resamples any pair whose
    # centres are closer than SCATTER_MIN_SEP.  No-op in Stages 1+2 (pieces in corners).
    enforce_scatter_sep = EventTerm(
        func=event_mdp.enforce_scatter_separation,
        mode="reset",
        params={
            "piece_names": ["furniture_table", "furniture_couch", "furniture_desk"],
            "scatter_zone": cu.SCATTER_ZONE,
            "min_separation": cu.SCATTER_MIN_SEP,
            "enabled": False,   # disabled until Stage 3; curriculum sets True on transition
        },
    )


##
# Curriculum
##

@configclass
class CurriculumCfg:
    """Progressive difficulty: empty room → 3 obstacles → full clutter."""

    navigation_difficulty = CurrTerm(
        func=curr_mdp.NavigationCurriculum,
        params={
            "stage1_steps": 51200,    # ~100 iters → Stage 2 (center blocker)
            "stage2_steps": 128000,   # ~250 iters → Stage 3 (scatter)
            "stage3_steps": 230400,   # ~450 iters → Stage 4 (moving obstacles)
        },
    )


##
# Main Environment Config
##

@configclass
class NovaCarterPatrolFlatEnvCfg(ManagerBasedRLEnvCfg):
    """Full RL environment config for Nova Carter Patrol in an indoor room."""

    scene: NovaCarterPatrolSceneCfg = NovaCarterPatrolSceneCfg(
        num_envs=256,
        env_spacing=12.0,
    )
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

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
