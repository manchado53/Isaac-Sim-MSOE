"""Preview the Carter Patrol indoor room environment — renders an MP4 video.

Uses InteractiveScene to spawn multiple fully populated environments
(each with its own robot, walls, and furniture), just like the real RL training.

Usage (on a GPU node):
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/isaac-sim-msoe/scripts/preview_env.py --headless --enable_cameras

Output: ~/isaac-sim-msoe/videos/env_preview.mp4
"""

from __future__ import annotations

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# -- CLI args --
parser = argparse.ArgumentParser(description="Preview Carter Patrol environment scene.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- Now safe to import Isaac Sim / Isaac Lab modules --
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Add project root for VideoRecorder import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sim_utils.video_capture import VideoRecorder


NUM_ENVS = 16

# Shared rigid body props for static objects
_STATIC_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
_STATIC_MASS_PROPS = sim_utils.MassPropertiesCfg(mass=100.0)
_STATIC_COLLISION_PROPS = sim_utils.CollisionPropertiesCfg()


@configclass
class PreviewSceneCfg(InteractiveSceneCfg):
    """Indoor room scene with robot, walls, and furniture — cloned per env."""

    # Floor
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
    )

    # Robot
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "dummy_base_prismatic_x_joint": 0.0,
                "dummy_base_prismatic_y_joint": 0.0,
                "dummy_base_revolute_z_joint": 0.0,
                "panda_joint1": 0.0, "panda_joint2": -0.569, "panda_joint3": 0.0,
                "panda_joint4": -2.810, "panda_joint5": 0.0, "panda_joint6": 2.0,
                "panda_joint7": 0.741, "panda_finger_joint.*": 0.035,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "base": ImplicitActuatorCfg(
                joint_names_expr=["dummy_base_.*"],
                effort_limit_sim=1000.0, stiffness=0.0, damping=1e5,
            ),
        },
    )

    # ── Walls (10m x 10m room, 2.5m tall) ──
    wall_north = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Wall_north",
        spawn=sim_utils.CuboidCfg(
            size=(10.3, 0.15, 2.5),
            rigid_props=_STATIC_RIGID_PROPS, mass_props=_STATIC_MASS_PROPS,
            collision_props=_STATIC_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.83, 0.78)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 5.0, 1.25)),
    )
    wall_south = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Wall_south",
        spawn=sim_utils.CuboidCfg(
            size=(10.3, 0.15, 2.5),
            rigid_props=_STATIC_RIGID_PROPS, mass_props=_STATIC_MASS_PROPS,
            collision_props=_STATIC_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.83, 0.78)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -5.0, 1.25)),
    )
    wall_east = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Wall_east",
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 10.0, 2.5),
            rigid_props=_STATIC_RIGID_PROPS, mass_props=_STATIC_MASS_PROPS,
            collision_props=_STATIC_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.83, 0.78)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5.0, 0.0, 1.25)),
    )
    wall_west = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Wall_west",
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 10.0, 2.5),
            rigid_props=_STATIC_RIGID_PROPS, mass_props=_STATIC_MASS_PROPS,
            collision_props=_STATIC_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.83, 0.78)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-5.0, 0.0, 1.25)),
    )

    # ── Furniture ──
    furniture_table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Furniture_table",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.6, 0.75),
            rigid_props=_STATIC_RIGID_PROPS, mass_props=_STATIC_MASS_PROPS,
            collision_props=_STATIC_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.35, 0.17)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, 1.0, 0.375)),
    )
    furniture_bookshelf = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Furniture_bookshelf",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 1.2, 1.8),
            rigid_props=_STATIC_RIGID_PROPS, mass_props=_STATIC_MASS_PROPS,
            collision_props=_STATIC_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.25, 0.12)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-3.0, 3.0, 0.9)),
    )
    furniture_chair = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Furniture_chair",
        spawn=sim_utils.CuboidCfg(
            size=(0.45, 0.45, 0.9),
            rigid_props=_STATIC_RIGID_PROPS, mass_props=_STATIC_MASS_PROPS,
            collision_props=_STATIC_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, -2.0, 0.45)),
    )
    furniture_desk = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Furniture_desk",
        spawn=sim_utils.CuboidCfg(
            size=(1.5, 0.7, 0.75),
            rigid_props=_STATIC_RIGID_PROPS, mass_props=_STATIC_MASS_PROPS,
            collision_props=_STATIC_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.5, 0.35)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.0, -1.0, 0.375)),
    )
    furniture_cabinet = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Furniture_cabinet",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.4, 1.4),
            rigid_props=_STATIC_RIGID_PROPS, mass_props=_STATIC_MASS_PROPS,
            collision_props=_STATIC_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.5, -3.5, 0.7)),
    )
    furniture_couch = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Furniture_couch",
        spawn=sim_utils.CuboidCfg(
            size=(1.8, 0.7, 0.6),
            rigid_props=_STATIC_RIGID_PROPS, mass_props=_STATIC_MASS_PROPS,
            collision_props=_STATIC_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.3, 0.55)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.0, 3.5, 0.3)),
    )
    furniture_bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Furniture_bin",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, 0.5),
            rigid_props=_STATIC_RIGID_PROPS, mass_props=_STATIC_MASS_PROPS,
            collision_props=_STATIC_COLLISION_PROPS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.25, 0.15)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.5, -1.5, 0.25)),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/dome_light",
        spawn=sim_utils.DomeLightCfg(color=(0.4, 0.4, 0.5), intensity=1000.0),
    )


def main():
    """Set up the scene with InteractiveScene and record a video."""

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0)
    sim = SimulationContext(sim_cfg)

    scene_cfg = PreviewSceneCfg(num_envs=NUM_ENVS, env_spacing=12.0)
    scene = InteractiveScene(scene_cfg)

    # Overhead camera (not cloned per env)
    camera_cfg = CameraCfg(
        prim_path="/World/OverheadCamera",
        update_period=0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1000.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )
    camera = Camera(camera_cfg)

    sim.reset()
    scene.reset()
    camera.reset()

    # Camera positioning via USD xform
    import omni.usd
    from pxr import UsdGeom, Gf

    stage = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath("/World/OverheadCamera")
    xform = UsdGeom.Xformable(cam_prim)

    env_origins = scene.env_origins.cpu().numpy()

    print("\n" + "=" * 60)
    print("  RECORDING ENVIRONMENT PREVIEW VIDEO")
    print(f"  {NUM_ENVS} indoor rooms, each with robot + walls + furniture")
    print("=" * 60)

    # Pan camera across environments, close up
    frames_per_env = 45  # 1.5 seconds per env at 30 fps
    total_frames = frames_per_env * NUM_ENVS

    with VideoRecorder("env_preview", fps=30) as recorder:
        for frame in range(total_frames):
            env_idx = frame // frames_per_env
            local_frame = frame % frames_per_env

            # Current and next env origins
            origin = env_origins[env_idx]
            ox, oy = float(origin[0]), float(origin[1])

            if env_idx < NUM_ENVS - 1:
                next_origin = env_origins[env_idx + 1]
                nox, noy = float(next_origin[0]), float(next_origin[1])
            else:
                nox, noy = ox, oy

            # Smooth transition: hold 70%, glide 30%
            t = local_frame / frames_per_env
            blend = 0.0 if t < 0.7 else (t - 0.7) / 0.3

            cam_target_x = ox + (nox - ox) * blend
            cam_target_y = oy + (noy - oy) * blend

            # Camera: offset back and up, looking down into the room
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(cam_target_x, cam_target_y - 7.0, 10.0))
            xform.AddRotateXYZOp().Set(Gf.Vec3f(55.0, 0.0, 0.0))

            sim.step(render=True)
            camera.update(dt=sim.get_physics_dt())
            scene.update(dt=sim.get_physics_dt())

            rgb_data = camera.data.output["rgb"]
            if rgb_data is not None and rgb_data.numel() > 0:
                img = rgb_data[0].cpu().numpy()
                if img.shape[-1] == 4:
                    recorder.capture_rgba(img)
                elif img.shape[-1] == 3:
                    rgba = np.concatenate(
                        [img, np.full((*img.shape[:2], 1), 255, dtype=img.dtype)], axis=-1
                    )
                    recorder.capture_rgba(rgba)

    print("\n  Video saved to: ~/isaac-sim-msoe/videos/env_preview.mp4")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
