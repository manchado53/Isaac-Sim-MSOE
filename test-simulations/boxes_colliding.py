#!/usr/bin/env python3
"""
Isaac Sim - Two boxes of different masses colliding.

A heavy blue box slides toward a lighter orange box, demonstrating
momentum transfer on collision.
"""

import os
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..'))
sys.path.insert(0, _project_dir)
from sim_utils.video_capture import VideoRecorder


def main():
    print("=" * 60)
    print("Isaac Sim - Boxes Colliding")
    print("=" * 60)

    world = World()
    print("\n✓ World created")

    world.scene.add_default_ground_plane()
    print("✓ Ground plane added")

    # Heavy box (blue) - starts on the left, will be launched rightward
    heavy_box = world.scene.add(
        DynamicCuboid(
            prim_path="/World/HeavyBox",
            name="heavy_box",
            position=np.array([-2.0, 0.0, 0.25]),
            size=0.5,
            color=np.array([0.2, 0.4, 1.0]),
            mass=10.0,
        )
    )
    print("✓ Heavy box (10 kg, blue) placed at x=-2.0")

    # Light box (orange) - starts on the right, stationary
    light_box = world.scene.add(
        DynamicCuboid(
            prim_path="/World/LightBox",
            name="light_box",
            position=np.array([0.5, 0.0, 0.25]),
            size=0.5,
            color=np.array([1.0, 0.5, 0.0]),
            mass=2.0,
        )
    )
    print("✓ Light box (2 kg, orange) placed at x=0.5")

    world.reset()
    print("\n✓ World reset")

    # Apply initial velocity to the heavy box so it slides into the light box
    heavy_box.set_linear_velocity(np.array([10.0, 0.0, 0.0]))
    print("✓ Heavy box given velocity 10.0 m/s in +X direction")

    num_frames = 300
    print(f"\n✓ Running simulation for {num_frames} steps with video recording...")

    with VideoRecorder("boxes_colliding") as recorder:
        for i in range(num_frames):
            world.step(render=True)
            recorder.capture()

            if i % 30 == 0:
                h_pos, _ = heavy_box.get_world_pose()
                l_pos, _ = light_box.get_world_pose()
                h_vel = heavy_box.get_linear_velocity()
                l_vel = light_box.get_linear_velocity()
                print(
                    f"  t={i/30:.1f}s  "
                    f"Heavy: x={h_pos[0]:+.2f} v={h_vel[0]:+.2f}  "
                    f"Light: x={l_pos[0]:+.2f} v={l_vel[0]:+.2f}"
                )

    print("\n" + "=" * 60)
    print("✓ Simulation completed!")
    print("✓ Video saved to: videos/boxes_colliding.mp4")
    print("=" * 60)

    simulation_app.close()


if __name__ == "__main__":
    main()
