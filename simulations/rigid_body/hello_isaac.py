#!/usr/bin/env python3
"""
Simple Isaac Sim test script
Tests that Isaac Sim is working by creating a basic scene with physics
"""

import os
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
import numpy as np

# Add project root to path for sim_utils import
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..'))
sys.path.insert(0, _project_dir)
from sim_utils.video_capture import VideoRecorder

def main():
    print("=" * 60)
    print("Isaac Sim Test - Hello Isaac!")
    print("=" * 60)

    world = World()
    print("\n✓ World created")

    world.scene.add_default_ground_plane()
    print("✓ Ground plane added")

    cube = world.scene.add(
        DynamicCuboid(
            prim_path="/World/Cube",
            name="falling_cube",
            position=np.array([0, 0, 2.0]),
            size=0.5,
            color=np.array([0.0, 0.5, 1.0])
        )
    )
    print("✓ Dynamic cube added at height 2.0m")

    world.reset()
    print("\n✓ World reset - simulation ready")

    num_steps = 150
    print(f"\nRunning physics simulation for {num_steps} steps with video recording...")

    with VideoRecorder("falling_cube") as recorder:
        for i in range(num_steps):
            world.step(render=True)
            recorder.capture()

            if i % 30 == 0:
                position, _ = cube.get_world_pose()
                print(f"  Step {i:3d}: Cube position = {position}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("Isaac Sim is working correctly on this node.")
    print("Video saved to: output/videos/falling_cube.mp4")
    print("=" * 60)

    simulation_app.close()

if __name__ == "__main__":
    main()
