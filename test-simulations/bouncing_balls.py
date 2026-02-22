#!/usr/bin/env python3
"""
Isaac Sim example with video recording.

Creates multiple bouncing balls and records the simulation to a video file.
This demonstrates the VideoRecorder utility for debugging simulations.
"""

import os
import sys

# Accept EULA if not already set
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

# Launch Isaac Sim headless
simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicSphere
import numpy as np

# Add project root to path for sim_utils import
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, '..'))
sys.path.insert(0, _project_dir)
from sim_utils.video_capture import VideoRecorder


def main():
    print("=" * 60)
    print("Isaac Sim - Bouncing Balls with Video Recording")
    print("=" * 60)

    # Create a world
    world = World()
    print("\n✓ World created")

    # Add a ground plane
    world.scene.add_default_ground_plane()
    print("✓ Ground plane added")

    # Create multiple bouncing balls at different heights and positions
    balls = []
    colors = [
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue
        [1.0, 1.0, 0.0],  # Yellow
        [1.0, 0.0, 1.0],  # Magenta
    ]

    print("\n✓ Creating bouncing balls...")
    for i in range(5):
        ball = world.scene.add(
            DynamicSphere(
                prim_path=f"/World/Ball_{i}",
                name=f"ball_{i}",
                position=np.array([
                    -2.0 + i * 1.0,  # Spread along X axis
                    0.0,
                    3.0 + i * 0.5    # Different heights
                ]),
                radius=0.15,
                color=np.array(colors[i])
            )
        )
        balls.append(ball)
        print(f"  - Ball {i}: position=({-2.0 + i * 1.0:.1f}, 0.0, {3.0 + i * 0.5:.1f})")

    # Reset the world
    world.reset()
    print("\n✓ World reset - simulation ready")

    # Run simulation with video recording
    num_frames = 300  # 10 seconds at 30 FPS
    print(f"\n✓ Running simulation for {num_frames} steps with video recording...")

    with VideoRecorder("bouncing_balls") as recorder:
        for i in range(num_frames):
            # Step the simulation with rendering enabled
            world.step(render=True)

            # Capture the frame
            recorder.capture()

            # Print progress every second (30 frames)
            if i % 30 == 0:
                elapsed_sec = i / 30
                print(f"  Simulated {elapsed_sec:.1f}s...")

    print("\n" + "=" * 60)
    print("✓ Simulation completed successfully!")
    print("✓ Video saved to: videos/bouncing_balls.mp4")
    print("=" * 60)

    # Cleanup
    simulation_app.close()


if __name__ == "__main__":
    main()
