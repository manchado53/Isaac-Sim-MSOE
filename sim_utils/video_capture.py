"""
Video capture utility for Isaac Sim simulations.

This module provides a VideoRecorder class that captures screenshots during
simulation and automatically converts them to a video file for debugging.

Usage:
    from sim_utils.video_capture import VideoRecorder

    with VideoRecorder("my_simulation") as recorder:
        for i in range(100):
            world.step(render=True)
            recorder.capture()

    # Video automatically saved to videos/my_simulation.mp4
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


class VideoRecorder:
    """
    Context manager for capturing simulation screenshots and creating videos.

    Captures frames during simulation and uses ffmpeg to create an MP4 video
    at 30 FPS. Screenshots are automatically deleted after video creation.
    """

    def __init__(
        self,
        video_name: str,
        fps: int = 30,
        output_dir: Optional[str] = None,
        keep_screenshots: bool = False,
        viewport_name: str = "Viewport"
    ):
        """
        Initialize the video recorder.

        Args:
            video_name: Name for the output video (without extension)
            fps: Frames per second for the output video (default: 30)
            output_dir: Directory to save videos (default: ../videos relative to this file)
            keep_screenshots: If True, don't delete screenshots after video creation
            viewport_name: Name of the Isaac Sim viewport to capture (default: "Viewport")
        """
        self.video_name = video_name
        self.fps = fps
        self.keep_screenshots = keep_screenshots
        self.viewport_name = viewport_name

        # Set up output directory
        if output_dir is None:
            # Default to ../videos relative to this file
            utils_dir = Path(__file__).parent
            project_dir = utils_dir.parent
            self.output_dir = project_dir / "videos"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create temporary directory for screenshots
        self.temp_dir = None
        self.frame_count = 0
        self.viewport_api = None

    def __enter__(self):
        """Enter context manager - set up screenshot directory."""
        self.temp_dir = tempfile.mkdtemp(prefix=f"isaac_sim_{self.video_name}_")
        self.frame_count = 0
        print(f"[VideoRecorder] Recording to: {self.output_dir / self.video_name}.mp4")
        print(f"[VideoRecorder] Temporary screenshots: {self.temp_dir}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - create video and cleanup."""
        if exc_type is not None:
            print(f"[VideoRecorder] Error during recording: {exc_val}")
            print(f"[VideoRecorder] Screenshots saved at: {self.temp_dir}")
            return False

        # Create video from screenshots
        if self.frame_count > 0:
            self._create_video()
        else:
            print("[VideoRecorder] No frames captured, skipping video creation")

        # Cleanup temporary directory
        if not self.keep_screenshots and self.temp_dir:
            shutil.rmtree(self.temp_dir)
            print(f"[VideoRecorder] Cleaned up temporary screenshots")
        elif self.keep_screenshots:
            print(f"[VideoRecorder] Screenshots preserved at: {self.temp_dir}")

        return False

    def capture(self):
        """
        Capture a screenshot from the Isaac Sim viewport.

        Call this method after each world.step() to capture a frame.
        """
        try:
            # Import Isaac Sim viewport utilities
            from omni.kit.viewport.utility import get_viewport_from_window_name, capture_viewport_to_file

            # Get viewport API on first capture
            if self.viewport_api is None:
                self.viewport_api = get_viewport_from_window_name(self.viewport_name)
                if self.viewport_api is None:
                    print(f"[VideoRecorder] Warning: Could not find viewport '{self.viewport_name}'")
                    return

            # Generate filename with zero-padded frame number
            filename = f"frame_{self.frame_count:06d}.png"
            filepath = os.path.join(self.temp_dir, filename)

            # Capture the viewport to file
            capture_viewport_to_file(self.viewport_api, filepath)

            self.frame_count += 1

            # Print progress every 30 frames (1 second at 30fps)
            if self.frame_count % 30 == 0:
                print(f"[VideoRecorder] Captured {self.frame_count} frames...")

        except ImportError as e:
            print(f"[VideoRecorder] Error importing Isaac Sim modules: {e}")
            print(f"[VideoRecorder] Make sure this is running in Isaac Sim Python environment")
        except Exception as e:
            print(f"[VideoRecorder] Error capturing frame {self.frame_count}: {e}")

    def _get_ffmpeg_path(self):
        """Find ffmpeg binary - prefer imageio-ffmpeg bundled binary, fall back to system."""
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass

        # Fall back to system ffmpeg
        if shutil.which('ffmpeg'):
            return 'ffmpeg'

        return None

    def _create_video(self):
        """Create MP4 video from captured screenshots using ffmpeg."""
        output_path = self.output_dir / f"{self.video_name}.mp4"

        print(f"[VideoRecorder] Creating video from {self.frame_count} frames...")

        ffmpeg_path = self._get_ffmpeg_path()
        if ffmpeg_path is None:
            print(f"[VideoRecorder] Error: ffmpeg not found (install imageio-ffmpeg or system ffmpeg)")
            print(f"[VideoRecorder] Screenshots preserved at: {self.temp_dir}")
            self.keep_screenshots = True
            return

        print(f"[VideoRecorder] Using ffmpeg: {ffmpeg_path}")

        cmd = [
            ffmpeg_path,
            '-framerate', str(self.fps),
            '-i', os.path.join(self.temp_dir, 'frame_%06d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-y',
            str(output_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )
            print(f"[VideoRecorder] ✓ Video saved: {output_path}")
            print(f"[VideoRecorder] Duration: {self.frame_count / self.fps:.2f}s at {self.fps} FPS")

        except subprocess.CalledProcessError as e:
            print(f"[VideoRecorder] Error creating video with ffmpeg:")
            print(f"[VideoRecorder] {e.stderr}")
            print(f"[VideoRecorder] Screenshots preserved at: {self.temp_dir}")
            self.keep_screenshots = True
        except FileNotFoundError:
            print(f"[VideoRecorder] Error: ffmpeg binary not executable at {ffmpeg_path}")
            print(f"[VideoRecorder] Screenshots preserved at: {self.temp_dir}")
            self.keep_screenshots = True
