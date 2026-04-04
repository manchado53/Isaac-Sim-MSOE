"""
Video capture utility for Isaac Sim simulations.

This module provides a VideoRecorder class with two capture modes:

1. Viewport capture (capture()) — captures from the Isaac Sim GUI viewport,
   saves PNGs to a temp dir, then assembles into MP4. Best for simple scenes.

2. Camera sensor capture (capture_rgba()) — accepts raw RGBA numpy arrays
   from an isaacsim.sensors.camera.Camera and pipes directly to ffmpeg.
   Best for follow cameras or any positioned Camera prim.

Usage (viewport):
    with VideoRecorder("my_sim") as recorder:
        for i in range(300):
            world.step(render=True)
            recorder.capture()

Usage (camera sensor):
    with VideoRecorder("my_sim", fps=15) as recorder:
        for i in range(300):
            world.step(render=True)
            rgba = camera.get_rgba()
            recorder.capture_rgba(rgba)
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


class VideoRecorder:
    """
    Context manager for recording Isaac Sim simulations to MP4 video.

    Supports two capture modes:
    - capture()      : captures from the Isaac Sim viewport (PNG-based)
    - capture_rgba() : accepts RGBA numpy arrays from a Camera sensor (ffmpeg pipe)
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
        Args:
            video_name: Name for the output video (without extension)
            fps: Frames per second for the output video (default: 30)
            output_dir: Directory to save videos (default: ../videos relative to this file)
            keep_screenshots: If True, don't delete PNGs after video creation (viewport mode only)
            viewport_name: Isaac Sim viewport window name (default: "Viewport")
        """
        self.video_name = video_name
        self.fps = fps
        self.keep_screenshots = keep_screenshots
        self.viewport_name = viewport_name

        if output_dir is None:
            utils_dir = Path(__file__).parent
            project_dir = utils_dir.parent
            self.output_dir = project_dir / "videos"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Support subdirectory paths in video_name (e.g. "validation/validate_cubes_cam_G")
        (self.output_dir / self.video_name).parent.mkdir(parents=True, exist_ok=True)

        self.temp_dir = None
        self.frame_count = 0
        self.viewport_api = None
        self._ffmpeg_proc = None  # used in capture_rgba mode

    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp(prefix=f"isaac_sim_{self.video_name}_")
        self.frame_count = 0
        self._ffmpeg_proc = None
        print(f"[VideoRecorder] Output: {self.output_dir / self.video_name}.mp4")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._ffmpeg_proc is not None:
            # rgba pipe mode — finalize the ffmpeg process
            self._ffmpeg_proc.stdin.close()
            self._ffmpeg_proc.wait(timeout=60)
            output_path = self.output_dir / f"{self.video_name}.mp4"
            if self._ffmpeg_proc.returncode == 0:
                print(f"[VideoRecorder] ✓ Video saved: {output_path} ({self.frame_count} frames)")
            else:
                print(f"[VideoRecorder] ffmpeg failed (exit code {self._ffmpeg_proc.returncode})")
        else:
            # viewport/PNG mode — assemble from captured PNGs
            if exc_type is not None:
                print(f"[VideoRecorder] Error during recording: {exc_val}")
                print(f"[VideoRecorder] Screenshots saved at: {self.temp_dir}")
                return False

            if self.frame_count > 0:
                self._create_video()
            else:
                print("[VideoRecorder] No frames captured, skipping video creation")

            if not self.keep_screenshots and self.temp_dir:
                shutil.rmtree(self.temp_dir)
            elif self.keep_screenshots:
                print(f"[VideoRecorder] Screenshots preserved at: {self.temp_dir}")

        return False

    def capture(self):
        """
        Capture a frame from the Isaac Sim viewport.
        Call after each world.step(render=True).
        """
        try:
            from omni.kit.viewport.utility import get_viewport_from_window_name, capture_viewport_to_file

            if self.viewport_api is None:
                self.viewport_api = get_viewport_from_window_name(self.viewport_name)
                if self.viewport_api is None:
                    print(f"[VideoRecorder] Warning: Could not find viewport '{self.viewport_name}'")
                    return

            filename = f"frame_{self.frame_count:06d}.png"
            filepath = os.path.join(self.temp_dir, filename)
            capture_viewport_to_file(self.viewport_api, filepath)

            self.frame_count += 1
            if self.frame_count % 30 == 0:
                print(f"[VideoRecorder] Captured {self.frame_count} frames...")

        except ImportError as e:
            print(f"[VideoRecorder] Error importing Isaac Sim modules: {e}")
        except Exception as e:
            print(f"[VideoRecorder] Error capturing frame {self.frame_count}: {e}")

    def capture_rgba(self, rgba):
        """
        Capture a frame from a Camera sensor RGBA numpy array.

        Call after world.step(render=True) with the result of camera.get_rgba().
        Pipes frames directly to ffmpeg — no temp PNG files.

        Args:
            rgba: numpy array of shape (H, W, 4) from camera.get_rgba()
        """
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 3:
            return

        if self._ffmpeg_proc is None:
            self._start_ffmpeg_pipe(width=rgba.shape[1], height=rgba.shape[0])

        try:
            self._ffmpeg_proc.stdin.write(rgba[:, :, :3].tobytes())
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                print(f"[VideoRecorder] Captured {self.frame_count} frames...")
        except BrokenPipeError:
            print(f"[VideoRecorder] ffmpeg pipe closed unexpectedly")

    def _start_ffmpeg_pipe(self, width: int, height: int):
        """Start an ffmpeg process accepting raw RGB frames on stdin."""
        output_path = self.output_dir / f"{self.video_name}.mp4"
        ffmpeg_path = self._get_ffmpeg_path()
        if ffmpeg_path is None:
            raise RuntimeError("[VideoRecorder] ffmpeg not found. Install it at ~/bin/ffmpeg or system-wide.")

        cmd = [
            ffmpeg_path, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}", "-pix_fmt", "rgb24", "-r", str(self.fps),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        self._ffmpeg_proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f"[VideoRecorder] Piping {width}x{height} @ {self.fps}fps → {output_path}")

    def _get_ffmpeg_path(self):
        """Find ffmpeg — prefers imageio-ffmpeg, then ~/bin/ffmpeg, then system ffmpeg."""
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass

        local_ffmpeg = os.path.expanduser("~/bin/ffmpeg")
        if os.path.isfile(local_ffmpeg) and os.access(local_ffmpeg, os.X_OK):
            return local_ffmpeg

        if shutil.which("ffmpeg"):
            return "ffmpeg"

        return None

    def _create_video(self):
        """Assemble MP4 from captured PNG screenshots (viewport mode)."""
        output_path = self.output_dir / f"{self.video_name}.mp4"
        print(f"[VideoRecorder] Creating video from {self.frame_count} frames...")

        ffmpeg_path = self._get_ffmpeg_path()
        if ffmpeg_path is None:
            print(f"[VideoRecorder] Error: ffmpeg not found")
            print(f"[VideoRecorder] Screenshots preserved at: {self.temp_dir}")
            self.keep_screenshots = True
            return

        cmd = [
            ffmpeg_path,
            "-framerate", str(self.fps),
            "-i", os.path.join(self.temp_dir, "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-y",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
            print(f"[VideoRecorder] ✓ Video saved: {output_path}")
            print(f"[VideoRecorder] Duration: {self.frame_count / self.fps:.2f}s at {self.fps} FPS")
        except subprocess.CalledProcessError as e:
            print(f"[VideoRecorder] ffmpeg error: {e.stderr}")
            print(f"[VideoRecorder] Screenshots preserved at: {self.temp_dir}")
            self.keep_screenshots = True
        except FileNotFoundError:
            print(f"[VideoRecorder] ffmpeg not executable at {ffmpeg_path}")
            self.keep_screenshots = True
