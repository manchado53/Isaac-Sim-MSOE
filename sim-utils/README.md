# Simulation Utilities

Python utilities for Isaac Sim simulations on Rosie.

## VideoRecorder

The `VideoRecorder` class captures screenshots during simulation and automatically converts them to MP4 videos for debugging and visualization.

### Quick Start

```python
from utils.video_capture import VideoRecorder

# Use as a context manager
with VideoRecorder("my_simulation") as recorder:
    for i in range(300):  # 10 seconds at 30 FPS
        world.step(render=True)  # Must enable rendering
        recorder.capture()

# Video automatically saved to videos/my_simulation.mp4
```

### Features

- **Automatic video creation**: Captures frames and creates MP4 video at 30 FPS
- **Temporary storage**: Screenshots are automatically deleted after video creation
- **Progress tracking**: Prints progress updates during capture and video creation
- **Error handling**: Preserves screenshots if video creation fails

### Configuration Options

```python
VideoRecorder(
    video_name="simulation",      # Name for output video (without extension)
    fps=30,                        # Frames per second (default: 30)
    output_dir=None,               # Custom output directory (default: ../videos)
    keep_screenshots=False,        # Keep individual frames after video creation
    viewport_name="Viewport"       # Isaac Sim viewport to capture
)
```

### Example

See [bouncing_balls_video.py](../simulations/bouncing_balls_video.py) for a complete example.

### Requirements

- **ffmpeg**: Must be installed on the system (available on Rosie)
- **Rendering enabled**: Must call `world.step(render=True)` to update viewport
- **Headless mode**: Works in headless Isaac Sim mode (no GUI required)

### Output

Videos are saved to `~/isaacsim/physics-sim/videos/` by default with the `.mp4` extension.
