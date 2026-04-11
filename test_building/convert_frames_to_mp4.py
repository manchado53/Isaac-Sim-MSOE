#!/usr/bin/env python3
"""
Convert floor_1_frames.npy → floor_1_scan_video.mp4.
No Isaac Sim needed — pure numpy + imageio.

Run with:
  sbatch submit_isaac_job.sh test_building/convert_frames_to_mp4.py
"""
import os
import sys
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "presentation_nav_visuals")
NPY_PATH    = os.path.join(OUTPUT_DIR, "floor_1_frames.npy")
MP4_PATH    = os.path.join(OUTPUT_DIR, "floor_1_scan_video.mp4")
VIDEO_FPS   = 30

print("Loading frames...")
sys.stdout.flush()
frames = np.load(NPY_PATH)          # shape: (N, H, W, 3)
print(f"  {frames.shape[0]} frames, raw size {frames.shape[2]}x{frames.shape[1]} px")
sys.stdout.flush()

# libx264 requires even dimensions — trim 1px if odd
h = frames.shape[1] & ~1   # round down to even
w = frames.shape[2] & ~1
if h != frames.shape[1] or w != frames.shape[2]:
    frames = frames[:, :h, :w, :]
    print(f"  Trimmed to even dimensions: {w}x{h} px")

duration = frames.shape[0] / VIDEO_FPS
print(f"  {frames.shape[0]} frames @ {VIDEO_FPS} fps = {duration:.1f}s")
print(f"\nWriting MP4 → {MP4_PATH}")
sys.stdout.flush()

import imageio
writer = imageio.get_writer(
    MP4_PATH,
    fps=VIDEO_FPS,
    codec="libx264",
    quality=9,
    macro_block_size=None,
    output_params=["-pix_fmt", "yuv420p"],
)
for i, frame in enumerate(frames):
    writer.append_data(frame)
    if i % 50 == 0:
        print(f"  {i}/{len(frames)} frames written...")
        sys.stdout.flush()
writer.close()

size_mb = os.path.getsize(MP4_PATH) / (1024 * 1024)
print(f"\nDone: {MP4_PATH}  ({size_mb:.1f} MB)")
sys.stdout.flush()
