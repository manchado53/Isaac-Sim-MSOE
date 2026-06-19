#!/usr/bin/env python3
"""Convert a directory of images into a 30fps MP4 video."""

import argparse
import glob
import os
import sys
import cv2


def main():
    parser = argparse.ArgumentParser(description="Convert image frames to a video.")
    parser.add_argument("image_dir", help="Path to directory containing images")
    parser.add_argument(
        "-o", "--output",
        help="Output video path (default: <image_dir>/output.mp4)",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Frames per second (default: 30)",
    )
    args = parser.parse_args()

    image_dir = os.path.abspath(args.image_dir)
    if not os.path.isdir(image_dir):
        print(f"Error: '{image_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(image_dir, ext)))
        files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    files = sorted(set(files))

    if not files:
        print(f"Error: no images found in '{image_dir}'.", file=sys.stderr)
        sys.exit(1)

    out_path = args.output or os.path.join(image_dir, "output.mp4")

    first = cv2.imread(files[0])
    if first is None:
        print(f"Error: could not read '{files[0]}'.", file=sys.stderr)
        sys.exit(1)
    h, w = first.shape[:2]

    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h)
    )

    print(f"Writing {len(files)} frames at {args.fps} fps → {out_path}")
    for i, path in enumerate(files):
        frame = cv2.imread(path)
        if frame is None:
            print(f"  Warning: skipping unreadable file '{path}'")
            continue
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(files)}")

    writer.release()
    size_mb = os.path.getsize(out_path) / 1e6
    duration = len(files) / args.fps
    print(f"Done — {duration:.1f}s, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
