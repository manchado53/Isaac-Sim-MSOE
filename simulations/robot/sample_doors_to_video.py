#!/usr/bin/env python3
"""Sample 1000 random frames from data/finetune_images_v4, group per door, render to MP4."""

import importlib.util
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "video_capture", PROJECT_ROOT / "sim_utils" / "video_capture.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
VideoRecorder = _mod.VideoRecorder

IMAGE_DIR = PROJECT_ROOT / "data" / "finetune_images_v4"
SAMPLE_SIZE = 1000
SEED = 42
FPS = 10
VIDEO_NAME = "finetune_images_v4_sample1000"

FNAME_RE = re.compile(r"^(?P<door>Doors_\d+)_(?P<idx>\d+)\.jpg$", re.IGNORECASE)


def main():
    files = [p for p in IMAGE_DIR.iterdir() if p.suffix.lower() == ".jpg"]
    if len(files) < SAMPLE_SIZE:
        print(f"Only {len(files)} images available (< {SAMPLE_SIZE}). Using all.")
        sampled = files
    else:
        random.seed(SEED)
        sampled = random.sample(files, SAMPLE_SIZE)

    def sort_key(path: Path):
        m = FNAME_RE.match(path.name)
        if not m:
            return (path.name, 0)
        return (m.group("door"), int(m.group("idx")))

    sampled.sort(key=sort_key)

    counts = {}
    for p in sampled:
        door = sort_key(p)[0]
        counts[door] = counts.get(door, 0) + 1
    print(f"Sampled {len(sampled)} frames across {len(counts)} doors:")
    for door in sorted(counts):
        print(f"  {door}: {counts[door]}")

    with VideoRecorder(VIDEO_NAME, fps=FPS) as recorder:
        for i, path in enumerate(sampled):
            with Image.open(path) as im:
                rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
            recorder.capture_rgba(rgb)
            if (i + 1) % 100 == 0:
                print(f"  wrote {i + 1}/{len(sampled)}")


if __name__ == "__main__":
    main()
