#!/usr/bin/env python3
"""
prepare_holdout_data.py — extract holdout NPZ shards into JPEG + JSONL for
held-out-door generalization eval.

Reads:
  data/door_production_v4_holdout_Doors_*.npz   (doors NOT in the training set)

Writes:
  data/finetune_images_v4_holdout/<door_id>_<idx>.jpg
  data/holdout_v4.jsonl   (every frame, labeled, in eval format)

Differs from prepare_finetune_data.py in two ways:
  1. No train/val split — the entire shard is the eval set.
  2. Default glob targets the holdout role only.

Usage (compute node — NFS write-through matters):
  python3 simulations/robot/prepare_holdout_data.py
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

CLOSED_THRESHOLD_DEG = 5.0
JPEG_QUALITY         = 92
VISION_TOKEN_STR     = "<|vision_start|><|image_pad|><|vision_end|>"

USER_PROMPT   = f"{VISION_TOKEN_STR}Is this door open or closed?"
CLOSED_ANSWER = "Door Status: [CLOSED]."
OPEN_ANSWER   = "Door Status: [OPEN]."


def make_conversation(answer: str) -> list:
    return [
        {"role": "user",      "content": USER_PROMPT},
        {"role": "assistant", "content": answer},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-glob",
                        default="data/door_production_v4_holdout_Doors_*.npz")
    parser.add_argument("--out", default="data/")
    parser.add_argument("--img-subdir", default="finetune_images_v4_holdout")
    parser.add_argument("--jsonl-name", default="holdout_v4.jsonl")
    args = parser.parse_args()

    project_dir = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

    shard_glob_abs = os.path.join(project_dir, args.shard_glob)
    out_dir        = os.path.join(project_dir, args.out)
    img_dir        = os.path.join(out_dir, args.img_subdir)
    jsonl_path     = os.path.join(out_dir, args.jsonl_name)
    os.makedirs(img_dir, exist_ok=True)

    shard_paths = sorted(glob.glob(shard_glob_abs))
    if not shard_paths:
        print(f"ERROR: no holdout shards matched {shard_glob_abs}")
        sys.exit(1)

    print(f"Found {len(shard_paths)} holdout shard(s):")
    for sp in shard_paths:
        print(f"  {sp}")
    print()

    records   = []
    n_closed  = 0
    n_open    = 0

    for sp in shard_paths:
        print(f"Streaming {os.path.basename(sp)} ...", flush=True)
        with np.load(sp, allow_pickle=False) as data:
            frames  = data["frames"]
            angles  = data["angles"].astype(np.float32)
            door_id = str(data["door_id"])
            n       = len(frames)

            print(f"  door_id={door_id}  frames={n}  "
                  f"angle range=[{angles.min():.2f}, {angles.max():.2f}]°", flush=True)

            for i in range(n):
                img_name = f"{door_id}_{i:05d}.jpg"
                img_path = os.path.join(img_dir, img_name)
                if not os.path.isfile(img_path):
                    Image.fromarray(frames[i]).save(
                        img_path, "JPEG", quality=JPEG_QUALITY)

                label  = "CLOSED" if angles[i] <= CLOSED_THRESHOLD_DEG else "OPEN"
                answer = CLOSED_ANSWER if label == "CLOSED" else OPEN_ANSWER
                records.append({
                    "conversations": make_conversation(answer),
                    "image":         os.path.relpath(img_path, out_dir),
                })
                if label == "CLOSED":
                    n_closed += 1
                else:
                    n_open += 1

                if (i + 1) % 2000 == 0:
                    print(f"    {i+1}/{n} ...", flush=True)

            print(f"    wrote {n} JPEGs for door={door_id}", flush=True)
        del frames

    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())

    total = n_closed + n_open
    print()
    print(f"Done. Wrote {total} records → {jsonl_path}")
    print(f"  CLOSED: {n_closed} ({100*n_closed/total:.1f}%)")
    print(f"  OPEN:   {n_open} ({100*n_open/total:.1f}%)")
    print(f"  Images: {img_dir}")


if __name__ == "__main__":
    main()
