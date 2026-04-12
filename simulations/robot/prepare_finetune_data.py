#!/usr/bin/env python3
"""
prepare_finetune_data.py — Convert door_labels.npz → JSONL for Qwen2-VL fine-tuning.

Reads data/door_labels.npz (produced by nova_carter_door_collection.py) and writes:
  data/finetune_images/frame_XXXXX.jpg  — individual JPEG frames
  data/train.jsonl                       — 70% training split  (1260 frames)
  data/val.jsonl                         — 15% validation split ( 270 frames)
  data/test.jsonl                        — 15% held-out test set ( 270 frames)

The test split is held out entirely and must NOT be used during training or
hyperparameter selection. Run eval_baseline.py against test.jsonl only for
final reported accuracy numbers.

Label thresholding (from VLM_fine-tuning.md):
  angle_deg = label * 90.0
  angle_deg ≤ 5° → [CLOSED]
  angle_deg  > 5° → [OPEN]

JSONL format (one JSON object per line):
  {
    "conversations": [
      {"role": "user",      "content": "<vision tokens>Is this door open or closed?"},
      {"role": "assistant", "content": "Door Status: [OPEN]."}
    ],
    "image": "data/finetune_images/frame_00042.jpg"
  }

Usage:
  python3 simulations/robot/prepare_finetune_data.py
  python3 simulations/robot/prepare_finetune_data.py --npz data/door_labels.npz --out data/
"""

import argparse
import json
import os
import random
import sys

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLOSED_THRESHOLD_DEG = 5.0   # angles ≤ this → CLOSED  (matches VLM_fine-tuning.md)
TRAIN_FRACTION       = 0.70  # 70/15/15 train/val/test split
VAL_FRACTION         = 0.15  # test gets the remainder (also 0.15)
JPEG_QUALITY         = 92    # JPEG compression quality for saved frames
VISION_TOKEN_STR     = "<|vision_start|><|image_pad|><|vision_end|>"

USER_PROMPT    = f"{VISION_TOKEN_STR}Is this door open or closed?"
CLOSED_ANSWER  = "Door Status: [CLOSED]."
OPEN_ANSWER    = "Door Status: [OPEN]."

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=False)
    frames = data["frames"]    # (N, H, W, 3) uint8
    # Support both key names: door_labels_static_multi_camera.npz uses "angles"
    # (raw degrees), older door_labels.npz uses "labels" (normalized [0,1]).
    if "angles" in data:
        # angles are already in degrees; normalize to [0,1] range over 90°
        labels = (data["angles"] / 90.0).astype(np.float32)
    else:
        labels = data["labels"]    # (N,) float32 in [0, 1]
    return frames, labels


def label_to_class(label: float) -> str:
    """Convert normalized label [0,1] to OPEN/CLOSED string."""
    return "CLOSED" if label * 90.0 <= CLOSED_THRESHOLD_DEG else "OPEN"


def make_conversation(answer: str) -> list:
    return [
        {"role": "user",      "content": USER_PROMPT},
        {"role": "assistant", "content": answer},
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="data/door_labels_static_multi_camera.npz")
    parser.add_argument("--out", default="data/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    npz_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", args.npz
    )
    out_dir  = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", args.out
    )
    img_dir  = os.path.join(out_dir, "finetune_images")
    os.makedirs(img_dir, exist_ok=True)

    if not os.path.isfile(npz_path):
        print(f"ERROR: {npz_path} not found. Run nova_carter_door_collection.py first.")
        sys.exit(1)

    print(f"Loading {npz_path} ...")
    frames, labels = load_npz(npz_path)
    N = len(frames)
    print(f"  {N} frames, label range [{labels.min():.3f}, {labels.max():.3f}]")

    # Build per-class index lists for stratified split
    closed_idx = [i for i, l in enumerate(labels) if label_to_class(l) == "CLOSED"]
    open_idx   = [i for i, l in enumerate(labels) if label_to_class(l) == "OPEN"]

    print(f"\nClass distribution before split:")
    print(f"  CLOSED: {len(closed_idx):5d} ({100*len(closed_idx)/N:.1f}%)")
    print(f"  OPEN:   {len(open_idx):5d} ({100*len(open_idx)/N:.1f}%)")

    if len(closed_idx) == 0:
        print("\nERROR: No CLOSED frames found.")
        print("  Re-run data collection with stratified episode sampling.")
        sys.exit(1)
    if len(open_idx) == 0:
        print("\nERROR: No OPEN frames found.")
        sys.exit(1)

    # Stratified 70/15/15 split
    random.shuffle(closed_idx)
    random.shuffle(open_idx)

    def split3(idx_list):
        n = len(idx_list)
        train_cut = int(n * TRAIN_FRACTION)
        val_cut   = int(n * (TRAIN_FRACTION + VAL_FRACTION))
        return idx_list[:train_cut], idx_list[train_cut:val_cut], idx_list[val_cut:]

    train_closed, val_closed, test_closed = split3(closed_idx)
    train_open,   val_open,   test_open   = split3(open_idx)

    train_idx = train_closed + train_open
    val_idx   = val_closed   + val_open
    test_idx  = test_closed  + test_open
    random.shuffle(train_idx)
    random.shuffle(val_idx)
    random.shuffle(test_idx)

    print(f"\nSplit (70/15/15 stratified):")
    print(f"  Train — CLOSED: {len(train_closed)}, OPEN: {len(train_open)}, total: {len(train_idx)}")
    print(f"  Val   — CLOSED: {len(val_closed)},   OPEN: {len(val_open)},   total: {len(val_idx)}")
    print(f"  Test  — CLOSED: {len(test_closed)},  OPEN: {len(test_open)},  total: {len(test_idx)}  [HELD OUT]")

    # Save frames as JPEG and build JSONL records
    def process_split(indices, jsonl_path, split_name):
        records = []
        for i, idx in enumerate(indices):
            img_filename = f"frame_{idx:05d}.jpg"
            img_path     = os.path.join(img_dir, img_filename)

            if not os.path.isfile(img_path):
                img = Image.fromarray(frames[idx])
                img.save(img_path, "JPEG", quality=JPEG_QUALITY)

            cls    = label_to_class(labels[idx])
            answer = CLOSED_ANSWER if cls == "CLOSED" else OPEN_ANSWER
            records.append({
                "conversations": make_conversation(answer),
                "image":         os.path.relpath(img_path, out_dir),
            })

            if (i + 1) % 500 == 0:
                print(f"  [{split_name}] processed {i+1}/{len(indices)} ...")

        with open(jsonl_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())   # force NFS write-through before finetune job reads
        print(f"  Saved {len(records)} records → {jsonl_path}")

    print(f"\nSaving frames to {img_dir} ...")
    process_split(train_idx, os.path.join(out_dir, "train.jsonl"), "train")
    process_split(val_idx,   os.path.join(out_dir, "val.jsonl"),   "val")
    process_split(test_idx,  os.path.join(out_dir, "test.jsonl"),  "test")

    print("\nDone.")
    print(f"  train.jsonl : {len(train_idx)} records  (70%)")
    print(f"  val.jsonl   : {len(val_idx)} records  (15%)")
    print(f"  test.jsonl  : {len(test_idx)} records  (15%) — held out for final eval")
    print(f"  Images      : {img_dir}")


if __name__ == "__main__":
    main()
