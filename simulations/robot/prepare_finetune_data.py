#!/usr/bin/env python3
"""
prepare_finetune_data.py — multi-shard NPZ → JSONL for Qwen2-VL fine-tuning.

Reads all training-role production-collection shards matching
  data/door_production_v4_train_Doors_*.npz
and writes:
  data/finetune_images_v4/<door_id>_<idx>.jpg  — one JPEG per frame
  data/train_v4.jsonl                          — 85% train split
  data/val_v4.jsonl                            — 15% val split

The holdout-role shards (door_production_v4_holdout_*.npz) are excluded by
the default glob and consumed separately by the evaluation pipeline.

Streaming design
----------------
Loading all frames at once would cost tens of GB of RAM. Instead we stream
one shard at a time: write its JPEGs, record lightweight metadata (door_id,
angle_band, light_idx, angles, image paths), then free the shard before
loading the next. Peak RAM is capped by the largest shard.

Stratification
--------------
Stratified shuffle-split on the joint key (door_id, angle_band, light_idx)
so every door, angle band, and lighting preset is represented proportionally
in both train and val.

Label thresholding
------------------
  angle_deg ≤ 5° → [CLOSED]
  angle_deg  > 5° → [OPEN]

Usage
-----
  # on a compute node (NFS write-through matters):
  python3 simulations/robot/prepare_finetune_data.py
  python3 simulations/robot/prepare_finetune_data.py \
      --shard-glob "data/door_production_v4_train_Doors_*.npz" \
      --img-subdir finetune_images_v4 \
      --train-name train_v4.jsonl --val-name val_v4.jsonl
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLOSED_THRESHOLD_DEG = 5.0
VAL_FRACTION         = 0.15
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


# ---------------------------------------------------------------------------
# Pass 1 — stream shards, write JPEGs, record metadata
# ---------------------------------------------------------------------------

def stream_shards(shard_paths, img_dir, out_dir):
    meta = {k: [] for k in
            ("angles", "door_ids", "angle_band", "light_idx", "rel_img_paths")}

    for sp in shard_paths:
        print(f"  Streaming {os.path.basename(sp)} ...", flush=True)
        with np.load(sp, allow_pickle=False) as data:
            frames     = data["frames"]
            angles     = data["angles"].astype(np.float32)
            angle_band = data["angle_band"].astype(np.int32)
            light_idx  = data["light_idx"].astype(np.int32)
            door_id    = str(data["door_id"])
            n = len(frames)

            rel_paths = np.empty(n, dtype=object)
            for i in range(n):
                img_name = f"{door_id}_{i:05d}.jpg"
                img_path = os.path.join(img_dir, img_name)
                if not os.path.isfile(img_path):
                    Image.fromarray(frames[i]).save(
                        img_path, "JPEG", quality=JPEG_QUALITY)
                rel_paths[i] = os.path.relpath(img_path, out_dir)

                if (i + 1) % 2000 == 0:
                    print(f"    {i+1}/{n} ...", flush=True)

            meta["angles"].append(angles)
            meta["angle_band"].append(angle_band)
            meta["light_idx"].append(light_idx)
            meta["door_ids"].append(np.array([door_id] * n, dtype=object))
            meta["rel_img_paths"].append(rel_paths)
            print(f"    wrote {n} JPEGs for door={door_id}", flush=True)

        del frames

    return {k: np.concatenate(v) for k, v in meta.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-glob",
                        default="data/door_production_v4_train_Doors_*.npz",
                        help="Glob for production NPZ shards (defaults to train role only)")
    parser.add_argument("--out", default="data/")
    parser.add_argument("--img-subdir", default="finetune_images_v4",
                        help="Subdir under --out for JPEG frames")
    parser.add_argument("--train-name", default="train_v4.jsonl")
    parser.add_argument("--val-name",   default="val_v4.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_dir = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

    shard_glob_abs = os.path.join(project_dir, args.shard_glob)
    out_dir        = os.path.join(project_dir, args.out)
    img_dir        = os.path.join(out_dir, args.img_subdir)
    os.makedirs(img_dir, exist_ok=True)

    shard_paths = sorted(glob.glob(shard_glob_abs))
    if not shard_paths:
        print(f"ERROR: no shards matched {shard_glob_abs}")
        sys.exit(1)

    print(f"Found {len(shard_paths)} shards:")
    for sp in shard_paths:
        print(f"  {sp}")
    print()

    print("Streaming shards → JPEG + metadata ...")
    meta = stream_shards(shard_paths, img_dir, out_dir)
    N = len(meta["angles"])
    print(f"\nTotal frames: {N}")

    angles = meta["angles"]
    cls = np.where(angles <= CLOSED_THRESHOLD_DEG, "CLOSED", "OPEN")
    n_closed = int((cls == "CLOSED").sum())
    n_open   = int((cls == "OPEN").sum())
    print(f"  CLOSED: {n_closed:6d} ({100*n_closed/N:.1f}%)")
    print(f"  OPEN:   {n_open:6d} ({100*n_open/N:.1f}%)")

    print("\nPer-door frame counts:")
    unique_doors, door_counts = np.unique(meta["door_ids"], return_counts=True)
    for d, c in zip(unique_doors, door_counts):
        print(f"  {d:<24s} {c:6d}")

    strat_key = np.array([
        f"{d}|{b}|{l}" for d, b, l in
        zip(meta["door_ids"], meta["angle_band"], meta["light_idx"])
    ])

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=VAL_FRACTION, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(np.zeros(N), strat_key))

    rng = np.random.default_rng(args.seed)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    print(f"\nStratified split (train/val = {100*(1-VAL_FRACTION):.0f}/{100*VAL_FRACTION:.0f}):")
    for name, idx in (("train", train_idx), ("val", val_idx)):
        nc = int((cls[idx] == "CLOSED").sum())
        no = int((cls[idx] == "OPEN").sum())
        print(f"  {name:<5s} total={len(idx):6d}  "
              f"CLOSED={nc} ({100*nc/len(idx):.1f}%)  "
              f"OPEN={no} ({100*no/len(idx):.1f}%)")

    def write_split(indices, jsonl_path, split_name):
        with open(jsonl_path, "w") as f:
            for idx in indices:
                answer = CLOSED_ANSWER if cls[idx] == "CLOSED" else OPEN_ANSWER
                rec = {
                    "conversations": make_conversation(answer),
                    "image":         meta["rel_img_paths"][idx],
                }
                f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print(f"  [{split_name}] {len(indices)} records → {jsonl_path}")

    train_path = os.path.join(out_dir, args.train_name)
    val_path   = os.path.join(out_dir, args.val_name)
    print()
    write_split(train_idx, train_path, "train")
    write_split(val_idx,   val_path,   "val")

    print("\nDone.")
    print(f"  {args.train_name} : {len(train_idx)} records  ({100*(1-VAL_FRACTION):.0f}%)")
    print(f"  {args.val_name}   : {len(val_idx)} records  ({100*VAL_FRACTION:.0f}%)")
    print(f"  Images           : {img_dir}")


if __name__ == "__main__":
    main()
