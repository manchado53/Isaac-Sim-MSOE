#!/usr/bin/env python3
"""
prepare_real_doors_final.py — build a JSONL eval set from real_doors_final/RGB.

Walks real_doors_final/RGB/test/{Closed,Open}/*.png (skips Semi — the unified
V4 adapter has no semi-open class, so it can't be scored cleanly here) and
emits data/val_real_doors_final.jsonl in the same format consumed by
eval_vlm_baseline.py. Image paths are written relative to the JSONL file
(i.e. ../real_doors_final/RGB/test/<label>/<file>).

Usage:
  python3 simulations/robot/prepare_real_doors_final.py
"""

import json
import os
import sys

# Use the unified 4-class training prompt format so the unified V4 adapter
# sees its native instruction format. The eval parser falls back to substring
# matches on "OPEN"/"CLOSED", so [DOOR_OPEN]/[DOOR_CLOSED] responses parse
# correctly to OPEN/CLOSED labels.
VISION_TOKEN_STR = "<|vision_start|><|image_pad|><|vision_end|>"
USER_PROMPT      = (
    f"{VISION_TOKEN_STR}What is shown in this image? "
    "Options: door open, door closed, glass cracked, glass intact."
)
CLOSED_ANSWER    = "State: [DOOR_CLOSED]."
OPEN_ANSWER      = "State: [DOOR_OPEN]."

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SRC_DIR     = os.path.join(PROJECT_DIR, "real_doors_final", "RGB", "test")
OUT_JSONL   = os.path.join(PROJECT_DIR, "data", "val_real_doors_final.jsonl")

LABEL_DIRS = [("Closed", CLOSED_ANSWER), ("Open", OPEN_ANSWER)]


def main():
    if not os.path.isdir(SRC_DIR):
        print(f"ERROR: {SRC_DIR} not found")
        sys.exit(1)

    records = []
    for folder, answer in LABEL_DIRS:
        d = os.path.join(SRC_DIR, folder)
        if not os.path.isdir(d):
            print(f"WARNING: {d} missing — skipping")
            continue
        files = sorted(f for f in os.listdir(d)
                       if f.lower().endswith((".png", ".jpg", ".jpeg")))
        print(f"  {folder}: {len(files)} images")
        for fn in files:
            rel_img = os.path.join("..", "real_doors_final", "RGB", "test", folder, fn)
            records.append({
                "conversations": [
                    {"role": "user",      "content": USER_PROMPT},
                    {"role": "assistant", "content": answer},
                ],
                "image": rel_img,
            })

    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    with open(OUT_JSONL, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())

    n_closed = sum(1 for r in records if "[DOOR_CLOSED]" in r["conversations"][1]["content"])
    n_open   = sum(1 for r in records if "[DOOR_OPEN]"   in r["conversations"][1]["content"])
    print(f"\nWrote {len(records)} records → {OUT_JSONL}")
    print(f"  CLOSED: {n_closed}  OPEN: {n_open}  (Semi excluded)")


if __name__ == "__main__":
    main()
