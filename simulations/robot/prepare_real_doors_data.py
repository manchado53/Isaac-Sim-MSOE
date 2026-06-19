#!/usr/bin/env python3
"""
prepare_real_doors_data.py — build a JSONL eval set from val_real_doors/.

Walks val_real_doors/{Closed,Open}/*.png and emits data/val_real_doors.jsonl
in the same format consumed by eval_vlm_baseline.py. Labels come from the
parent folder name. Image paths are written relative to the JSONL file
(i.e. ../val_real_doors/<label>/<file>) so the eval loader can resolve them
without symlinks or env vars.

Usage:
  python3 simulations/robot/prepare_real_doors_data.py
"""

import json
import os
import sys

VISION_TOKEN_STR = "<|vision_start|><|image_pad|><|vision_end|>"
USER_PROMPT      = f"{VISION_TOKEN_STR}Is this door open or closed?"
CLOSED_ANSWER    = "Door Status: [CLOSED]."
OPEN_ANSWER      = "Door Status: [OPEN]."

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SRC_DIR     = os.path.join(PROJECT_DIR, "val_real_doors")
OUT_JSONL   = os.path.join(PROJECT_DIR, "data", "val_real_doors.jsonl")

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
            # Path relative to the JSONL file (which lives in data/)
            rel_img = os.path.join("..", "val_real_doors", folder, fn)
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

    n_closed = sum(1 for r in records if "[CLOSED]" in r["conversations"][1]["content"])
    n_open   = sum(1 for r in records if "[OPEN]"   in r["conversations"][1]["content"])
    print(f"\nWrote {len(records)} records → {OUT_JSONL}")
    print(f"  CLOSED: {n_closed}  OPEN: {n_open}")


if __name__ == "__main__":
    main()
