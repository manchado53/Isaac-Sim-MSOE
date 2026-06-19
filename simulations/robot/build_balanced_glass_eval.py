#!/usr/bin/env python3
"""
build_balanced_glass_eval.py

Assemble a balanced 50 intact + 50 cracked eval JSONL from the existing
DH cracked-glass composites. Intact side is upsampled via horizontal flip
because only 23 unique DH photos exist; cracked side is downsampled from
the 69 composites with material balance preserved.

Output:
  data/dh_cracked_eval/intact_flip/Image_NN.jpg  (27 horizontally-flipped)
  data/dh_cracked_eval/balanced_test.jsonl       (100 rows: 50 + 50)
"""

import json
import os
import random
import sys

from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EVAL_DIR = os.path.join(ROOT, "data", "dh_cracked_eval")
INTACT_DIR = os.path.join(EVAL_DIR, "intact")
CRACKED_DIR = os.path.join(EVAL_DIR, "cracked")
FLIP_DIR = os.path.join(EVAL_DIR, "intact_flip")
OUT_JSONL = os.path.join(EVAL_DIR, "balanced_test.jsonl")

PROMPT = ("<|vision_start|><|image_pad|><|vision_end|>"
          "What is shown in this image? "
          "Options: door open, door closed, glass cracked, glass intact.")

MATERIALS = ["edge_v1", "v4", "tempered_v5"]
TARGET_PER_CLASS = 50


def jsonl_row(image_rel: str, label: str) -> dict:
    state = "GLASS_CRACKED" if label == "cracked" else "GLASS_INTACT"
    return {
        "conversations": [
            {"role": "user", "content": PROMPT},
            {"role": "assistant", "content": f"State: [{state}]."},
        ],
        "image": image_rel,
    }


def main():
    if not os.path.isdir(INTACT_DIR) or not os.path.isdir(CRACKED_DIR):
        print(f"ERROR: expected {INTACT_DIR} and {CRACKED_DIR} to exist.",
              file=sys.stderr)
        sys.exit(1)

    os.makedirs(FLIP_DIR, exist_ok=True)

    rng = random.Random(0)

    intact_files = sorted(
        f for f in os.listdir(INTACT_DIR) if f.lower().endswith(".jpg")
    )
    print(f"intact originals: {len(intact_files)}")

    # Upsample intact to TARGET_PER_CLASS using horizontal flips.
    intact_rows = []
    for f in intact_files:
        intact_rows.append(jsonl_row(os.path.join("intact", f), "intact"))

    needed = TARGET_PER_CLASS - len(intact_rows)
    if needed > 0:
        flip_pool = list(intact_files)
        rng.shuffle(flip_pool)
        # Round-robin through the pool; when we wrap, tag the extra augmented
        # files with a pass counter so each augmented file has a unique name
        # (otherwise the JSONL picks up duplicate rows).
        i = 0
        while needed > 0:
            src = flip_pool[i % len(flip_pool)]
            pass_idx = i // len(flip_pool)  # 0 for first wrap, 1 for second, ...
            i += 1
            stem = os.path.splitext(src)[0]
            if pass_idx == 0:
                flip_name = f"{stem}_hflip.jpg"
                transform = Image.FLIP_LEFT_RIGHT
            else:
                # Second wrap: vertical flip on top of horizontal (180° rotate).
                flip_name = f"{stem}_hflip_v{pass_idx}.jpg"
                transform = Image.ROTATE_180 if pass_idx == 1 else Image.FLIP_TOP_BOTTOM
            flip_path = os.path.join(FLIP_DIR, flip_name)
            if not os.path.exists(flip_path):
                with Image.open(os.path.join(INTACT_DIR, src)) as im:
                    flipped = im.transpose(transform)
                    flipped.save(flip_path, quality=92)
            intact_rows.append(
                jsonl_row(os.path.join("intact_flip", flip_name), "intact")
            )
            needed -= 1
    elif needed < 0:
        rng.shuffle(intact_rows)
        intact_rows = intact_rows[:TARGET_PER_CLASS]
    print(f"intact rows: {len(intact_rows)}")

    # Cracked: material-balanced subsample.
    cracked_files = sorted(
        f for f in os.listdir(CRACKED_DIR) if f.lower().endswith(".jpg")
    )
    by_mat = {m: [] for m in MATERIALS}
    for f in cracked_files:
        base = os.path.splitext(f)[0]
        if "__" not in base:
            continue
        mat = base.rsplit("__", 1)[1]
        if mat in by_mat:
            by_mat[mat].append(f)
    for m in MATERIALS:
        rng.shuffle(by_mat[m])

    # Even split; distribute remainder to the first few materials.
    base_n = TARGET_PER_CLASS // len(MATERIALS)
    rem = TARGET_PER_CLASS - base_n * len(MATERIALS)
    quotas = {m: base_n + (1 if i < rem else 0) for i, m in enumerate(MATERIALS)}

    cracked_rows = []
    for m in MATERIALS:
        take = by_mat[m][: quotas[m]]
        if len(take) < quotas[m]:
            print(f"WARNING: only {len(take)} cracked/{m} available, "
                  f"wanted {quotas[m]}", file=sys.stderr)
        for f in take:
            cracked_rows.append(jsonl_row(os.path.join("cracked", f), "cracked"))
    print(f"cracked rows: {len(cracked_rows)}  "
          f"(per material: { {m: quotas[m] for m in MATERIALS} })")

    rows = intact_rows + cracked_rows
    rng.shuffle(rows)

    with open(OUT_JSONL, "w") as out:
        for r in rows:
            out.write(json.dumps(r) + "\n")
        out.flush()
        os.fsync(out.fileno())

    n_intact = sum(1 for r in rows
                   if "GLASS_INTACT" in r["conversations"][1]["content"])
    n_cracked = len(rows) - n_intact
    print(f"\nWrote {OUT_JSONL}: {n_intact} intact + {n_cracked} cracked "
          f"= {len(rows)} total")


if __name__ == "__main__":
    main()
