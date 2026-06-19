#!/usr/bin/env python3
"""
prepare_unified_data.py — build a 4-class door+glass JSONL for joint fine-tuning.

Reads:
  data/train_v4.jsonl                                       (42,228 door records)
  data/val_v4.jsonl                                         ( 7,452 door records)
  data_glass/tanner_vlm_dataset_balanced/train.jsonl        ( 8,915 glass records)
  data_glass/tanner_vlm_dataset_balanced/val.jsonl          (   810 glass records)

Writes:
  data/train_unified.jsonl   (door_down + glass, single unified schema)
  data/val_unified.jsonl     (door val + glass val, unified schema)

Unified schema
--------------
prompt : "What is shown in this image? Options: door open, door closed, glass cracked, glass intact."
labels : State: [DOOR_OPEN]. | State: [DOOR_CLOSED]. | State: [GLASS_CRACKED]. | State: [GLASS_INTACT].

Door records come in with the old schema ("Is this door open or closed?" →
"Door Status: [OPEN]./[CLOSED].") and are rewritten. Image paths stay
relative to data/ (no re-render).

Glass records already use the unified schema; only image paths are
rewritten from "frames/train/..." (relative to the glass dataset) to
"../data_glass/tanner_vlm_dataset_balanced/frames/train/..." (relative
to data/).

Door downsampling
-----------------
Default target: 9,000 door records (4,500 OPEN + 4,500 CLOSED),
stratified by door_id within each label. Pass --door-target to change.

Why downsample: the full 42k door set severely over-trained the
binary-classification pattern (eval loss plateaued at epoch 1) and
biased the model to emit door-status labels regardless of prompt. A
~9k door set matches the glass scale, keeps ~4 classes roughly equal,
and cuts training time from ~18h to ~7h on V100.
"""

import argparse
import collections
import json
import os
import random
import sys

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))

UNIFIED_PROMPT = (
    "<|vision_start|><|image_pad|><|vision_end|>"
    "What is shown in this image? Options: door open, door closed, "
    "glass cracked, glass intact."
)

DOOR_LABEL_MAP = {
    "Door Status: [OPEN].":   "State: [DOOR_OPEN].",
    "Door Status: [CLOSED].": "State: [DOOR_CLOSED].",
}

GLASS_DATASET_REL = "data_glass/tanner_vlm_dataset_balanced"


def make_record(answer: str, image_rel_to_data: str) -> dict:
    return {
        "conversations": [
            {"role": "user",      "content": UNIFIED_PROMPT},
            {"role": "assistant", "content": answer},
        ],
        "image": image_rel_to_data,
    }


def load_door_jsonl(path: str):
    """Yields (new_answer, image_rel_to_data, door_id) for each door record."""
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            old_ans = r["conversations"][1]["content"]
            new_ans = DOOR_LABEL_MAP.get(old_ans)
            if new_ans is None:
                raise ValueError(f"Unexpected door label: {old_ans!r} in {path}")
            img = r["image"]                              # "finetune_images_v4/Doors_XXX_00001.jpg"
            door_id = os.path.basename(img).rsplit("_", 1)[0]
            yield new_ans, img, door_id


def load_glass_jsonl(path: str, split_name: str):
    """
    Glass records already use the unified schema. Image path is
    "frames/{split}/..." relative to the glass dataset dir; rewrite to
    "../data_glass/tanner_vlm_dataset_balanced/frames/{split}/..."
    relative to data/.
    """
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            ans = r["conversations"][1]["content"]
            img_rel_glass  = r["image"]                   # "frames/train/..."
            img_rel_data   = os.path.join("..", GLASS_DATASET_REL, img_rel_glass)
            yield ans, img_rel_data


def stratified_downsample(records, key_fn, target_n, seed=42):
    """
    records: list of (answer, image_path, strat_key)
    Groups by (answer, strat_key) and samples proportionally to reach target_n.
    Within each (answer,strat_key) bucket, samples uniformly at random.
    Returns a new list with len ~= target_n.
    """
    rng = random.Random(seed)
    by_ans = collections.defaultdict(list)
    for rec in records:
        by_ans[rec[0]].append(rec)

    out = []
    per_ans_target = target_n // len(by_ans)
    for ans, recs in by_ans.items():
        by_key = collections.defaultdict(list)
        for r in recs:
            by_key[r[2]].append(r)
        total_in_ans = len(recs)
        for key, bucket in by_key.items():
            # proportional target within this answer class
            share = per_ans_target * len(bucket) / total_in_ans
            take = max(1, round(share))
            take = min(take, len(bucket))
            rng.shuffle(bucket)
            out.extend(bucket[:take])
    rng.shuffle(out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--door-train",  default="data/train_v4_curated.jsonl",
                   help="Curated 10k door set (pre-balanced 5k/5k, 8 doors stratified)")
    p.add_argument("--door-val",    default="data/val_v4.jsonl")
    p.add_argument("--glass-train", default=f"{GLASS_DATASET_REL}/train.jsonl")
    p.add_argument("--glass-val",   default=f"{GLASS_DATASET_REL}/val.jsonl")
    p.add_argument("--out-train",      default="data/train_unified.jsonl")
    p.add_argument("--out-val",        default="data/val_unified.jsonl")
    p.add_argument("--out-val-door",   default="data/val_unified_door.jsonl",
                   help="Per-domain val for isolated door eval runs")
    p.add_argument("--out-val-glass",  default="data/val_unified_glass.jsonl",
                   help="Per-domain val for isolated glass eval runs")
    p.add_argument("--door-target", type=int, default=0,
                   help="Target door train size (0 = keep all — default for curated input)")
    p.add_argument("--seed",        type=int, default=42)
    args = p.parse_args()

    print(f"Project root: {_project_dir}\n")

    # ------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------
    print("=== TRAIN ===")
    door_train_path = os.path.join(_project_dir, args.door_train)
    door_recs = list(load_door_jsonl(door_train_path))
    print(f"door train loaded       : {len(door_recs)}")
    ans_counts = collections.Counter(r[0] for r in door_recs)
    for a, n in ans_counts.most_common():
        print(f"  {a:<32s} {n}")

    if args.door_target > 0 and args.door_target < len(door_recs):
        door_recs = stratified_downsample(door_recs,
                                          key_fn=lambda r: r[2],
                                          target_n=args.door_target,
                                          seed=args.seed)
        print(f"door downsampled to     : {len(door_recs)}")
        ans_counts = collections.Counter(r[0] for r in door_recs)
        for a, n in ans_counts.most_common():
            print(f"  {a:<32s} {n}")

    glass_train_path = os.path.join(_project_dir, args.glass_train)
    glass_recs = list(load_glass_jsonl(glass_train_path, "train"))
    print(f"\nglass train loaded      : {len(glass_recs)}")
    ans_counts = collections.Counter(r[0] for r in glass_recs)
    for a, n in ans_counts.most_common():
        print(f"  {a:<32s} {n}")

    all_train = ([(ans, img) for ans, img, _ in door_recs]
                 + list(glass_recs))

    rng = random.Random(args.seed)
    rng.shuffle(all_train)

    out_train = os.path.join(_project_dir, args.out_train)
    _write_jsonl(all_train, out_train)
    print(f"\nwrote {len(all_train)} records → {out_train}")
    _print_mix(all_train)

    # ------------------------------------------------------------------
    # VAL
    # ------------------------------------------------------------------
    print("\n=== VAL ===")
    door_val = [(ans, img) for ans, img, _ in
                load_door_jsonl(os.path.join(_project_dir, args.door_val))]
    glass_val = list(load_glass_jsonl(
        os.path.join(_project_dir, args.glass_val), "val"))

    all_val = door_val + glass_val
    rng.shuffle(all_val)

    out_val       = os.path.join(_project_dir, args.out_val)
    out_val_door  = os.path.join(_project_dir, args.out_val_door)
    out_val_glass = os.path.join(_project_dir, args.out_val_glass)
    _write_jsonl(all_val,   out_val)
    _write_jsonl(door_val,  out_val_door)
    _write_jsonl(glass_val, out_val_glass)
    print(f"\nwrote {len(all_val)} records → {out_val}")
    _print_mix(all_val)
    print(f"\nper-domain val files for isolated eval:")
    print(f"  door  : {len(door_val):6d} → {out_val_door}")
    print(f"  glass : {len(glass_val):6d} → {out_val_glass}")

    # ------------------------------------------------------------------
    # Sanity check: all image paths resolve relative to data/
    # ------------------------------------------------------------------
    data_dir = os.path.join(_project_dir, "data")
    missing = 0
    for ans, img in all_val[:200]:
        if not os.path.isfile(os.path.join(data_dir, img)):
            missing += 1
    if missing:
        print(f"\nWARN: {missing}/200 sampled val images missing on disk")
    else:
        print(f"\nsampled val image path check: OK")


def _write_jsonl(records, path):
    with open(path, "w") as f:
        for ans, img in records:
            f.write(json.dumps(make_record(ans, img)) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _print_mix(records):
    cnt = collections.Counter(ans for ans, _ in records)
    for a, n in cnt.most_common():
        print(f"  {a:<32s} {n:6d}  ({100*n/len(records):.1f}%)")


if __name__ == "__main__":
    main()
