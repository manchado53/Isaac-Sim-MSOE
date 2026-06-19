#!/usr/bin/env python3
"""
downsample_glass_val_test.py

Produces a rebalanced copy of tanner_vlm_dataset with 1:1 cracked:intact in val
and test. Train passes through unchanged.

Strategy (option C — viewpoint-paired, material-rotated):
  For each panel, each (lighting, arc, distance, UV, height) bucket has 1
  intact frame (M-clear) and 3 cracked frames (one per material: edge_v1 /
  v4 / tempered_v5). Keep all intact frames; for each bucket choose exactly
  one cracked frame by deterministically rotating the material index across
  buckets — so every retained crack image pairs with an intact counterpart
  at the same viewpoint, yet all three crack materials remain represented
  (~135 of each per panel).

Output layout (nothing in the source dataset is overwritten):
  data_glass/tanner_vlm_dataset_balanced/
    frames/train  -> symlink to ../../tanner_vlm_dataset/frames/train
    frames/val/   (405 intact + 405 cracked symlinks per panel)
    frames/test/  (810 intact + 810 cracked symlinks per panel)
    train.jsonl   -> symlink to ../tanner_vlm_dataset/train.jsonl
    val.jsonl     rewritten
    test.jsonl    rewritten
    README.md     rewritten (notes the rebalancing)
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict

FILENAME_RE = re.compile(
    r"^(?P<label>cracked|intact)_pan(?P<panel>\d+)"
    r"_L-(?P<L>.+?)_M-(?P<M>.+?)"
    r"_arc(?P<arc>[+-]\d+)_d(?P<d>\d+)_uv(?P<uv>\d+)_h(?P<h>[+-]\d+)"
    r"_f(?P<f>\d+)\.jpg$"
)


def parse_filename(basename: str) -> dict:
    m = FILENAME_RE.match(basename)
    if not m:
        raise ValueError(f"filename did not match expected schema: {basename}")
    return m.groupdict()


def stable_index(key: str, modulus: int) -> int:
    """Deterministic, hash-randomization-free integer in [0, modulus)."""
    digest = hashlib.md5(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def rebalance_split(records: list, seed_tag: str) -> tuple:
    """Return (kept_records, dropped_count, per_panel_summary).

    Groups cracked records by (panel, L, arc, d, uv, h) — the same buckets
    that index intact frames — and keeps one material per bucket via
    deterministic rotation. Intact records are all kept.
    """
    intact = []
    # bucket_key -> [(record, material)]
    cracked_buckets: dict = defaultdict(list)

    for rec in records:
        base = os.path.basename(rec["image"])
        meta = parse_filename(base)
        if meta["label"] == "intact":
            intact.append(rec)
        else:
            key = (meta["panel"], meta["L"], meta["arc"],
                   meta["d"], meta["uv"], meta["h"])
            cracked_buckets[key].append((rec, meta["M"]))

    kept_cracked = []
    mat_counter: Counter = Counter()
    for key, entries in cracked_buckets.items():
        # Sort materials so rotation is reproducible independent of JSONL order.
        entries_sorted = sorted(entries, key=lambda e: e[1])
        mats = [m for _, m in entries_sorted]
        rot_key = f"{seed_tag}|{'|'.join(key)}"
        idx = stable_index(rot_key, len(entries_sorted))
        chosen_rec, chosen_mat = entries_sorted[idx]
        kept_cracked.append(chosen_rec)
        mat_counter[chosen_mat] += 1

    kept = intact + kept_cracked

    # Per-panel summary for sanity-checking.
    panel_summary: dict = defaultdict(lambda: {"intact": 0, "cracked": 0})
    for rec in kept:
        base = os.path.basename(rec["image"])
        meta = parse_filename(base)
        panel_summary[meta["panel"]][meta["label"]] += 1

    dropped = len(records) - len(kept)
    return kept, dropped, dict(panel_summary), mat_counter


def load_jsonl(path: str) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str, records: list) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def make_symlink(src_abs: str, dst_abs: str) -> None:
    if os.path.lexists(dst_abs):
        os.remove(dst_abs)
    os.symlink(src_abs, dst_abs)


def link_frames(records: list, src_frames_dir: str, dst_frames_dir: str) -> None:
    os.makedirs(dst_frames_dir, exist_ok=True)
    for rec in records:
        base = os.path.basename(rec["image"])
        src = os.path.join(src_frames_dir, base)
        dst = os.path.join(dst_frames_dir, base)
        if not os.path.isfile(src):
            raise FileNotFoundError(src)
        make_symlink(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="tanner_vlm_dataset",
                        help="Source dataset dir (relative to this script)")
    parser.add_argument("--dst", default="tanner_vlm_dataset_balanced",
                        help="Output dir (relative to this script)")
    parser.add_argument("--seed-tag", default="v1",
                        help="Namespace tag mixed into the rotation hash; "
                             "change to get a different deterministic draw.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(script_dir, args.src)
    dst_dir = os.path.join(script_dir, args.dst)

    if not os.path.isdir(src_dir):
        print(f"ERROR: source dir not found: {src_dir}", file=sys.stderr)
        sys.exit(1)
    if os.path.exists(dst_dir):
        print(f"ERROR: output dir already exists: {dst_dir}\n"
              f"Delete or rename it first.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(dst_dir)
    os.makedirs(os.path.join(dst_dir, "frames"))

    # Pass-through: symlink the whole train frames dir + train.jsonl.
    make_symlink(
        os.path.join(src_dir, "frames", "train"),
        os.path.join(dst_dir, "frames", "train"),
    )
    make_symlink(
        os.path.join(src_dir, "train.jsonl"),
        os.path.join(dst_dir, "train.jsonl"),
    )

    # Rebalance val and test.
    for split in ("val", "test"):
        print(f"\n=== {split} ===")
        src_jsonl = os.path.join(src_dir, f"{split}.jsonl")
        src_frames = os.path.join(src_dir, "frames", split)
        dst_jsonl = os.path.join(dst_dir, f"{split}.jsonl")
        dst_frames = os.path.join(dst_dir, "frames", split)

        records = load_jsonl(src_jsonl)
        kept, dropped, panel_summary, mat_counter = rebalance_split(
            records, seed_tag=f"{args.seed_tag}|{split}"
        )

        print(f"  records: {len(records)} -> {len(kept)}  (dropped {dropped})")
        print(f"  crack material distribution after downsample:")
        for m, c in sorted(mat_counter.items()):
            print(f"    {m:<12s} {c}")
        print(f"  per-panel (intact / cracked):")
        for p in sorted(panel_summary):
            s = panel_summary[p]
            print(f"    pan{p}: {s['intact']} / {s['cracked']}")

        write_jsonl(dst_jsonl, kept)
        link_frames(kept, src_frames, dst_frames)

    # README.
    readme = f"""# Rebalanced cracked-glass dataset

Generated from `{args.src}/` by `../downsample_glass_val_test.py`.

## What changed vs source

- **train**: unchanged (symlinked through).
- **val**: 1,620 -> 810 (1:1 cracked:intact).
- **test**: 3,240 -> 1,620 (1:1 cracked:intact).

Per-panel counts are now 405 intact + 405 cracked in each val/test panel.

## How cracked frames were downsampled

Each panel has one intact frame and three cracked frames per
(lighting, arc, distance, UV, height) bucket — one per material variant
({{edge_v1, v4, tempered_v5}}). For each bucket we kept exactly one cracked
frame; which material was chosen rotates deterministically across buckets
via `md5(seed_tag|split|panel|L|arc|d|uv|h) % 3`, so:

1. Every retained crack frame pairs with a same-viewpoint, same-lighting,
   same-UV intact frame.
2. All three crack materials remain represented (~135 of each per panel).
3. The draw is reproducible: re-running with the same `--seed-tag` yields
   identical splits.

Seed tag used: `{args.seed_tag}`.

## File layout

- `frames/train/` symlinks to `{args.src}/frames/train/`
- `frames/val/` and `frames/test/` contain only the retained frames as
  symlinks into `{args.src}/frames/...`
- `train.jsonl` symlinks to `{args.src}/train.jsonl`
- `val.jsonl` and `test.jsonl` are rewritten.

Image paths in the JSONLs remain relative to this directory, so the dataset
can be consumed with `--data_dir data_glass/{args.dst}` in `finetune_qwen2vl.py`.
"""
    with open(os.path.join(dst_dir, "README.md"), "w") as f:
        f.write(readme)

    print(f"\nDone. Output: {dst_dir}")


if __name__ == "__main__":
    main()
