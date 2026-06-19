#!/usr/bin/env python3
"""
build_tanner_dataset.py -- Assemble a self-contained Qwen2-VL-ready
dataset for joint door/glass training.

Reads three source JSONLs (curated train + raw val + raw test), emits
Qwen2-VL conversation-format JSONLs, and hardlinks the referenced JPEGs
into a new dir tree. Nothing in the source dataset is modified or
deleted -- hardlinks live on the same filesystem and share inodes, so
the `frames/` trees in the source and the output both point to the same
bytes on disk.

Output tree:

    <out>/
    ├── frames/
    │   ├── train/    ~8,915 JPEGs (hardlinks to curated rows)
    │   ├── val/      1,620 JPEGs  (hardlinks to all val rows)
    │   └── test/     3,240 JPEGs  (hardlinks to all test rows)
    ├── train.jsonl   Qwen2-VL conversation format
    ├── val.jsonl
    ├── test.jsonl
    └── README.md

The 4-class prompt is unified across domains so Tanner can merge the
cracked-glass dataset with his door dataset for joint training. Answer
strings encode both the domain and the state:
    State: [DOOR_OPEN] | [DOOR_CLOSED] | [GLASS_CRACKED] | [GLASS_INTACT]
"""

import argparse
import json
import os
import sys
from pathlib import Path


VISION_TOKEN_STR = "<|vision_start|><|image_pad|><|vision_end|>"
USER_PROMPT = (
    f"{VISION_TOKEN_STR}"
    "What is shown in this image? "
    "Options: door open, door closed, glass cracked, glass intact."
)
GLASS_CRACKED_ANSWER = "State: [GLASS_CRACKED]."
GLASS_INTACT_ANSWER  = "State: [GLASS_INTACT]."


def make_conversation(answer: str) -> list:
    return [
        {"role": "user",      "content": USER_PROMPT},
        {"role": "assistant", "content": answer},
    ]


def label_to_answer(label: str) -> str:
    if label == "cracked":
        return GLASS_CRACKED_ANSWER
    if label == "intact":
        return GLASS_INTACT_ANSWER
    raise ValueError(f"unexpected label: {label!r}")


def convert_rows(src_rows: list, split: str, dataset_dir: Path, out_dir: Path):
    """Hardlink JPEGs + return Qwen2-VL records with rewritten image paths."""
    out_frames = out_dir / "frames" / split
    out_frames.mkdir(parents=True, exist_ok=True)
    records = []
    for r in src_rows:
        src_img = dataset_dir / r["image"]
        if not src_img.is_file():
            raise FileNotFoundError(f"missing JPEG for row: {src_img}")
        # Flatten filename (no subdirs) under out_frames/<split>/
        dst_img = out_frames / src_img.name
        if not dst_img.exists():
            os.link(src_img, dst_img)   # hardlink, same fs, same inode
        records.append({
            "conversations": make_conversation(label_to_answer(r["label"])),
            "image":         f"frames/{split}/{src_img.name}",
        })
    return records


def write_jsonl(path: Path, records: list):
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def write_readme(out_dir: Path, counts: dict, panel_info: dict,
                 source_dataset: str):
    """Emit README describing the dataset layout + the Qwen2-VL schema."""
    content = f"""# Cracked-glass VLM dataset (Qwen2-VL format)

Self-contained dataset for fine-tuning Qwen2-VL on the 4-class
door/glass anomaly task. Intended to be merged with the door-state
dataset for joint training.

## Contents

```
{out_dir.name}/
├── frames/
│   ├── train/    {counts['train']} JPEGs
│   ├── val/      {counts['val']} JPEGs
│   └── test/     {counts['test']} JPEGs
├── train.jsonl
├── val.jsonl
├── test.jsonl
└── README.md
```

**Totals**: {counts['train']} train, {counts['val']} val, {counts['test']} test = {sum(counts.values())} JPEGs.

## JSONL schema (Qwen2-VL conversation format)

Drop-in for Tanner's `finetune_qwen2vl.py`. Each line:

```json
{{
  "conversations": [
    {{"role": "user",      "content": "<vision-token>What is shown in this image? Options: door open, door closed, glass cracked, glass intact."}},
    {{"role": "assistant", "content": "State: [GLASS_CRACKED]."}}
  ],
  "image": "frames/train/cracked_panNN_....jpg"
}}
```

- `image` path is **relative to this directory** (same convention as
  Tanner's door JSONLs — `os.path.join(data_dir, rec["image"])`).
- Vision token prefix `<|vision_start|><|image_pad|><|vision_end|>` is
  included in the user prompt.

## Unified 4-class prompt

Same user prompt across both domains:

> What is shown in this image? Options: door open, door closed, glass
> cracked, glass intact.

Answer strings (consistent `State: [LABEL].` format):

| Domain | Positive | Negative |
|---|---|---|
| Door  | `State: [DOOR_OPEN].`     | `State: [DOOR_CLOSED].` |
| Glass | `State: [GLASS_CRACKED].` | `State: [GLASS_INTACT].`|

**Tanner's door dataset**: his `prepare_finetune_data.py` needs to be
updated to emit the new `State: [DOOR_OPEN/CLOSED].` strings instead of
his original `Door Status: [OPEN/CLOSED].`. Two-line change, then
re-run his prepare step.

## Panel-level splits

| Split | Panels | Rationale |
|---|---|---|
| train | {", ".join(f"#{p}" for p in panel_info['train'])} | 10 geographic clusters, curated to {counts['train']} via DINOv2 + stratified k-means |
| val   | {", ".join(f"#{p}" for p in panel_info['val'])}   | single held-out panel for fast mid-training checkpoints |
| test  | {", ".join(f"#{p}" for p in panel_info['test'])}  | two held-out panels for final generalization eval |

Train split is panel-stratified (no panel appears in both train and
val/test), so held-out metrics probe cross-panel generalization rather
than within-panel memorisation.

## Sweep axes (train source, pre-curation)

Per panel, full factorial grid:
- 5 lighting presets (`bright_fluorescent`, `daylight_flood`, `warm_evening`, `overcast_flat`, `high_contrast`)
- 4 states (1 intact + 3 crack variants: `edge_v1`, `v4`, `tempered_v5`)
- 3 arc angles ({{-0.4, 0, 0.4}} rad)
- 3 arc distances ({{1.5, 2.0, 2.5}} m)
- 3 UV tiling levels ({{1x, 2x, 4x}})
- 3 height offsets ({{-0.5, 0, 0.5}} m)

Panel 3's arc=-0.4 viewpoints were excluded at render time (dark
interior + bright backlit doorway made cracks unreadable).

## Curation

Train curated from 15,660 → {counts['train']} via the pattern from
`origin/tanner_vlm:curate_diversity.py`:

1. Embed every train JPEG with `facebook/dinov2-base` (CLS token,
   L2-normalized, 768-d).
2. Stratified k-means per `(label, panel_idx)` bucket.
3. For each centroid keep the frame nearest to it.

Val and test are **not** curated -- kept whole so held-out metrics
reflect the full distribution.

## Provenance

- Source dataset: `{source_dataset}`
- Built with: `SyntheticDataGen/cracked_glass/build_tanner_dataset.py`
"""
    (out_dir / "README.md").write_text(content)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   help="Source dataset dir (contains labels_*.jsonl + frames/).")
    p.add_argument("--curated-jsonl", required=True,
                   help="Curated train JSONL from curate_diversity.py.")
    p.add_argument("--out", required=True,
                   help="Destination dir (new, will be created).")
    args = p.parse_args()

    dataset_dir = Path(args.dataset).resolve()
    out_dir     = Path(args.out).resolve()
    curated     = Path(args.curated_jsonl).resolve()

    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"ERROR: {out_dir} exists and is non-empty. Remove or pick a fresh path.")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source dataset: {dataset_dir}")
    print(f"Curated train:  {curated}")
    print(f"Output:         {out_dir}")
    print()

    # Load source rows.
    train_rows = [json.loads(l) for l in curated.open()]
    val_rows   = [json.loads(l) for l in (dataset_dir / "labels_val.jsonl").open()]
    test_rows  = [json.loads(l) for l in (dataset_dir / "labels_test.jsonl").open()]

    # Convert each split.
    print(f"Converting train ({len(train_rows)} rows)...")
    train_recs = convert_rows(train_rows, "train", dataset_dir, out_dir)
    print(f"  hardlinked {len(train_recs)} JPEGs -> frames/train/")

    print(f"Converting val ({len(val_rows)} rows)...")
    val_recs = convert_rows(val_rows, "val", dataset_dir, out_dir)
    print(f"  hardlinked {len(val_recs)} JPEGs -> frames/val/")

    print(f"Converting test ({len(test_rows)} rows)...")
    test_recs = convert_rows(test_rows, "test", dataset_dir, out_dir)
    print(f"  hardlinked {len(test_recs)} JPEGs -> frames/test/")

    # Write JSONLs.
    write_jsonl(out_dir / "train.jsonl", train_recs)
    write_jsonl(out_dir / "val.jsonl",   val_recs)
    write_jsonl(out_dir / "test.jsonl",  test_recs)
    print(f"\nWrote train.jsonl, val.jsonl, test.jsonl")

    # Sanity: per-split counts + class balance + panel IDs.
    from collections import Counter
    counts = {"train": len(train_recs), "val": len(val_recs), "test": len(test_recs)}
    panel_info = {
        "train": sorted({r["panel_idx"] for r in train_rows}),
        "val":   sorted({r["panel_idx"] for r in val_rows}),
        "test":  sorted({r["panel_idx"] for r in test_rows}),
    }
    print(f"\nPer-split counts: {counts}")
    print(f"Panels per split: {panel_info}")
    print(f"Train class balance: {Counter(r['label'] for r in train_rows)}")

    write_readme(out_dir, counts, panel_info, str(dataset_dir))
    print(f"Wrote README.md")

    print(f"\nDone. Dataset at {out_dir}")


if __name__ == "__main__":
    main()
