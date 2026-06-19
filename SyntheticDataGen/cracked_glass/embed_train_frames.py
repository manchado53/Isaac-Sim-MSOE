#!/usr/bin/env python3
"""
embed_train_frames.py -- DINOv2 embeddings for diversity-based curation.

Port of origin/tanner_vlm:simulations/robot/embed_train_frames.py (commit
207c0ba). Two deltas:
  - label source: rec["label"] ("intact"|"cracked") instead of parsing the
    assistant's "[CLOSED]"/"[OPEN]" answer.
  - id source:    rec["panel_idx"] (int) instead of regex on the JPEG path.

Reads <dataset>/labels_train.jsonl, runs every JPEG through
facebook/dinov2-base, and saves L2-normalized 768-d embeddings plus
per-frame metadata. These are the input to curate_diversity.py.

Output (next to the source JSONL):
  <dataset>/train_embeddings.npy       float32 (N, 768), L2-normalized
  <dataset>/train_embedding_meta.json  list of {image, label, panel_idx, line_idx}

One-time cost -- embeddings get cached so curation can be re-run with
different --k values without re-embedding.

Usage:
  python embed_train_frames.py --jsonl datasets/cracked_glass/<ts>/labels_train.jsonl
  python embed_train_frames.py --jsonl <path> --batch 128
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))


def parse_jsonl(jsonl_path: str):
    """Return list of (image_abs_path, label_str, panel_idx, line_idx, image_rel)."""
    data_dir = os.path.dirname(jsonl_path)
    records  = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            label = rec["label"]
            panel_idx = int(rec["panel_idx"])
            # Collector writes frames/<split>/<frame_id>.jpg relative to
            # the dataset root (same dir as the JSONL).
            split = rec.get("split", "train")
            img_rel = rec.get("image") or f"frames/{split}/{rec['frame_id']}.jpg"
            img_abs = os.path.join(data_dir, img_rel)
            records.append((img_abs, label, panel_idx, i, img_rel))
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl",    required=True,
                   help="Path to labels_train.jsonl (only train-split rows).")
    p.add_argument("--model",    default="facebook/dinov2-base")
    p.add_argument("--batch",    type=int, default=128)
    p.add_argument("--out_emb",  default=None,
                   help="Default: <dataset>/train_embeddings.npy next to JSONL.")
    p.add_argument("--out_meta", default=None,
                   help="Default: <dataset>/train_embedding_meta.json next to JSONL.")
    args = p.parse_args()

    jsonl_path = os.path.abspath(args.jsonl)
    data_dir   = os.path.dirname(jsonl_path)
    out_emb    = args.out_emb  or os.path.join(data_dir, "train_embeddings.npy")
    out_meta   = args.out_meta or os.path.join(data_dir, "train_embedding_meta.json")

    print(f"Reading {jsonl_path} ...")
    records = parse_jsonl(jsonl_path)
    n_total    = len(records)
    n_intact   = sum(1 for r in records if r[1] == "intact")
    n_cracked  = n_total - n_intact
    panel_ids  = sorted({r[2] for r in records})
    print(f"  {n_total} frames  (intact: {n_intact}, cracked: {n_cracked})")
    print(f"  Panels: {len(panel_ids)} -- {panel_ids}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading {args.model} on {device} ...")
    t0 = time.time()
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(
        args.model, torch_dtype=torch.float16,
    ).to(device).eval()
    embed_dim = model.config.hidden_size
    print(f"  Loaded in {time.time()-t0:.1f}s, embedding dim = {embed_dim}")

    embeddings = np.zeros((n_total, embed_dim), dtype=np.float32)
    meta = []

    print(f"\nEmbedding {n_total} frames in batches of {args.batch} ...")
    t_start = time.time()
    n_done = 0

    with torch.no_grad():
        for batch_start in range(0, n_total, args.batch):
            batch = records[batch_start:batch_start + args.batch]
            imgs = [Image.open(r[0]).convert("RGB") for r in batch]
            inputs = processor(images=imgs, return_tensors="pt").to(device)
            # CLS token from last hidden state = pooled image embedding
            out = model(**inputs).last_hidden_state[:, 0, :]
            embeddings[batch_start:batch_start + len(batch)] = out.float().cpu().numpy()

            for img_abs, label, panel_idx, line_idx, img_rel in batch:
                meta.append({
                    "image":     img_rel,
                    "label":     label,
                    "panel_idx": panel_idx,
                    "line_idx":  line_idx,
                })

            n_done += len(batch)
            if n_done % 1000 == 0 or n_done == n_total:
                elapsed = time.time() - t_start
                rate    = n_done / max(elapsed, 1e-6)
                eta     = (n_total - n_done) / max(rate, 1e-6)
                print(f"  [{n_done:5d}/{n_total}]  {rate:.1f} fps  "
                      f"elapsed={elapsed/60:.1f} min  eta={eta/60:.1f} min")

    # L2-normalize so Euclidean distance == cosine distance for k-means
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings /= norms

    print(f"\nSaving embeddings -> {out_emb}")
    np.save(out_emb, embeddings)
    print(f"  shape={embeddings.shape}  dtype={embeddings.dtype}  "
          f"size={embeddings.nbytes/1e6:.1f} MB")

    print(f"Saving metadata   -> {out_meta}")
    with open(out_meta, "w") as f:
        json.dump(meta, f)
        f.flush()
        os.fsync(f.fileno())

    print(f"\nDone in {(time.time()-t_start)/60:.1f} min total.")


if __name__ == "__main__":
    main()
