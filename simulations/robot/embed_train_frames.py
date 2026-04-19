#!/usr/bin/env python3
"""
embed_train_frames.py — DINOv2 embeddings for diversity-based curation.

Reads data/train.jsonl, runs every JPEG through facebook/dinov2-base, and
saves L2-normalized 768-d embeddings + per-frame metadata. Embeddings are
the input to curate_diversity.py.

Output:
  data/train_v4_embeddings.npy       float32, shape (N, 768), L2-normalized
  data/train_v4_embedding_meta.json  list of {image, label, door_id, line_idx}

One-time cost — embeddings get cached so curation can be re-run with
different k values without re-embedding.

Usage (V100/H100, dgx partition — see submit_curate_diversity.sh):
  python embed_train_frames.py
  python embed_train_frames.py --jsonl data/train_v4.jsonl --batch 128
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))

DOOR_ID_RE = re.compile(r"(Doors_\d+)")


def parse_jsonl(jsonl_path: str):
    """Return list of (image_abs_path, label_str, door_id, line_idx)."""
    data_dir = os.path.dirname(jsonl_path)
    records  = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            rec    = json.loads(line)
            answer = rec["conversations"][1]["content"]
            label  = "CLOSED" if "[CLOSED]" in answer.upper() else "OPEN"
            img_rel = rec["image"]
            img_abs = os.path.join(data_dir, img_rel)
            m = DOOR_ID_RE.search(img_rel)
            door_id = m.group(1) if m else "unknown"
            records.append((img_abs, label, door_id, i, img_rel))
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default="data/train_v4.jsonl")
    p.add_argument("--model", default="facebook/dinov2-base")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--out_emb",  default="data/train_v4_embeddings.npy")
    p.add_argument("--out_meta", default="data/train_v4_embedding_meta.json")
    args = p.parse_args()

    jsonl_path = os.path.join(_project_dir, args.jsonl)
    out_emb    = os.path.join(_project_dir, args.out_emb)
    out_meta   = os.path.join(_project_dir, args.out_meta)

    print(f"Reading {jsonl_path} ...")
    records = parse_jsonl(jsonl_path)
    n_total = len(records)
    n_closed = sum(1 for r in records if r[1] == "CLOSED")
    n_open   = n_total - n_closed
    door_ids = sorted({r[2] for r in records})
    print(f"  {n_total} frames  (CLOSED: {n_closed}, OPEN: {n_open})")
    print(f"  Doors: {len(door_ids)} — {door_ids}")

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
            imgs = []
            for img_abs, _, _, _, _ in batch:
                imgs.append(Image.open(img_abs).convert("RGB"))

            inputs = processor(images=imgs, return_tensors="pt").to(device)
            # CLS token from last hidden state = pooled image embedding
            out = model(**inputs).last_hidden_state[:, 0, :]
            embeddings[batch_start:batch_start + len(batch)] = out.float().cpu().numpy()

            for img_abs, label, door_id, line_idx, img_rel in batch:
                meta.append({
                    "image":    img_rel,
                    "label":    label,
                    "door_id":  door_id,
                    "line_idx": line_idx,
                })

            n_done += len(batch)
            if n_done % 1000 == 0 or n_done == n_total:
                elapsed = time.time() - t_start
                rate    = n_done / max(elapsed, 1e-6)
                eta     = (n_total - n_done) / max(rate, 1e-6)
                print(f"  [{n_done:5d}/{n_total}]  {rate:.1f} fps  "
                      f"elapsed={elapsed/60:.1f} min  eta={eta/60:.1f} min")

    # L2-normalize so Euclidean distance ≡ cosine distance for k-means
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings /= norms

    print(f"\nSaving embeddings → {out_emb}")
    np.save(out_emb, embeddings)
    print(f"  shape={embeddings.shape}  dtype={embeddings.dtype}  "
          f"size={embeddings.nbytes/1e6:.1f} MB")

    print(f"Saving metadata   → {out_meta}")
    with open(out_meta, "w") as f:
        json.dump(meta, f)
        f.flush()
        os.fsync(f.fileno())

    print(f"\nDone in {(time.time()-t_start)/60:.1f} min total.")


if __name__ == "__main__":
    main()
