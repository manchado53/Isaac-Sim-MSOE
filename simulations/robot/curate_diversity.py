#!/usr/bin/env python3
"""
curate_diversity.py — Stratified k-means diversity sampling.

Reads DINOv2 embeddings produced by embed_train_frames.py and selects K frames
that maximize visual coverage while preserving exact 50/50 CLOSED/OPEN balance.

Stratification: clustering runs INDEPENDENTLY within each (door_id, label)
bucket. K is split:
  - 50% to CLOSED, 50% to OPEN          (label balance, hard constraint)
  - within each label, proportional to per-door frame count
For each bucket we run k-means with k = bucket allocation, then keep the
frame nearest each centroid.

Output: data/train_v4_curated.jsonl — drop-in replacement for train_v4.jsonl.

Usage:
  python curate_diversity.py --k 10000
  python curate_diversity.py --k 5000 --jsonl data/train_v4.jsonl
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict, Counter

import numpy as np

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))


def kmeans_select(emb_subset: np.ndarray, k: int, seed: int):
    """
    Run k-means on emb_subset (n, d), return indices (into emb_subset) of the
    frames closest to each of the k centroids. Uses faiss if available,
    falls back to sklearn MiniBatchKMeans.
    """
    n, d = emb_subset.shape
    if k >= n:
        # Trivial: ask for more clusters than points → keep them all
        return np.arange(n)

    try:
        import faiss
        km = faiss.Kmeans(d=d, k=k, niter=25, seed=seed, verbose=False, gpu=False)
        km.train(emb_subset.astype(np.float32))
        index = faiss.IndexFlatL2(d)
        index.add(emb_subset.astype(np.float32))
        _, nearest = index.search(km.centroids, 1)
        sel = np.unique(nearest.flatten())
        # If two centroids picked the same frame, top up with random unselected
        if len(sel) < k:
            remaining = np.setdiff1d(np.arange(n), sel)
            rng = np.random.default_rng(seed)
            extra = rng.choice(remaining, size=k - len(sel), replace=False)
            sel = np.concatenate([sel, extra])
        return sel
    except ImportError:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics.pairwise import euclidean_distances
        km = MiniBatchKMeans(n_clusters=k, random_state=seed,
                             batch_size=min(4096, n), n_init=3, max_iter=100)
        km.fit(emb_subset)
        # nearest frame per centroid
        d_mat = euclidean_distances(km.cluster_centers_, emb_subset)
        nearest = d_mat.argmin(axis=1)
        sel = np.unique(nearest)
        if len(sel) < k:
            remaining = np.setdiff1d(np.arange(n), sel)
            rng = np.random.default_rng(seed)
            extra = rng.choice(remaining, size=k - len(sel), replace=False)
            sel = np.concatenate([sel, extra])
        return sel


def allocate_k(bucket_counts: dict, total_k: int):
    """
    Proportional allocation with at-least-one guarantee. Returns dict
    mapping bucket_key -> k_for_bucket. Sum equals total_k (rounding-safe).
    """
    n_total = sum(bucket_counts.values())
    raw = {key: total_k * n / n_total for key, n in bucket_counts.items()}
    # Floor and distribute remainder by largest fractional part
    floored = {k: max(1, int(np.floor(v))) for k, v in raw.items()}
    leftover = total_k - sum(floored.values())
    if leftover > 0:
        fracs = sorted(raw.items(), key=lambda kv: kv[1] - np.floor(kv[1]),
                       reverse=True)
        for key, _ in fracs[:leftover]:
            floored[key] += 1
    elif leftover < 0:
        # Over-allocated due to min-1 floor; trim from largest buckets
        biggest = sorted(floored.items(), key=lambda kv: kv[1], reverse=True)
        for i in range(-leftover):
            key = biggest[i % len(biggest)][0]
            if floored[key] > 1:
                floored[key] -= 1
    # Cap at bucket size
    for key, n in bucket_counts.items():
        if floored[key] > n:
            floored[key] = n
    return floored


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k",         type=int, default=10000,
                   help="Total frame count in the curated set")
    p.add_argument("--emb",       default="data/train_v4_embeddings.npy")
    p.add_argument("--meta",      default="data/train_v4_embedding_meta.json")
    p.add_argument("--jsonl",     default="data/train_v4.jsonl",
                   help="Source JSONL — used to copy full records to output")
    p.add_argument("--out_jsonl", default="data/train_v4_curated.jsonl")
    p.add_argument("--seed",      type=int, default=42)
    args = p.parse_args()

    emb_path   = os.path.join(_project_dir, args.emb)
    meta_path  = os.path.join(_project_dir, args.meta)
    jsonl_path = os.path.join(_project_dir, args.jsonl)
    out_path   = os.path.join(_project_dir, args.out_jsonl)

    print(f"Loading embeddings ← {emb_path}")
    embeddings = np.load(emb_path)
    print(f"  shape={embeddings.shape}  dtype={embeddings.dtype}")

    print(f"Loading metadata   ← {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)
    assert len(meta) == len(embeddings), \
        f"meta/emb length mismatch: {len(meta)} vs {len(embeddings)}"

    print(f"Loading source JSONL ← {jsonl_path}")
    with open(jsonl_path) as f:
        source_lines = f.readlines()
    assert len(source_lines) == len(meta), \
        f"jsonl/meta length mismatch: {len(source_lines)} vs {len(meta)}"

    # ------------------------------------------------------------------
    # Group rows by (label, door_id) buckets
    # ------------------------------------------------------------------
    buckets = defaultdict(list)  # (label, door) -> list of global indices
    for i, m in enumerate(meta):
        buckets[(m["label"], m["door_id"])].append(i)

    label_totals = Counter(m["label"] for m in meta)
    print(f"\nLabel totals — CLOSED: {label_totals['CLOSED']}  OPEN: {label_totals['OPEN']}")

    print(f"\nBucket sizes (label, door):")
    for key in sorted(buckets):
        print(f"  {key[0]:6s}  {key[1]:18s}  {len(buckets[key]):5d}")

    # ------------------------------------------------------------------
    # K allocation: 50/50 across labels, proportional within each label
    # ------------------------------------------------------------------
    k_per_label = args.k // 2
    print(f"\nTarget K = {args.k}  →  CLOSED: {k_per_label}  OPEN: {k_per_label}")

    closed_buckets = {k: len(v) for k, v in buckets.items() if k[0] == "CLOSED"}
    open_buckets   = {k: len(v) for k, v in buckets.items() if k[0] == "OPEN"}
    closed_alloc = allocate_k(closed_buckets, k_per_label)
    open_alloc   = allocate_k(open_buckets,   k_per_label)
    alloc = {**closed_alloc, **open_alloc}

    print(f"\nPer-bucket allocation:")
    for key in sorted(alloc):
        n  = len(buckets[key])
        ka = alloc[key]
        print(f"  {key[0]:6s}  {key[1]:18s}  bucket={n:5d}  k={ka:5d}  "
              f"({100*ka/n:.1f}% kept)")

    # ------------------------------------------------------------------
    # Per-bucket k-means
    # ------------------------------------------------------------------
    print(f"\nRunning per-bucket k-means ...")
    selected_global = []
    for key in sorted(buckets):
        global_idx = np.array(buckets[key])
        emb_sub    = embeddings[global_idx]
        k_for      = alloc[key]
        t0 = time.time()
        local_sel = kmeans_select(emb_sub, k_for, seed=args.seed)
        chosen = global_idx[local_sel]
        selected_global.extend(chosen.tolist())
        print(f"  {key[0]:6s} {key[1]:18s}  bucket={len(global_idx):5d}  "
              f"k={k_for:5d}  → selected {len(chosen):5d}  "
              f"({time.time()-t0:.1f}s)")

    selected_global = sorted(set(selected_global))
    print(f"\nTotal selected: {len(selected_global)} (target {args.k})")

    # ------------------------------------------------------------------
    # Sanity check label balance
    # ------------------------------------------------------------------
    sel_labels = Counter(meta[i]["label"] for i in selected_global)
    sel_doors  = Counter(meta[i]["door_id"] for i in selected_global)
    print(f"\nFinal label balance — CLOSED: {sel_labels['CLOSED']}  OPEN: {sel_labels['OPEN']}")
    print(f"Final door distribution:")
    for d in sorted(sel_doors):
        print(f"  {d:18s}  {sel_doors[d]:5d}")

    # ------------------------------------------------------------------
    # Write curated JSONL
    # ------------------------------------------------------------------
    print(f"\nWriting → {out_path}")
    with open(out_path, "w") as f:
        for i in selected_global:
            f.write(source_lines[i])
        f.flush()
        os.fsync(f.fileno())
    print(f"  {len(selected_global)} lines written")

    # ------------------------------------------------------------------
    # Save selection summary
    # ------------------------------------------------------------------
    summary_path = out_path.replace(".jsonl", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "k_target":          args.k,
            "k_actual":          len(selected_global),
            "source_jsonl":      args.jsonl,
            "embeddings":        args.emb,
            "seed":              args.seed,
            "label_balance":     dict(sel_labels),
            "door_distribution": dict(sel_doors),
            "bucket_allocation": {f"{k[0]}__{k[1]}": v for k, v in alloc.items()},
            "selected_indices":  selected_global,
        }, f, indent=2)
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
