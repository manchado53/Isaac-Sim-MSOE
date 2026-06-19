#!/usr/bin/env python3
"""
curate_diversity.py -- Stratified k-means diversity sampling.

Port of origin/tanner_vlm:simulations/robot/curate_diversity.py (commit
207c0ba). Bucket logic is label-agnostic; only the hardcoded label
literals changed:
  CLOSED -> intact
  OPEN   -> cracked
  door_id -> panel_idx

Reads DINOv2 embeddings produced by embed_train_frames.py and selects K
frames that maximize visual coverage while preserving exact 50/50
intact/cracked balance.

Stratification: clustering runs INDEPENDENTLY within each
(label, panel_idx) bucket. K is split:
  - 50% to intact, 50% to cracked         (label balance, hard constraint)
  - within each label, proportional to per-panel frame count
For each bucket we run k-means with k = bucket allocation, then keep the
frame nearest each centroid.

Output: <dataset>/train_curated.jsonl -- drop-in for labels_train.jsonl.

Usage:
  python curate_diversity.py --k 10000 --jsonl datasets/cracked_glass/<ts>/labels_train.jsonl
  python curate_diversity.py --k 5000  --jsonl <path>
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict, Counter

import numpy as np


def kmeans_select(emb_subset: np.ndarray, k: int, seed: int):
    """
    Run k-means on emb_subset (n, d), return indices (into emb_subset) of
    the frames closest to each of the k centroids. Uses faiss if
    available, falls back to sklearn MiniBatchKMeans.
    """
    n, d = emb_subset.shape
    if k >= n:
        # Trivial: asked for more clusters than points -> keep them all.
        return np.arange(n)

    try:
        import faiss
        km = faiss.Kmeans(d=d, k=k, niter=25, seed=seed, verbose=False, gpu=False)
        km.train(emb_subset.astype(np.float32))
        index = faiss.IndexFlatL2(d)
        index.add(emb_subset.astype(np.float32))
        _, nearest = index.search(km.centroids, 1)
        sel = np.unique(nearest.flatten())
        # If two centroids picked the same frame, top up with random unselected.
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
        d_mat   = euclidean_distances(km.cluster_centers_, emb_subset)
        nearest = d_mat.argmin(axis=1)
        sel     = np.unique(nearest)
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
    floored = {k: max(1, int(np.floor(v))) for k, v in raw.items()}
    leftover = total_k - sum(floored.values())
    if leftover > 0:
        fracs = sorted(raw.items(), key=lambda kv: kv[1] - np.floor(kv[1]),
                       reverse=True)
        for key, _ in fracs[:leftover]:
            floored[key] += 1
    elif leftover < 0:
        biggest = sorted(floored.items(), key=lambda kv: kv[1], reverse=True)
        for i in range(-leftover):
            key = biggest[i % len(biggest)][0]
            if floored[key] > 1:
                floored[key] -= 1
    for key, n in bucket_counts.items():
        if floored[key] > n:
            floored[key] = n
    return floored


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k",         type=int, default=10000,
                   help="Total frame count in the curated set.")
    p.add_argument("--jsonl",     required=True,
                   help="Source labels_train.jsonl -- used to copy full records to output.")
    p.add_argument("--emb",       default=None,
                   help="Default: <dataset>/train_embeddings.npy next to JSONL.")
    p.add_argument("--meta",      default=None,
                   help="Default: <dataset>/train_embedding_meta.json next to JSONL.")
    p.add_argument("--out_jsonl", default=None,
                   help="Default: <dataset>/train_curated.jsonl next to JSONL.")
    p.add_argument("--seed",      type=int, default=42)
    args = p.parse_args()

    jsonl_path = os.path.abspath(args.jsonl)
    data_dir   = os.path.dirname(jsonl_path)
    emb_path   = args.emb       or os.path.join(data_dir, "train_embeddings.npy")
    meta_path  = args.meta      or os.path.join(data_dir, "train_embedding_meta.json")
    out_path   = args.out_jsonl or os.path.join(data_dir, "train_curated.jsonl")

    print(f"Loading embeddings <- {emb_path}")
    embeddings = np.load(emb_path)
    print(f"  shape={embeddings.shape}  dtype={embeddings.dtype}")

    print(f"Loading metadata   <- {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)
    assert len(meta) == len(embeddings), \
        f"meta/emb length mismatch: {len(meta)} vs {len(embeddings)}"

    print(f"Loading source JSONL <- {jsonl_path}")
    with open(jsonl_path) as f:
        source_lines = f.readlines()
    assert len(source_lines) == len(meta), \
        f"jsonl/meta length mismatch: {len(source_lines)} vs {len(meta)}"

    # ------------------------------------------------------------------
    # Group rows by (label, panel_idx) buckets
    # ------------------------------------------------------------------
    buckets = defaultdict(list)  # (label, panel_idx) -> list of global indices
    for i, m in enumerate(meta):
        buckets[(m["label"], m["panel_idx"])].append(i)

    label_totals = Counter(m["label"] for m in meta)
    print(f"\nLabel totals -- intact: {label_totals['intact']}  cracked: {label_totals['cracked']}")

    print(f"\nBucket sizes (label, panel_idx):")
    for key in sorted(buckets):
        print(f"  {key[0]:8s}  pan{key[1]:<3d}  {len(buckets[key]):5d}")

    # ------------------------------------------------------------------
    # K allocation: 50/50 across labels, proportional within each label
    # ------------------------------------------------------------------
    k_per_label = args.k // 2
    print(f"\nTarget K = {args.k}  ->  intact: {k_per_label}  cracked: {k_per_label}")

    intact_buckets  = {k: len(v) for k, v in buckets.items() if k[0] == "intact"}
    cracked_buckets = {k: len(v) for k, v in buckets.items() if k[0] == "cracked"}
    intact_alloc    = allocate_k(intact_buckets,  k_per_label)
    cracked_alloc   = allocate_k(cracked_buckets, k_per_label)
    alloc = {**intact_alloc, **cracked_alloc}

    print(f"\nPer-bucket allocation:")
    for key in sorted(alloc):
        n  = len(buckets[key])
        ka = alloc[key]
        print(f"  {key[0]:8s}  pan{key[1]:<3d}  bucket={n:5d}  k={ka:5d}  "
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
        print(f"  {key[0]:8s} pan{key[1]:<3d}  bucket={len(global_idx):5d}  "
              f"k={k_for:5d}  -> selected {len(chosen):5d}  "
              f"({time.time()-t0:.1f}s)")

    selected_global = sorted(set(selected_global))
    print(f"\nTotal selected: {len(selected_global)} (target {args.k})")

    # ------------------------------------------------------------------
    # Sanity check label balance
    # ------------------------------------------------------------------
    sel_labels = Counter(meta[i]["label"]     for i in selected_global)
    sel_panels = Counter(meta[i]["panel_idx"] for i in selected_global)
    print(f"\nFinal label balance -- intact: {sel_labels['intact']}  cracked: {sel_labels['cracked']}")
    print(f"Final panel distribution:")
    for p in sorted(sel_panels):
        print(f"  pan{p:<3d}  {sel_panels[p]:5d}")

    # ------------------------------------------------------------------
    # Write curated JSONL
    # ------------------------------------------------------------------
    print(f"\nWriting -> {out_path}")
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
            "k_target":           args.k,
            "k_actual":           len(selected_global),
            "source_jsonl":       args.jsonl,
            "embeddings":         args.emb or os.path.basename(emb_path),
            "seed":               args.seed,
            "label_balance":      dict(sel_labels),
            "panel_distribution": dict(sel_panels),
            "bucket_allocation":  {f"{k[0]}__pan{k[1]}": v for k, v in alloc.items()},
            "selected_indices":   selected_global,
        }, f, indent=2)
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
