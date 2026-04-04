#!/usr/bin/env python3
"""
eval_baseline.py — Evaluate the BASE Qwen2-VL-2B-Instruct (no fine-tuning)
against the door_labels_static.npz dataset.

Run this BEFORE fine-tuning to get a baseline accuracy. After fine-tuning,
run eval_finetuned.py (or rerun this with --adapter) to compare.

Results are printed to stdout and saved to data/baseline_eval_results.json.

Usage (GPU node required — submit via SLURM or interactive session):
  ~/isaacsim/python.sh simulations/robot/eval_baseline.py
  ~/isaacsim/python.sh simulations/robot/eval_baseline.py --n_samples 100
  ~/isaacsim/python.sh simulations/robot/eval_baseline.py --adapter data/qwen2vl_door_lora
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VLM_MODEL_ID     = "Qwen/Qwen2-VL-2B-Instruct"
CLOSED_THRESHOLD = 5.0          # degrees — matches training threshold
PROMPT           = "Is this door open or closed? Answer with exactly: Door Status: [OPEN]. or Door Status: [CLOSED]."
MAX_NEW_TOKENS   = 24

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset(npz_path: str, n_samples: int, seed: int):
    """
    Load frames from the npz and return a stratified sample.
    Returns list of (frame_rgb_array, true_label_str, angle_deg).
    """
    d       = np.load(npz_path)
    frames  = d["frames"]   # (N, H, W, 3) uint8
    angles  = d["angles"]   # (N,) float32 degrees

    closed_idx = np.where(angles <= CLOSED_THRESHOLD)[0]
    open_idx   = np.where(angles >  CLOSED_THRESHOLD)[0]

    rng = np.random.default_rng(seed)

    # Stratified sample: half CLOSED, half OPEN
    n_each = n_samples // 2
    c_pick = rng.choice(closed_idx, size=min(n_each, len(closed_idx)), replace=False)
    o_pick = rng.choice(open_idx,   size=min(n_each, len(open_idx)),   replace=False)
    indices = np.concatenate([c_pick, o_pick])
    rng.shuffle(indices)

    samples = []
    for i in indices:
        label = "CLOSED" if angles[i] <= CLOSED_THRESHOLD else "OPEN"
        samples.append((frames[i], label, float(angles[i])))
    return samples


def predict(model, processor, frame_array: np.ndarray, device: str) -> str:
    """
    Run one inference pass. Returns the raw model response string.
    """
    image = Image.fromarray(frame_array)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    # Decode only the newly generated tokens (not the prompt)
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    response     = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    response_raw = processor.batch_decode(generated, skip_special_tokens=False)[0].strip()
    print(f"    [RAW] {response_raw!r}")
    return response


def parse_prediction(response: str) -> str:
    """Extract OPEN or CLOSED from the model response. Returns 'UNKNOWN' if neither found."""
    r = response.upper()
    if "[CLOSED]" in r:
        return "CLOSED"
    if "[OPEN]" in r:
        return "OPEN"
    # Looser fallback
    if "CLOSED" in r:
        return "CLOSED"
    if "OPEN" in r:
        return "OPEN"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz",       default="data/door_labels_static.npz",
                        help="Path to dataset npz (relative to project root)")
    parser.add_argument("--n_samples", type=int, default=600,
                        help="Total frames to evaluate (half CLOSED, half OPEN)")
    parser.add_argument("--adapter",   default=None,
                        help="Path to LoRA adapter dir — omit for baseline (no fine-tuning)")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    npz_path = os.path.join(_project_dir, args.npz)
    if not os.path.isfile(npz_path):
        print(f"ERROR: dataset not found: {npz_path}")
        sys.exit(1)

    mode = "FINE-TUNED" if args.adapter else "BASELINE (no fine-tuning)"
    print(f"\n{'='*60}")
    print(f"  Qwen2-VL-2B Door Classification Eval — {mode}")
    print(f"{'='*60}")
    print(f"  Model:    {VLM_MODEL_ID}")
    print(f"  Adapter:  {args.adapter or 'none'}")
    print(f"  Dataset:  {npz_path}")
    print(f"  Samples:  {args.n_samples} (stratified {args.n_samples//2} CLOSED + {args.n_samples//2} OPEN)")
    print()

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    attn_impl = "flash_attention_2" if (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    ) else "sdpa"
    print(f"Device: {device}  |  Attention: {attn_impl}")

    print(f"Loading {VLM_MODEL_ID} ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation=attn_impl,
    )

    if args.adapter:
        from peft import PeftModel
        print(f"Loading LoRA adapter from {args.adapter} ...")
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()   # merge weights for faster inference

    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s\n")

    # ------------------------------------------------------------------
    # Load samples
    # ------------------------------------------------------------------
    samples = load_dataset(npz_path, args.n_samples, args.seed)
    print(f"Evaluating {len(samples)} frames ...\n")

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    results = []
    correct = 0
    per_class = {"CLOSED": {"correct": 0, "total": 0},
                 "OPEN":   {"correct": 0, "total": 0}}

    for i, (frame, true_label, angle_deg) in enumerate(samples):
        t_start  = time.time()
        response = predict(model, processor, frame, device)
        pred     = parse_prediction(response)
        is_correct = (pred == true_label)
        elapsed  = time.time() - t_start

        if is_correct:
            correct += 1
        per_class[true_label]["total"]   += 1
        if is_correct:
            per_class[true_label]["correct"] += 1

        status = "✓" if is_correct else "✗"
        print(f"  [{i+1:3d}/{len(samples)}] {status}  true={true_label:6s} ({angle_deg:5.1f}°)  "
              f"pred={pred:7s}  response: {response!r}  ({elapsed:.1f}s)")

        results.append({
            "index":      i,
            "angle_deg":  angle_deg,
            "true_label": true_label,
            "predicted":  pred,
            "correct":    is_correct,
            "response":   response,
        })

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    overall_acc = correct / len(samples)
    closed_acc  = (per_class["CLOSED"]["correct"] / per_class["CLOSED"]["total"]
                   if per_class["CLOSED"]["total"] > 0 else 0.0)
    open_acc    = (per_class["OPEN"]["correct"] / per_class["OPEN"]["total"]
                   if per_class["OPEN"]["total"] > 0 else 0.0)

    # Confusion matrix counts
    tp = sum(1 for r in results if r["true_label"] == "OPEN"   and r["predicted"] == "OPEN")
    tn = sum(1 for r in results if r["true_label"] == "CLOSED" and r["predicted"] == "CLOSED")
    fp = sum(1 for r in results if r["true_label"] == "CLOSED" and r["predicted"] == "OPEN")
    fn = sum(1 for r in results if r["true_label"] == "OPEN"   and r["predicted"] == "CLOSED")
    unk = sum(1 for r in results if r["predicted"] == "UNKNOWN")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    print(f"\n{'='*60}")
    print(f"  RESULTS — {mode}")
    print(f"{'='*60}")
    print(f"  Overall accuracy : {overall_acc*100:.1f}%  ({correct}/{len(samples)})")
    print(f"  CLOSED accuracy  : {closed_acc*100:.1f}%  ({per_class['CLOSED']['correct']}/{per_class['CLOSED']['total']})")
    print(f"  OPEN accuracy    : {open_acc*100:.1f}%  ({per_class['OPEN']['correct']}/{per_class['OPEN']['total']})")
    print(f"  Precision        : {precision*100:.1f}%")
    print(f"  Recall           : {recall*100:.1f}%")
    print(f"  F1               : {f1:.3f}")
    print(f"  UNKNOWN responses: {unk}")
    print(f"\n  Confusion matrix (OPEN = positive class):")
    print(f"                  Predicted OPEN  Predicted CLOSED")
    print(f"  True OPEN              {tp:3d}               {fn:3d}")
    print(f"  True CLOSED            {fp:3d}               {tn:3d}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    out_path = os.path.join(
        _project_dir, "data",
        "finetuned_eval_results.json" if args.adapter else "baseline_eval_results.json"
    )
    summary = {
        "mode":            mode,
        "model":           VLM_MODEL_ID,
        "adapter":         args.adapter,
        "n_samples":       len(samples),
        "overall_accuracy": overall_acc,
        "closed_accuracy": closed_acc,
        "open_accuracy":   open_acc,
        "precision":       precision,
        "recall":          recall,
        "f1":              f1,
        "unknown_count":   unk,
        "confusion":       {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "per_sample":      results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
