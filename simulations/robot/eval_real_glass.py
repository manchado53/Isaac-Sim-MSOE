#!/usr/bin/env python3
"""
eval_real_glass.py — Evaluate Qwen2-VL-2B on the DH cracked-glass eval set.

Tests sim-to-real transfer on the glass anomaly. The eval set is produced by:
  Stage 1: extract_crack_textures.py  — pulls edge_v1 / v4 / tempered_v5
           crack overlays from the synthetic dataset via |cracked - intact|.
  Stage 2: segment_dh_panels.py       — SAM2 + manual review to mask glass in
           30 Diercks Hall photos.
  Stage 3: composite_cracked_glass.py — overlays the extracted textures onto
           each photo's masked glass region. Produces 30 intact + 90 cracked
           images under data/dh_cracked_eval/.

Iterates data/dh_cracked_eval/{intact,cracked}/ and reports 4-class confusion
(including sim-to-real failure modes where the model predicts a door class on
a glass panel). Prompt is identical to the training JSONL so the fine-tuned
model sees exactly what it was trained on.

Usage (GPU node):
  python simulations/robot/eval_real_glass.py                              # baseline
  python simulations/robot/eval_real_glass.py --adapter data/qwen2vl_door_lora
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

VLM_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
# Must match the training JSONL prompt in data_glass/tanner_vlm_dataset_balanced/train.jsonl
PROMPT = ("What is shown in this image? "
          "Options: door open, door closed, glass cracked, glass intact.")
MAX_NEW_TOKENS = 32

CLASSES = ["DOOR_OPEN", "DOOR_CLOSED", "GLASS_CRACKED", "GLASS_INTACT"]

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))


def discover_samples(root: str):
    """Walk data/dh_cracked_eval/{intact,cracked}/ for test samples.

    Returns list of (rel_path, true_label, material). Material is None for
    intact, else one of {"edge_v1", "v4", "tempered_v5"} parsed from filename.
    """
    samples = []
    for subdir, label in [("intact", "GLASS_INTACT"), ("cracked", "GLASS_CRACKED")]:
        folder = os.path.join(root, subdir)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            mat = None
            if label == "GLASS_CRACKED":
                base = os.path.splitext(fname)[0]
                if "__" in base:
                    mat = base.rsplit("__", 1)[1]
            rel = os.path.join(subdir, fname)
            samples.append((rel, label, mat))
    return samples


def predict(model, processor, image: Image.Image, device: str) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], return_tensors="pt", padding=True,
    ).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        )
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def parse_prediction(response: str) -> str:
    r = response.upper()
    # Bracketed tags first (training format: "State: [GLASS_CRACKED].").
    for cls in CLASSES:
        if f"[{cls}]" in r:
            return cls
    # Fall back to option-text phrasing.
    if "GLASS CRACKED" in r or "GLASS_CRACKED" in r:
        return "GLASS_CRACKED"
    if "GLASS INTACT" in r or "GLASS_INTACT" in r:
        return "GLASS_INTACT"
    if "DOOR OPEN" in r or "DOOR_OPEN" in r:
        return "DOOR_OPEN"
    if "DOOR CLOSED" in r or "DOOR_CLOSED" in r:
        return "DOOR_CLOSED"
    return "UNKNOWN"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter dir — omit for base (no fine-tuning)")
    ap.add_argument("--data-root", default="data/dh_cracked_eval")
    ap.add_argument("--results", default=None,
                    help="Output JSON path (default: <data-root>/real_glass_results[_ft|_baseline].json)")
    args = ap.parse_args()

    data_root = (args.data_root if os.path.isabs(args.data_root)
                 else os.path.join(_project_dir, args.data_root))
    samples = discover_samples(data_root)
    if not samples:
        print(f"ERROR: no samples under {data_root}", file=sys.stderr)
        sys.exit(1)

    mode = "FINE-TUNED" if args.adapter else "BASE"
    print("=" * 60)
    print(f"  Qwen2-VL-2B DH Cracked-Glass Eval — {mode}")
    print("=" * 60)
    print(f"  Data root: {data_root}")
    n_intact = sum(1 for _, lbl, _ in samples if lbl == "GLASS_INTACT")
    n_crack = sum(1 for _, lbl, _ in samples if lbl == "GLASS_CRACKED")
    print(f"  Samples:   {len(samples)}  (intact={n_intact}, cracked={n_crack})")

    adapter_path = None
    if args.adapter:
        p = args.adapter if os.path.isabs(args.adapter) else os.path.join(_project_dir, args.adapter)
        if not os.path.isdir(p):
            print(f"ERROR: adapter not found: {p}", file=sys.stderr)
            sys.exit(1)
        adapter_path = p

    device = "cuda" if torch.cuda.is_available() else "cpu"
    attn = ("flash_attention_2" if torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 8 else "sdpa")
    print(f"  Device: {device}  Attention: {attn}")

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_ID, torch_dtype=torch.float16, device_map="auto",
        attn_implementation=attn,
    )
    if adapter_path:
        from peft import PeftModel
        print(f"  Loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        # Do NOT call merge_and_unload() — crashes with modules_to_save.
    model.eval()
    print(f"  Model loaded in {time.time() - t0:.1f}s\n")

    # Confusion matrix: [true][pred] -> count
    cm = {t: {p: 0 for p in CLASSES + ["UNKNOWN"]} for t in CLASSES}
    per_material = {}  # material -> {total, correct}
    rows = []

    for i, (rel, true, mat) in enumerate(samples, 1):
        abs_path = os.path.join(data_root, rel)
        image = Image.open(abs_path).convert("RGB")
        t_s = time.time()
        resp = predict(model, processor, image, device)
        pred = parse_prediction(resp)
        elapsed = time.time() - t_s

        cm[true][pred] += 1
        if mat is not None:
            pm = per_material.setdefault(mat, {"total": 0, "correct": 0})
            pm["total"] += 1
            if pred == true:
                pm["correct"] += 1

        ok = "CORRECT" if pred == true else "WRONG"
        print(f"  [{i}/{len(samples)}] {rel}  true={true}  pred={pred}  [{ok}]  "
              f"({elapsed:.1f}s)  resp={resp!r}")
        rows.append({
            "path": rel, "true": true, "pred": pred,
            "material": mat, "response": resp, "seconds": round(elapsed, 2),
        })

    # Per-class accuracy
    per_class = {}
    for true in CLASSES:
        total = sum(cm[true].values())
        correct = cm[true][true]
        per_class[true] = {"total": total, "correct": correct,
                           "acc": correct / total if total else 0.0}

    overall_correct = sum(cm[t][t] for t in CLASSES)
    overall_total = sum(sum(cm[t].values()) for t in CLASSES)
    overall_acc = overall_correct / overall_total if overall_total else 0.0

    print("\n" + "=" * 60)
    print(f"  SUMMARY — {mode}")
    print("=" * 60)
    print(f"  Overall: {overall_acc*100:.1f}%  ({overall_correct}/{overall_total})")
    for cls in CLASSES:
        pc = per_class[cls]
        if pc["total"]:
            print(f"    {cls:15s} {pc['acc']*100:5.1f}%  ({pc['correct']}/{pc['total']})")
    if per_material:
        print("\n  Per-material (cracked only):")
        for mat, v in sorted(per_material.items()):
            acc = v["correct"] / v["total"] if v["total"] else 0.0
            print(f"    {mat:15s} {acc*100:5.1f}%  ({v['correct']}/{v['total']})")

    print("\n  Confusion matrix (rows=true, cols=pred):")
    header = ["true\\pred"] + CLASSES + ["UNKNOWN"]
    print("    " + "  ".join(f"{h:>14}" for h in header))
    for t in CLASSES:
        row = [t] + [str(cm[t][p]) for p in CLASSES + ["UNKNOWN"]]
        print("    " + "  ".join(f"{c:>14}" for c in row))

    out_path = args.results or os.path.join(
        data_root,
        f"real_glass_results_{'ft' if adapter_path else 'baseline'}.json",
    )
    with open(out_path, "w") as f:
        json.dump({
            "mode": mode,
            "adapter": adapter_path,
            "prompt": PROMPT,
            "overall": {"correct": overall_correct, "total": overall_total,
                        "acc": overall_acc},
            "per_class": per_class,
            "per_material": per_material,
            "confusion_matrix": cm,
            "samples": rows,
        }, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print(f"\n  Results -> {out_path}")


if __name__ == "__main__":
    main()
