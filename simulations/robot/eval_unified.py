#!/usr/bin/env python3
"""
eval_unified.py — 4-class evaluation for the unified door+glass adapter.

Runs on Qwen3-VL-2B-Instruct (matches finetune_qwen3vl.py / the unified
adapter `data/qwen3vl_unified_lora_v4`). Extends the single-domain
eval_baseline.py pattern to the 4-class label set:
DOOR_OPEN, DOOR_CLOSED, GLASS_CRACKED, GLASS_INTACT.

Prints overall accuracy, per-class accuracy, per-domain accuracy
(door vs glass), and a 4x4 confusion matrix. Also writes a per-sample
JSON so you can slice later.

Usage:
  ~/.conda/envs/vlm/bin/python simulations/robot/eval_unified.py \
      --test_jsonl data/val_unified.jsonl \
      --adapter   data/qwen3vl_unified_lora_v4 \
      --output    data/eval_unified_val.json
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

VLM_MODEL_ID   = "Qwen/Qwen3-VL-2B-Instruct"
MAX_NEW_TOKENS = 24

PROMPT = (
    "What is shown in this image? Options: door open, door closed, "
    "glass cracked, glass intact. Answer with exactly one of: "
    "State: [DOOR_OPEN]. | State: [DOOR_CLOSED]. | State: [GLASS_CRACKED]. | State: [GLASS_INTACT]."
)

LABELS = ["DOOR_OPEN", "DOOR_CLOSED", "GLASS_CRACKED", "GLASS_INTACT"]
DOOR_LABELS  = {"DOOR_OPEN", "DOOR_CLOSED"}
GLASS_LABELS = {"GLASS_CRACKED", "GLASS_INTACT"}

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))


def load_test_jsonl(jsonl_path: str):
    data_dir = os.path.dirname(jsonl_path)
    samples  = []
    with open(jsonl_path) as f:
        for line in f:
            rec   = json.loads(line)
            ans   = rec["conversations"][1]["content"].upper()
            label = next((L for L in LABELS if f"[{L}]" in ans), None)
            if label is None:
                raise ValueError(f"Could not parse label from: {ans!r}")
            img_path = os.path.join(data_dir, rec["image"])
            samples.append((img_path, label, rec["image"]))
    return samples


def parse_prediction(response: str) -> str:
    r = response.upper()
    for L in LABELS:
        if f"[{L}]" in r:
            return L
    # looser fallback
    for L in LABELS:
        if L in r:
            return L
    return "UNKNOWN"


def predict(model, processor, img_path, device):
    image = Image.open(img_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": PROMPT},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    generated = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_jsonl", default="data/val_unified.jsonl")
    p.add_argument("--adapter",    default=None)
    p.add_argument("--output",     default=None,
                   help="Output JSON path (relative to project root). "
                        "Defaults to data/eval_unified_{mode}.json")
    args = p.parse_args()

    jsonl_path = (args.test_jsonl if os.path.isabs(args.test_jsonl)
                  else os.path.join(_project_dir, args.test_jsonl))
    if not os.path.isfile(jsonl_path):
        print(f"ERROR: {jsonl_path} not found"); sys.exit(1)

    mode = "FINE-TUNED" if args.adapter else "BASELINE"
    print(f"\n=== Qwen2-VL 4-Class Eval — {mode} ===")
    print(f"Model:   {VLM_MODEL_ID}")
    print(f"Adapter: {args.adapter or 'none'}")
    print(f"Val:     {jsonl_path}\n")

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    attn_impl = "flash_attention_2" if (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    ) else "sdpa"
    print(f"Device: {device}  Attn: {attn_impl}")

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_ID, torch_dtype=torch.float16,
        device_map="auto", attn_implementation=attn_impl,
    )
    if args.adapter:
        from peft import PeftModel
        adapter_path = (args.adapter if os.path.isabs(args.adapter)
                        else os.path.join(_project_dir, args.adapter))
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s\n")

    samples = load_test_jsonl(jsonl_path)
    cnt = Counter(lbl for _, lbl, _ in samples)
    print(f"Loaded {len(samples)} samples:  "
          + "  ".join(f"{L}={cnt[L]}" for L in LABELS))
    print("Running inference ...\n")

    results = []
    conf = {t: {p: 0 for p in LABELS + ["UNKNOWN"]} for t in LABELS}
    for i, (img_path, true_label, rel_img) in enumerate(samples):
        t_s = time.time()
        response = predict(model, processor, img_path, device)
        pred = parse_prediction(response)
        ok = (pred == true_label)
        conf[true_label][pred] += 1

        status = "✓" if ok else "✗"
        if i % 50 == 0 or not ok:
            print(f"  [{i+1:4d}/{len(samples)}] {status} true={true_label:<14s} "
                  f"pred={pred:<14s} ({time.time()-t_s:.1f}s)  {response!r}")

        results.append({
            "index": i, "image": rel_img,
            "true_label": true_label, "predicted": pred,
            "correct": ok, "response": response,
        })

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total    = len(samples)
    overall  = sum(1 for r in results if r["correct"]) / total
    per_cls  = {L: (conf[L][L] / max(1, sum(conf[L].values()))) for L in LABELS}
    door_s   = [r for r in results if r["true_label"] in DOOR_LABELS]
    glass_s  = [r for r in results if r["true_label"] in GLASS_LABELS]
    door_acc  = sum(r["correct"] for r in door_s)  / max(1, len(door_s))
    glass_acc = sum(r["correct"] for r in glass_s) / max(1, len(glass_s))

    print(f"\n=== RESULTS — {mode} ===")
    print(f"Overall         : {overall*100:.2f}%  ({sum(r['correct'] for r in results)}/{total})")
    print(f"Door  accuracy  : {door_acc*100:.2f}%  ({sum(r['correct'] for r in door_s)}/{len(door_s)})")
    print(f"Glass accuracy  : {glass_acc*100:.2f}%  ({sum(r['correct'] for r in glass_s)}/{len(glass_s)})")
    print("Per-class:")
    for L in LABELS:
        print(f"  {L:<14s} {per_cls[L]*100:6.2f}%  ({conf[L][L]}/{sum(conf[L].values())})")

    print("\nConfusion (rows=true, cols=pred):")
    header = ["true \\ pred"] + [L[:10] for L in LABELS] + ["UNK"]
    print("  " + "  ".join(f"{h:>11s}" for h in header))
    for L in LABELS:
        row = [conf[L][p] for p in LABELS] + [conf[L]["UNKNOWN"]]
        print("  " + f"{L:>11s}  " + "  ".join(f"{v:>11d}" for v in row))

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if args.output:
        out_path = (args.output if os.path.isabs(args.output)
                    else os.path.join(_project_dir, args.output))
    else:
        out_path = os.path.join(
            _project_dir, "data",
            f"eval_unified_{'finetuned' if args.adapter else 'baseline'}.json")
    summary = {
        "mode":            mode, "model": VLM_MODEL_ID, "adapter": args.adapter,
        "test_jsonl":      args.test_jsonl, "n_samples": total,
        "overall":         overall,
        "door_accuracy":   door_acc, "glass_accuracy": glass_acc,
        "per_class":       per_cls,
        "confusion":       {t: conf[t] for t in LABELS},
        "per_sample":      results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
