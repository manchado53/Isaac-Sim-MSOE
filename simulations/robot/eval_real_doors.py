#!/usr/bin/env python3
"""
eval_real_doors.py — Evaluate Qwen2-VL-2B-Instruct on real door images.

Tests sim-to-real transfer: does the model (base or fine-tuned) generalize to
real photographs, or did fine-tuning overfit to the digital twin render style?

Images auto-discovered from:
  real_doors/closed/   — CLOSED ground truth
  real_doors/open/     — OPEN ground truth

Usage (GPU node required):
  # Base model (no fine-tuning)
  ~/isaacsim/python.sh simulations/robot/eval_real_doors.py

  # Fine-tuned model
  ~/isaacsim/python.sh simulations/robot/eval_real_doors.py --adapter data/qwen2vl_door_lora
"""

import argparse
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

VLM_MODEL_ID   = "Qwen/Qwen2-VL-2B-Instruct"
PROMPT         = "Is this door open or closed? Answer with exactly: Door Status: [OPEN]. or Door Status: [CLOSED]."
MAX_NEW_TOKENS = 24

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))

def _discover_samples():
    samples = []
    for label, subdir in [("CLOSED", "real_doors/closed"), ("OPEN", "real_doors/open")]:
        folder = os.path.join(_project_dir, subdir)
        if os.path.isdir(folder):
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith(".png"):
                    samples.append((os.path.join(subdir, fname), label))
    return samples

SAMPLES = _discover_samples()


# ---------------------------------------------------------------------------
# Helpers (same pattern as eval_baseline.py)
# ---------------------------------------------------------------------------

def predict(model, processor, frame_array: np.ndarray, device: str) -> str:
    """Run one inference pass. Returns the raw model response string."""
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
    parser.add_argument("--adapter", default=None,
                        help="Path to LoRA adapter dir — omit for base model (no fine-tuning)")
    args = parser.parse_args()

    mode = "FINE-TUNED" if args.adapter else "BASE MODEL (no fine-tuning)"
    print(f"\n{'='*60}")
    print(f"  Qwen2-VL-2B Real Door Eval — {mode}")
    print(f"{'='*60}")
    print(f"  Model:   {VLM_MODEL_ID}")
    print(f"  Adapter: {args.adapter or 'none'}")
    n_closed = sum(1 for _, lbl in SAMPLES if lbl == "CLOSED")
    n_open   = sum(1 for _, lbl in SAMPLES if lbl == "OPEN")
    print(f"  Images:  {len(SAMPLES)} real door photos ({n_closed} CLOSED, {n_open} OPEN)")
    print()

    # Verify all image paths exist before loading the model
    missing = []
    for rel_path, _ in SAMPLES:
        abs_path = os.path.join(_project_dir, rel_path)
        if not os.path.isfile(abs_path):
            missing.append(abs_path)
    if missing:
        print("ERROR: Missing image files:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    if args.adapter:
        adapter_path = os.path.join(_project_dir, args.adapter) if not os.path.isabs(args.adapter) else args.adapter
        if not os.path.isdir(adapter_path):
            print(f"ERROR: Adapter not found: {adapter_path}")
            sys.exit(1)
    else:
        adapter_path = None

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    device    = "cuda" if torch.cuda.is_available() else "cpu"
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

    if adapter_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter from {adapter_path} ...")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s\n")

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    correct  = 0
    per_class = {"CLOSED": {"correct": 0, "total": 0},
                 "OPEN":   {"correct": 0, "total": 0}}

    for i, (rel_path, true_label) in enumerate(SAMPLES):
        abs_path = os.path.join(_project_dir, rel_path)
        filename = os.path.basename(rel_path)

        img   = Image.open(abs_path).convert("RGB")
        frame = np.array(img, dtype=np.uint8)

        t_start  = time.time()
        response = predict(model, processor, frame, device)
        pred     = parse_prediction(response)
        elapsed  = time.time() - t_start

        is_correct = (pred == true_label)
        if is_correct:
            correct += 1
        per_class[true_label]["total"]   += 1
        if is_correct:
            per_class[true_label]["correct"] += 1

        status = "CORRECT" if is_correct else "WRONG"
        print(f"  [{i+1}/{len(SAMPLES)}] {filename}")
        print(f"         true={true_label}  pred={pred}  [{status}]  ({elapsed:.1f}s)")
        print(f"         response: {response!r}")
        print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    overall_acc = correct / len(SAMPLES)
    closed_acc  = (per_class["CLOSED"]["correct"] / per_class["CLOSED"]["total"]
                   if per_class["CLOSED"]["total"] > 0 else 0.0)
    open_acc    = (per_class["OPEN"]["correct"] / per_class["OPEN"]["total"]
                   if per_class["OPEN"]["total"] > 0 else 0.0)

    print(f"{'='*60}")
    print(f"  SUMMARY — {mode} on real door images")
    print(f"{'='*60}")
    print(f"  Overall accuracy : {overall_acc*100:.1f}%  ({correct}/{len(SAMPLES)})")
    print(f"  CLOSED accuracy  : {closed_acc*100:.1f}%  ({per_class['CLOSED']['correct']}/{per_class['CLOSED']['total']})")
    print(f"  OPEN accuracy    : {open_acc*100:.1f}%  ({per_class['OPEN']['correct']}/{per_class['OPEN']['total']})")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
