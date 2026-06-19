#!/usr/bin/env python3
"""
eval_baseline.py — Evaluate Qwen2-VL-2B-Instruct on the held-out test split.

Reads data/test.jsonl (produced by prepare_finetune_data.py — 15% held-out set,
270 frames, never seen during training or val-based hyperparameter selection).

Run without --adapter for the zero-shot baseline, or with --adapter for the
fine-tuned model. Both report identical metrics so results are directly comparable.

Results are printed to stdout and saved to:
  data/baseline_eval_results.json   (no --adapter)
  data/finetuned_eval_results.json  (with --adapter)

Usage (GPU node required — submit via SLURM or interactive session):
  ~/isaacsim/python.sh simulations/robot/eval_baseline.py
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

def load_test_jsonl(jsonl_path: str):
    """
    Load the held-out test split from data/test.jsonl.
    Each record has an image path (relative to the data/ dir) and a label
    derived from the assistant turn ("Door Status: [CLOSED]." or "[OPEN].").
    Returns list of (frame_rgb_array, true_label_str, image_path).
    """
    data_dir = os.path.dirname(jsonl_path)
    samples  = []
    with open(jsonl_path) as f:
        for line in f:
            rec   = json.loads(line)
            # Label is in the assistant conversation turn
            answer = rec["conversations"][1]["content"]
            label  = "CLOSED" if "[CLOSED]" in answer.upper() else "OPEN"
            # Image path is relative to data/ directory
            img_path = os.path.join(data_dir, rec["image"])
            img      = Image.open(img_path).convert("RGB")
            frame    = np.array(img, dtype=np.uint8)
            samples.append((frame, label, rec["image"]))
    return samples


def build_decision_token_sets(tokenizer):
    """
    Return (closed_ids, open_ids) — the set of all vocab token IDs that
    could start or contain the words 'CLOSED' or 'OPEN' (case-insensitive).
    Used at inference time to read P([CLOSED]) vs P([OPEN]) from generation
    scores.

    Matches both:
      1. Full substring — token text contains 'CLOSED' or 'OPEN'
      2. Prefix — token text IS a prefix of 'CLOSED' or 'OPEN' with len >= 3
         (catches cases where Qwen BPE splits 'CLOSED' into e.g. 'CLOSE'+'D')

    One-time cost (~1-2s) over the full vocab at startup.
    """
    closed_prefixes = {"CLO", "CLOS", "CLOSE", "CLOSED"}
    open_prefixes   = {"OPE", "OPEN"}

    closed_ids, open_ids = set(), set()
    for tok_id in range(len(tokenizer)):
        s = tokenizer.decode([tok_id]).strip().upper()
        if not s:
            continue
        if "CLOSED" in s or s in closed_prefixes:
            closed_ids.add(tok_id)
        elif "OPEN" in s or s in open_prefixes:
            open_ids.add(tok_id)
    return closed_ids, open_ids


def predict(model, processor, frame_array: np.ndarray, device: str,
            closed_token_ids: set, open_token_ids: set):
    """
    Run one inference pass. Returns (response_str, p_closed, p_open).

    p_closed / p_open are the probabilities the model assigned to
    CLOSED-like and OPEN-like tokens at the first generation step that
    selected one of them. Either may be None if no decision token was
    produced within MAX_NEW_TOKENS.
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
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    # Decode only the newly generated tokens (not the prompt)
    generated = out.sequences[:, inputs["input_ids"].shape[1]:]
    response     = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    response_raw = processor.batch_decode(generated, skip_special_tokens=False)[0].strip()
    print(f"    [RAW] {response_raw!r}")

    # Walk generated tokens, find the first step whose chosen token is
    # CLOSED- or OPEN-like, then compute p_closed / p_open at that step.
    p_closed, p_open = None, None
    for step, tok_id in enumerate(generated[0].tolist()):
        if tok_id in closed_token_ids or tok_id in open_token_ids:
            probs = torch.softmax(out.scores[step][0].float(), dim=-1)
            p_closed = float(probs[list(closed_token_ids)].sum().item())
            p_open   = float(probs[list(open_token_ids)].sum().item())
            break

    return response, p_closed, p_open


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
    parser.add_argument("--test_jsonl", default="data/test.jsonl",
                        help="Path to held-out test JSONL (relative to project root)")
    parser.add_argument("--adapter",    default=None,
                        help="Path to LoRA adapter dir — omit for baseline (no fine-tuning)")
    parser.add_argument("--output",     default=None,
                        help="Output JSON path (relative to project root). "
                             "Defaults to finetuned_eval_results_v2.json / baseline_eval_results_v2.json.")
    args = parser.parse_args()

    jsonl_path = os.path.join(_project_dir, args.test_jsonl)
    if not os.path.isfile(jsonl_path):
        print(f"ERROR: test split not found: {jsonl_path}")
        print("  Run prepare_finetune_data.py first to generate the 70/15/15 splits.")
        sys.exit(1)

    mode = "FINE-TUNED" if args.adapter else "BASELINE (no fine-tuning)"
    print(f"\n{'='*60}")
    print(f"  Qwen2-VL-2B Door Classification Eval — {mode}")
    print(f"{'='*60}")
    print(f"  Model:    {VLM_MODEL_ID}")
    print(f"  Adapter:  {args.adapter or 'none'}")
    print(f"  Test set: {jsonl_path}")
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
        # Do NOT call merge_and_unload() — crashes with modules_to_save adapters
        # in peft 0.18.1. PeftModel inference is identical in accuracy.

    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s\n")

    # ------------------------------------------------------------------
    # Build CLOSED/OPEN token ID sets for confidence extraction
    # ------------------------------------------------------------------
    print("Building decision token sets (CLOSED / OPEN) ...")
    t0 = time.time()
    closed_token_ids, open_token_ids = build_decision_token_sets(processor.tokenizer)
    print(f"  CLOSED-like tokens: {len(closed_token_ids)}  "
          f"OPEN-like tokens: {len(open_token_ids)}  "
          f"({time.time()-t0:.1f}s)\n")

    # ------------------------------------------------------------------
    # Load test split
    # ------------------------------------------------------------------
    print(f"Loading test split from {jsonl_path} ...")
    samples = load_test_jsonl(jsonl_path)
    n_closed = sum(1 for _, lbl, _ in samples if lbl == "CLOSED")
    n_open   = sum(1 for _, lbl, _ in samples if lbl == "OPEN")
    print(f"  {len(samples)} frames — CLOSED: {n_closed}, OPEN: {n_open}")
    print(f"Evaluating {len(samples)} frames ...\n")

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    results = []
    correct = 0
    per_class = {"CLOSED": {"correct": 0, "total": 0},
                 "OPEN":   {"correct": 0, "total": 0}}

    for i, (frame, true_label, img_path) in enumerate(samples):
        t_start  = time.time()
        response, p_closed, p_open = predict(
            model, processor, frame, device,
            closed_token_ids, open_token_ids,
        )
        pred     = parse_prediction(response)
        is_correct = (pred == true_label)
        elapsed  = time.time() - t_start

        # Confidence in the chosen class — max(p_closed, p_open) when both are
        # present. None if no decision token was generated.
        if p_closed is not None and p_open is not None:
            confidence = max(p_closed, p_open)
        else:
            confidence = None

        if is_correct:
            correct += 1
        per_class[true_label]["total"]   += 1
        if is_correct:
            per_class[true_label]["correct"] += 1

        status = "✓" if is_correct else "✗"
        conf_str = f"{confidence:.3f}" if confidence is not None else "  -  "
        print(f"  [{i+1:3d}/{len(samples)}] {status}  true={true_label:6s}  "
              f"pred={pred:7s}  conf={conf_str}  response: {response!r}  ({elapsed:.1f}s)")

        results.append({
            "index":      i,
            "image":      img_path,
            "true_label": true_label,
            "predicted":  pred,
            "correct":    is_correct,
            "response":   response,
            "p_closed":   p_closed,
            "p_open":     p_open,
            "confidence": confidence,
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
    if args.output:
        out_path = (args.output if os.path.isabs(args.output)
                    else os.path.join(_project_dir, args.output))
    else:
        out_path = os.path.join(
            _project_dir, "data",
            "finetuned_eval_results_v2.json" if args.adapter else "baseline_eval_results_v2.json"
        )
    summary = {
        "mode":             mode,
        "model":            VLM_MODEL_ID,
        "adapter":          args.adapter,
        "test_jsonl":       args.test_jsonl,
        "n_samples":        len(samples),
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
