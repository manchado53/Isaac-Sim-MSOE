#!/usr/bin/env python3
"""
eval_vlm_baseline.py — Zero-shot VLM comparison runner.

Generalises simulations/robot/eval_baseline.py so the same inference +
confidence-extraction logic can be run against any of the candidate base
models in the VLM comparison (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, SmolVLM2,
InternVL3). Only the loader + preprocessor wiring changes per model —
the prompt, parser, confidence extraction, and output schema are identical
to the reference run stored at data/baseline_eval_results.json.

Usage (submitted via submit_vlm_comparison.sh on the dgx partition):
  python eval_vlm_baseline.py \
      --model Qwen/Qwen2.5-VL-3B-Instruct \
      --model_class qwen2_5_vl \
      --val_jsonl data/val.jsonl \
      --output data/baseline_eval_qwen2_5_vl_3b.json
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
from transformers import AutoProcessor, AutoModelForImageTextToText

# ---------------------------------------------------------------------------
# Config — kept identical to eval_baseline.py
# ---------------------------------------------------------------------------

DEFAULT_PROMPT = "Is this door open or closed? Answer with exactly: Door Status: [OPEN]. or Door Status: [CLOSED]."
MAX_NEW_TOKENS = 24

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, "..", ".."))


# ---------------------------------------------------------------------------
# Loader registry — one class per model family. The inference loop below is
# the same for every entry; only the HF class differs.
# ---------------------------------------------------------------------------

def _load_qwen2_vl(model_id, dtype, attn_impl):
    from transformers import Qwen2VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto", attn_implementation=attn_impl,
    )
    return model, processor


def _load_qwen2_5_vl(model_id, dtype, attn_impl):
    from transformers import Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto", attn_implementation=attn_impl,
    )
    return model, processor


def _load_qwen3_vl(model_id, dtype, attn_impl):
    from transformers import Qwen3VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto", attn_implementation=attn_impl,
    )
    return model, processor


def _load_auto(model_id, dtype, attn_impl, trust_remote_code=False):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto",
        attn_implementation=attn_impl, trust_remote_code=trust_remote_code,
    )
    return model, processor


def _load_smolvlm2(model_id, dtype, attn_impl):
    return _load_auto(model_id, dtype, attn_impl, trust_remote_code=False)


def _load_internvl3(model_id, dtype, attn_impl):
    # Use the HF-native port (e.g. OpenGVLab/InternVL3-2B-hf), which ships an
    # InternVLConfig → AutoModelForImageTextToText maps natively. The original
    # OpenGVLab/InternVL3-2B repo uses its own InternVLChatConfig + .chat() API
    # and is not compatible with this generic runner.
    return _load_auto(model_id, dtype, attn_impl, trust_remote_code=False)


LOADERS = {
    "qwen2_vl":   _load_qwen2_vl,
    "qwen2_5_vl": _load_qwen2_5_vl,
    "qwen3_vl":   _load_qwen3_vl,
    "smolvlm2":   _load_smolvlm2,
    "internvl3":  _load_internvl3,
    "auto":       _load_auto,
}


# ---------------------------------------------------------------------------
# Helpers (copied verbatim from eval_baseline.py so behaviour is identical)
# ---------------------------------------------------------------------------

def load_val_jsonl(jsonl_path: str):
    """
    Load records from a val/test JSONL. Each line has {image, conversations};
    label is derived from the assistant turn ([CLOSED] / [OPEN]). Image paths
    are relative to the directory containing the JSONL.
    """
    data_dir = os.path.dirname(jsonl_path)
    samples  = []
    with open(jsonl_path) as f:
        for line in f:
            rec    = json.loads(line)
            answer = rec["conversations"][1]["content"].upper()
            label  = "CLOSED" if ("[CLOSED]" in answer or "[DOOR_CLOSED]" in answer) else "OPEN"
            img_path = os.path.join(data_dir, rec["image"])
            img      = Image.open(img_path).convert("RGB")
            frame    = np.array(img, dtype=np.uint8)
            samples.append((frame, label, rec["image"]))
    return samples


def build_decision_token_sets(tokenizer):
    """
    Return (closed_ids, open_ids) — the set of vocab token IDs that could
    start or contain 'CLOSED' or 'OPEN' (case-insensitive). Handles BPE /
    SentencePiece subword splits (e.g. 'CLOSE'+'D', '▁OPEN'). One-time cost.
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
            closed_token_ids: set, open_token_ids: set, prompt_text: str):
    """One inference pass. Returns (response_str, p_closed, p_open)."""
    image = Image.fromarray(frame_array)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt_text},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], return_tensors="pt", padding=True,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated    = out.sequences[:, inputs["input_ids"].shape[1]:]
    response     = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    p_closed, p_open = None, None
    for step, tok_id in enumerate(generated[0].tolist()):
        if tok_id in closed_token_ids or tok_id in open_token_ids:
            probs = torch.softmax(out.scores[step][0].float(), dim=-1)
            p_closed = float(probs[list(closed_token_ids)].sum().item())
            p_open   = float(probs[list(open_token_ids)].sum().item())
            break

    return response, p_closed, p_open


def parse_prediction(response: str) -> str:
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


def slugify(model_id: str) -> str:
    s = model_id.split("/")[-1].lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True,
                        help="HuggingFace model ID (e.g. Qwen/Qwen2.5-VL-3B-Instruct)")
    parser.add_argument("--model_class", default="auto",
                        choices=list(LOADERS.keys()),
                        help="Loader to use; 'auto' uses AutoModelForImageTextToText")
    parser.add_argument("--val_jsonl",   default="data/val.jsonl",
                        help="Validation JSONL path (relative to project root)")
    parser.add_argument("--output",      default=None,
                        help="Output JSON path; defaults to data/baseline_eval_<slug>.json")
    parser.add_argument("--adapter",     default=None,
                        help="Optional LoRA adapter dir (e.g. data/qwen3vl_door_lora_v4_curated). "
                             "Loaded with PeftModel; do NOT call merge_and_unload — incompatible "
                             "with modules_to_save adapters in peft 0.18.1.")
    parser.add_argument("--limit",       type=int, default=None,
                        help="Evaluate only the first N samples (debug / sanity check)")
    parser.add_argument("--prompt",      default=None,
                        help="Override the inference prompt (default: door-only "
                             "Door Status: [OPEN/CLOSED] format). Use this to match "
                             "the training prompt of adapters trained on a different "
                             "format (e.g. unified 4-class).")
    args = parser.parse_args()
    prompt_text = args.prompt if args.prompt else DEFAULT_PROMPT

    val_path = os.path.join(_project_dir, args.val_jsonl)
    if not os.path.isfile(val_path):
        print(f"ERROR: val split not found: {val_path}")
        sys.exit(1)

    out_path = args.output or os.path.join(
        _project_dir, "data", f"baseline_eval_{slugify(args.model)}.json"
    )

    print(f"\n{'='*60}")
    print(f"  Zero-shot VLM Eval — {args.model}")
    print(f"{'='*60}")
    print(f"  Loader:   {args.model_class}")
    print(f"  Val set:  {val_path}")
    print(f"  Output:   {out_path}")
    print(f"  Prompt:   {prompt_text!r}")
    print()

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    attn_impl = "flash_attention_2" if (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    ) else "sdpa"
    dtype     = torch.float16
    print(f"Device: {device}  |  Attention: {attn_impl}  |  dtype: {dtype}")

    loader = LOADERS[args.model_class]
    print(f"Loading {args.model} via {loader.__name__} ...")
    t0 = time.time()
    model, processor = loader(args.model, dtype, attn_impl)

    if args.adapter:
        from peft import PeftModel
        adapter_path = (args.adapter if os.path.isabs(args.adapter)
                        else os.path.join(_project_dir, args.adapter))
        print(f"Loading LoRA adapter from {adapter_path} ...")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s\n")

    # Tokenizer handle — most processors expose `.tokenizer`; some newer
    # ones also expose `.text_tokenizer` or similar. Fall back if needed.
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = processor  # some processors are tokenizers directly
    print("Building decision token sets (CLOSED / OPEN) ...")
    t0 = time.time()
    closed_token_ids, open_token_ids = build_decision_token_sets(tokenizer)
    print(f"  CLOSED-like tokens: {len(closed_token_ids)}  "
          f"OPEN-like tokens: {len(open_token_ids)}  "
          f"({time.time()-t0:.1f}s)\n")
    if not closed_token_ids or not open_token_ids:
        print("ERROR: decision token set is empty — tokenizer introspection failed.")
        print("       Confidence extraction would be null for every sample — aborting.")
        sys.exit(2)

    # ------------------------------------------------------------------
    # Load val split
    # ------------------------------------------------------------------
    print(f"Loading val split from {val_path} ...")
    samples = load_val_jsonl(val_path)
    if args.limit is not None:
        samples = samples[:args.limit]
    n_closed = sum(1 for _, lbl, _ in samples if lbl == "CLOSED")
    n_open   = sum(1 for _, lbl, _ in samples if lbl == "OPEN")
    print(f"  {len(samples)} frames — CLOSED: {n_closed}, OPEN: {n_open}")
    print(f"Evaluating {len(samples)} frames ...\n")

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    results   = []
    correct   = 0
    per_class = {"CLOSED": {"correct": 0, "total": 0},
                 "OPEN":   {"correct": 0, "total": 0}}

    for i, (frame, true_label, img_path) in enumerate(samples):
        t_start  = time.time()
        response, p_closed, p_open = predict(
            model, processor, frame, device, closed_token_ids, open_token_ids,
            prompt_text,
        )
        pred       = parse_prediction(response)
        is_correct = (pred == true_label)
        elapsed    = time.time() - t_start

        confidence = (max(p_closed, p_open)
                      if (p_closed is not None and p_open is not None) else None)

        if is_correct:
            correct += 1
        per_class[true_label]["total"] += 1
        if is_correct:
            per_class[true_label]["correct"] += 1

        status   = "✓" if is_correct else "✗"
        conf_str = f"{confidence:.3f}" if confidence is not None else "  -  "
        print(f"  [{i+1:4d}/{len(samples)}] {status}  true={true_label:6s}  "
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
    print(f"  RESULTS — {args.model}")
    print(f"{'='*60}")
    print(f"  Overall accuracy : {overall_acc*100:.1f}%  ({correct}/{len(samples)})")
    print(f"  CLOSED accuracy  : {closed_acc*100:.1f}%  ({per_class['CLOSED']['correct']}/{per_class['CLOSED']['total']})")
    print(f"  OPEN accuracy    : {open_acc*100:.1f}%  ({per_class['OPEN']['correct']}/{per_class['OPEN']['total']})")
    print(f"  Precision        : {precision*100:.1f}%")
    print(f"  Recall           : {recall*100:.1f}%")
    print(f"  F1               : {f1:.3f}")
    print(f"  UNKNOWN responses: {unk}")
    print(f"{'='*60}\n")

    summary = {
        "mode":             "FINE-TUNED" if args.adapter else "BASELINE (no fine-tuning)",
        "model":            args.model,
        "model_class":      args.model_class,
        "adapter":          args.adapter,
        "val_jsonl":        args.val_jsonl,
        "n_samples":        len(samples),
        "overall_accuracy": overall_acc,
        "closed_accuracy":  closed_acc,
        "open_accuracy":    open_acc,
        "precision":        precision,
        "recall":           recall,
        "f1":               f1,
        "unknown_count":    unk,
        "confusion":        {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "per_sample":       results,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
