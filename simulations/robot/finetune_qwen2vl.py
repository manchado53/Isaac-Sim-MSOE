#!/usr/bin/env python3
"""
finetune_qwen2vl.py — LoRA fine-tuning of Qwen2-VL-2B-Instruct for door state classification.

Architecture (from VLM_fine-tuning.md):
  - Vision encoder   : FROZEN  (preserves general object/scene knowledge)
  - Vision projector : UNFROZEN full training (adapts to sim-specific visual features)
  - LLM backbone     : LoRA on q_proj + v_proj (learns OPEN/CLOSED response format)

Input:
  data/train.jsonl, data/val.jsonl  (produced by prepare_finetune_data.py)
  data/finetune_images/*.jpg

Output:
  data/qwen2vl_door_lora/           (adapter weights, loadable with peft)

Usage:
  sbatch scripts/slurm/submit_finetune_job.sh
  # or directly:
  python3 simulations/robot/finetune_qwen2vl.py [--data_dir data/] [--output_dir data/qwen2vl_door_lora]

Notes:
  - Requires: transformers>=4.45, peft>=0.12, torch>=2.1, Pillow, scikit-learn
  - Run on dgxh100 partition (H100) for Flash Attention 2 support.
    T4 (teaching partition) does NOT support FA2 — falls back to SDPA automatically.
  - Effective batch size = per_device_train_batch_size × gradient_accumulation_steps = 8
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VLM_MODEL_ID  = "Qwen/Qwen2-VL-2B-Instruct"
LORA_RANK     = 16      # start conservative; increase to 32/64 if val accuracy plateaus
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]

CLOSED_TOKEN = "[CLOSED]"
OPEN_TOKEN   = "[OPEN]"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class DoorStateDataset(Dataset):
    """Loads JSONL records and returns tokenized inputs for Qwen2-VL Trainer."""

    def __init__(self, jsonl_path: str, data_dir: str, processor, max_new_tokens: int = 16):
        self.records    = []
        self.data_dir   = data_dir
        self.processor  = processor
        self.max_new_tokens = max_new_tokens

        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        convs = rec["conversations"]
        img_path = os.path.join(self.data_dir, rec["image"])

        image = Image.open(img_path).convert("RGB")

        # Build the chat text via apply_chat_template
        text = self.processor.apply_chat_template(
            convs, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        # Flatten batch dim added by processor
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Labels: mask system + user turns, only train on assistant response tokens.
        # Strategy: find the LAST <|im_start|> token (always the assistant turn regardless
        # of whether Qwen2's template prepends a system turn), then scan forward past the
        # "assistant\n" header to where the actual response content begins.
        input_ids = inputs["input_ids"]
        labels    = input_ids.clone()

        im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        positions   = (input_ids == im_start_id).nonzero(as_tuple=True)[0]

        mask_until = int(0.8 * len(input_ids))  # safe fallback
        if len(positions) > 0:
            last_pos = positions[-1].item()
            # Scan forward past "assistant\n" (typically 2–3 tokens after <|im_start|>)
            for j in range(last_pos + 1, min(last_pos + 8, len(input_ids))):
                tok = self.processor.tokenizer.decode(
                    [input_ids[j].item()], skip_special_tokens=False
                )
                if "\n" in tok:
                    mask_until = j + 1  # mask through the newline inclusive
                    break
            else:
                mask_until = last_pos + 3  # fallback: im_start + "assistant" + "\n"

        labels[:mask_until] = -100
        inputs["labels"] = labels
        return inputs


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------


def collate_fn(batch):
    """Pad a batch of variable-length tokenized inputs."""
    keys = batch[0].keys()
    result = {}
    for key in keys:
        tensors = [item[key] for item in batch]
        if key == "labels":
            result[key] = torch.nn.utils.rnn.pad_sequence(
                tensors, batch_first=True, padding_value=-100
            )
        elif key == "attention_mask":
            result[key] = torch.nn.utils.rnn.pad_sequence(
                tensors, batch_first=True, padding_value=0
            )
        else:
            result[key] = torch.nn.utils.rnn.pad_sequence(
                tensors, batch_first=True, padding_value=0
            )
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def build_compute_metrics(processor):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # logits: (N, seq_len, vocab) — causal LM: logit at position t predicts token at t+1.
        # labels: (N, seq_len) with -100 for masked (non-response) positions.
        # To predict label token at position j, use logits[:, j-1, :].
        y_pred = []
        y_true = []

        for i in range(labels.shape[0]):
            valid = np.where(labels[i] != -100)[0]
            if len(valid) == 0:
                continue

            # Ground truth: decode the non-masked response tokens
            true_text = processor.tokenizer.decode(
                labels[i][valid].tolist(), skip_special_tokens=True
            )

            # Prediction: logit at valid[j]-1 predicts the token at valid[j]
            pred_ids  = logits[i, valid - 1, :].argmax(axis=-1)
            pred_text = processor.tokenizer.decode(pred_ids.tolist(), skip_special_tokens=True)

            y_true.append(1 if OPEN_TOKEN in true_text else 0)
            y_pred.append(1 if OPEN_TOKEN in pred_text else 0)

        if not y_true:
            return {"accuracy": 0.0, "f1": 0.0}

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1":       f1_score(y_true, y_pred, zero_division=0),
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data/")
    parser.add_argument("--output_dir", default="data/qwen2vl_door_lora")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--lr",         type=float, default=1e-5)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir   = os.path.abspath(os.path.join(script_dir, "..", "..", args.data_dir))
    output_dir = os.path.abspath(os.path.join(script_dir, "..", "..", args.output_dir))

    train_jsonl = os.path.join(data_dir, "train.jsonl")
    val_jsonl   = os.path.join(data_dir, "val.jsonl")

    for path in [train_jsonl, val_jsonl]:
        if not os.path.isfile(path):
            print(f"ERROR: {path} not found. Run prepare_finetune_data.py first.")
            raise SystemExit(1)

    # ------------------------------------------------------------------
    # Load model + processor
    # ------------------------------------------------------------------
    print(f"Loading {VLM_MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)

    # Use Flash Attention 2 on Ampere+ (H100/A100), SDPA fallback on T4
    attn_impl = "flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "sdpa"
    print(f"  Attention implementation: {attn_impl}")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation=attn_impl,
    )

    # ------------------------------------------------------------------
    # Apply LoRA to LLM backbone first, THEN unfreeze projector.
    # Order matters: get_peft_model() calls mark_only_lora_as_trainable()
    # internally which re-freezes ALL non-LoRA params — so the projector
    # must be unfrozen AFTER get_peft_model(), not before.
    # ------------------------------------------------------------------
    print("Configuring parameter groups ...")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        # modules_to_save ensures PEFT includes non-LoRA trainable weights
        # (the vision projector) in adapter_model.safetensors at save time.
        # Without this, visual.merger updates are silently discarded on save.
        modules_to_save=["model.visual.merger"],
    )
    model = get_peft_model(model, lora_config)

    # Cast merger params fp16 → fp32 so AMP grad scaler can update them.
    # modules_to_save already handles unfreezing; do NOT set requires_grad here
    # or it also unfreezes the frozen original_module copy inside the wrapper,
    # doubling the trainable param count to 68M instead of the correct 34M.
    n_proj = 0
    for name, param in model.named_parameters():
        if "merger" in name and param.requires_grad:
            param.data = param.data.float()
            n_proj += param.numel()
    if n_proj:
        print(f"  Vision projector (merger): trainable + cast to fp32  ({n_proj:,} params)")
    else:
        print("  WARN: No merger params found as trainable — check modules_to_save.")

    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    print(f"\nBuilding datasets from {data_dir} ...")
    train_dataset = DoorStateDataset(train_jsonl, data_dir, processor)
    val_dataset   = DoorStateDataset(val_jsonl,   data_dir, processor)
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,   # effective batch = 8
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Prevent overfitting: eval_loss hit 0.001 by epoch 3 but the model
        # collapsed to predicting OPEN-only. Early stopping helps, but the real
        # fix is modules_to_save above. Keep epochs low (1-2) until verified.
        logging_steps=20,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        prediction_loss_only=True,   # avoids accumulating full logit tensors during eval
                                     # which OOMs on V100 (32k vocab × seq_len × 360 samples)
                                     # accuracy is measured post-training via eval_baseline.py
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    print(f"\nStarting training for {args.epochs} epoch(s) ...")
    trainer.train()

    # Save adapter only (not full model weights)
    print(f"\nSaving LoRA adapter to {output_dir} ...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
