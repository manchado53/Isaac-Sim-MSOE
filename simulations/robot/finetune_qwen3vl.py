#!/usr/bin/env python3
"""
finetune_qwen3vl.py — LoRA + projector fine-tuning of Qwen3-VL-2B-Instruct
                     for door state classification.

Mirrors the Qwen2-VL fine-tune recipe documented in VLM_fine-tuning.md, with
two architectural differences specific to Qwen3-VL:

  1. Vision projector consists of TWO module groups, both unfrozen:
       - model.visual.merger              (final pre-LLM projector)
       - model.visual.deepstack_merger_list  (3 intermediate mergers feeding
                                              features into deeper LLM layers)
     Training only `merger` (mirroring Qwen2-VL) leaves the deepstack mergers
     producing pre-trained-natural-image features into the LLM, which limits
     sim-domain adaptation. Training both groups gives the LoRA the cleanest
     visual signal to lean on.

  2. LLM attention modules are at `model.language_model.layers.N.self_attn.*`.
     LoRA target_modules ["q_proj", "v_proj"] still resolve correctly because
     PEFT matches on the leaf name.

Architecture:
  - Vision encoder (model.visual.blocks)              : FROZEN
  - Final merger (model.visual.merger)                : UNFROZEN, full weights
  - Deepstack mergers (model.visual.deepstack_merger_list)  : UNFROZEN, full weights
  - LLM backbone (model.language_model)               : LoRA on q_proj + v_proj

Input:
  data/train_v4.jsonl, data/val_v4.jsonl  (produced by prepare_finetune_data.py)
  data/finetune_images_v2/*.jpg

Output:
  data/qwen3vl_door_lora_v4/   (adapter weights, loadable with peft)

Usage:
  sbatch submit_finetune_qwen3vl.sh
  # or directly:
  python3 simulations/robot/finetune_qwen3vl.py [--epochs 3] [--lr 1e-5]
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
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VLM_MODEL_ID   = "Qwen/Qwen3-VL-2B-Instruct"
LORA_RANK      = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]

# Both projector groups are saved (and so unfrozen by PEFT). The final merger is
# a single module, but `deepstack_merger_list` is a `nn.ModuleList` container —
# PEFT refuses to wrap containers, so we expand it to its children at runtime
# (see build_modules_to_save below). The fp16→fp32 cast loop uses the substring
# "merger" to catch both families.
FINAL_MERGER_PATH    = "model.visual.merger"
DEEPSTACK_LIST_PATH  = "model.visual.deepstack_merger_list"


def build_modules_to_save(model):
    """Return a list of concrete module paths to save/unfreeze.

    PEFT's `modules_to_save` cannot target a `nn.ModuleList` directly, so we
    enumerate the direct children of `deepstack_merger_list` (e.g. `.0`, `.1`,
    `.2`) and combine them with the single final merger.
    """
    deepstack_children = []
    prefix = DEEPSTACK_LIST_PATH + "."
    for name, _ in model.named_modules():
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            if "." not in suffix:  # direct child only, e.g. "0"
                deepstack_children.append(name)
    return [FINAL_MERGER_PATH] + deepstack_children

CLOSED_TOKEN = "[CLOSED]"
OPEN_TOKEN   = "[OPEN]"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class DoorStateDataset(Dataset):
    """Loads JSONL records and returns tokenized inputs for Qwen3-VL Trainer."""

    def __init__(self, jsonl_path: str, data_dir: str, processor):
        self.records   = []
        self.data_dir  = data_dir
        self.processor = processor

        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec      = self.records[idx]
        convs    = rec["conversations"]
        img_path = os.path.join(self.data_dir, rec["image"])
        image    = Image.open(img_path).convert("RGB")

        text = self.processor.apply_chat_template(
            convs, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            text=text, images=[image], return_tensors="pt", padding=True,
        )
        # Only squeeze the batch dim from text tensors. `pixel_values` is
        # already shape [num_patches, D] (no batch dim). `image_grid_thw` is
        # shape [num_images, 3] and must stay 2D — squeezing it to [3] breaks
        # the model's pos-embed interpolation (expects a list of rows).
        # `mm_token_type_ids` is a Qwen3-VL-specific text-side tensor of shape
        # [1, seq_len]; must be squeezed like input_ids so pad_sequence treats
        # it as a 1D per-sample sequence in the collator.
        TEXT_KEYS = {"input_ids", "attention_mask", "mm_token_type_ids"}
        inputs = {
            k: (v.squeeze(0) if k in TEXT_KEYS else v) for k, v in inputs.items()
        }

        # Mask everything before the assistant turn so loss is only on the
        # response tokens. Qwen3-VL uses the same ChatML special tokens as
        # Qwen2-VL (<|im_start|>, <|im_end|>), so this logic is identical.
        input_ids = inputs["input_ids"]
        labels    = input_ids.clone()

        im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        positions   = (input_ids == im_start_id).nonzero(as_tuple=True)[0]

        mask_until = int(0.8 * len(input_ids))
        if len(positions) > 0:
            last_pos = positions[-1].item()
            for j in range(last_pos + 1, min(last_pos + 8, len(input_ids))):
                tok = self.processor.tokenizer.decode(
                    [input_ids[j].item()], skip_special_tokens=False
                )
                if "\n" in tok:
                    mask_until = j + 1
                    break
            else:
                mask_until = last_pos + 3

        labels[:mask_until] = -100
        inputs["labels"] = labels
        return inputs


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------


def collate_fn(batch):
    # Qwen VL vision tensors are already per-patch (pixel_values) or per-image
    # (image_grid_thw) flat tensors — concatenate across the batch instead of
    # padding. Padding them creates phantom patches that desync from the
    # position embeddings (derived from image_grid_thw), which crashes the
    # visual forward pass with a shape mismatch.
    VISION_CAT_KEYS = {"pixel_values", "image_grid_thw"}

    keys   = batch[0].keys()
    result = {}
    for key in keys:
        tensors = [item[key] for item in batch]
        if key in VISION_CAT_KEYS:
            result[key] = torch.cat(tensors, dim=0)
        elif key == "labels":
            result[key] = torch.nn.utils.rnn.pad_sequence(
                tensors, batch_first=True, padding_value=-100,
            )
        else:
            # input_ids, attention_mask, position_ids → text sequences
            result[key] = torch.nn.utils.rnn.pad_sequence(
                tensors, batch_first=True, padding_value=0,
            )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data/")
    parser.add_argument("--output_dir",  default="data/qwen3vl_door_lora_v4")
    parser.add_argument("--train_jsonl", default="train_v4.jsonl")
    parser.add_argument("--val_jsonl",   default="val_v4.jsonl")
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--lr",          type=float, default=1e-5)
    args = parser.parse_args()

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    data_dir    = os.path.abspath(os.path.join(script_dir, "..", "..", args.data_dir))
    output_dir  = os.path.abspath(os.path.join(script_dir, "..", "..", args.output_dir))

    train_jsonl = os.path.join(data_dir, args.train_jsonl)
    val_jsonl   = os.path.join(data_dir, args.val_jsonl)

    for path in [train_jsonl, val_jsonl]:
        if not os.path.isfile(path):
            print(f"ERROR: {path} not found. Run prepare_finetune_data.py first.")
            raise SystemExit(1)

    # ------------------------------------------------------------------
    # Load model + processor
    # ------------------------------------------------------------------
    print(f"Loading {VLM_MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)

    try:
        import flash_attn  # noqa: F401
        _has_fa2 = True
    except ImportError:
        _has_fa2 = False
    attn_impl = ("flash_attention_2"
                 if (_has_fa2 and torch.cuda.get_device_capability()[0] >= 8)
                 else "sdpa")
    print(f"  Attention implementation: {attn_impl}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation=attn_impl,
    )

    # ------------------------------------------------------------------
    # LoRA on LLM attention; full unfreeze on both projector groups.
    # Order matters: get_peft_model() re-freezes everything except LoRA,
    # then modules_to_save un-freezes the merger groups again.
    # ------------------------------------------------------------------
    print("Configuring parameter groups ...")

    modules_to_save = build_modules_to_save(model)
    print(f"  modules_to_save ({len(modules_to_save)}): {modules_to_save}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(model, lora_config)

    # Cast both merger groups fp16 → fp32 so AMP grad scaler can update them.
    # Substring "merger" matches both `visual.merger` and `visual.deepstack_merger_list.*`.
    n_proj = 0
    for name, param in model.named_parameters():
        if "merger" in name and param.requires_grad:
            param.data = param.data.float()
            n_proj += param.numel()
    if n_proj:
        print(f"  Vision projectors (merger + deepstack): trainable + cast to fp32  "
              f"({n_proj:,} params)")
    else:
        print("  WARN: No merger params found as trainable — check MODULES_TO_SAVE.")

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
        gradient_accumulation_steps=4,    # effective batch = 8
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        load_best_model_at_end=False,
        logging_steps=20,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        prediction_loss_only=True,
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

    print(f"\nSaving LoRA adapter to {output_dir} ...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
