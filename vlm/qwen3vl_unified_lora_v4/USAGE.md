# Qwen3-VL-2B Unified Door+Glass LoRA — Usage

LoRA adapter fine-tuned on a 4-class image classification task:

- `door open`
- `door closed`
- `glass cracked`
- `glass intact`

Base model: **`Qwen/Qwen3-VL-2B-Instruct`** (downloaded automatically from Hugging Face on first load).

## 1. Install

```bash
pip install "torch>=2.1" "transformers>=4.45" "peft>=0.12" pillow accelerate
```

GPU with ≥ 8 GB VRAM recommended (fp16). CPU works but is slow.

## 2. Folder layout

Unzip the tarball. You should see:

```
qwen3vl_unified_lora_v4/
├── adapter_config.json
├── adapter_model.safetensors      # the trained weights (~397 MB)
├── chat_template.jinja
├── processor_config.json
├── tokenizer.json
├── tokenizer_config.json
└── USAGE.md                       # this file
```

## 3. Load and run inference

```python
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ADAPTER_DIR = "qwen3vl_unified_lora_v4"   # path to the unzipped folder
BASE_MODEL  = "Qwen/Qwen3-VL-2B-Instruct"

processor = AutoProcessor.from_pretrained(ADAPTER_DIR)
base = Qwen3VLForConditionalGeneration.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, device_map="auto",
)
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()

PROMPT = (
    "<|vision_start|><|image_pad|><|vision_end|>"
    "What is shown in this image? Options: door open, door closed, "
    "glass cracked, glass intact."
)

image = Image.open("your_image.jpg").convert("RGB")
convs = [{"role": "user", "content": PROMPT}]
text = processor.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
answer = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
print(answer)
```

## 4. Expected output

The model replies with one of exactly four strings:

- `State: [DOOR_OPEN].`
- `State: [DOOR_CLOSED].`
- `State: [GLASS_CRACKED].`
- `State: [GLASS_INTACT].`

## 5. Gotchas

- **Do not call `model.merge_and_unload()`** — it crashes because the adapter
  includes trained visual-projector weights (`modules_to_save`) that cannot be
  merged into LoRA deltas. Use the `PeftModel` wrapper directly.
- Use the exact prompt string above. Training only saw that one phrasing.
- First run downloads the ~4 GB base model to `~/.cache/huggingface`.
