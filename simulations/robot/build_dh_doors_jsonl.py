#!/usr/bin/env python3
"""Build data/dh_doors_test.jsonl from the labelled real-world door photos.

Each file under data/dh_doors/ is named IMG_XXXX_{OPEN|CLOSED}.jpg — the label
is parsed out of the filename. Output matches the unified JSONL format used by
eval_unified.py (image path relative to the JSONL directory, the unified
prompt, and the State: [...] answer format used during fine-tuning).
"""

import json
import os

_proj    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMG_DIR  = os.path.join(_proj, "data", "dh_doors")
OUT_PATH = os.path.join(_proj, "data", "dh_doors_test.jsonl")

USER_PROMPT = (
    "<|vision_start|><|image_pad|><|vision_end|>"
    "What is shown in this image? Options: door open, door closed, "
    "glass cracked, glass intact."
)


def main() -> None:
    files = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith(".jpg"))
    counts = {"OPEN": 0, "CLOSED": 0}
    with open(OUT_PATH, "w") as f:
        for name in files:
            stem = os.path.splitext(name)[0]
            if stem.endswith("_OPEN"):
                label = "DOOR_OPEN"
                counts["OPEN"] += 1
            elif stem.endswith("_CLOSED"):
                label = "DOOR_CLOSED"
                counts["CLOSED"] += 1
            else:
                print(f"  skipping unlabelled file: {name}")
                continue
            rec = {
                "conversations": [
                    {"role": "user",      "content": USER_PROMPT},
                    {"role": "assistant", "content": f"State: [{label}]."},
                ],
                "image": f"dh_doors/{name}",
            }
            f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(f"wrote {OUT_PATH}  (OPEN={counts['OPEN']}  CLOSED={counts['CLOSED']})")


if __name__ == "__main__":
    main()
