#!/usr/bin/env python3
"""Run one ultrasound image through the released SonoMed-VLM LoRA adapter."""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

BASE_MODEL = "google/medgemma-1.5-4b-it"
BASE_REVISION = "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b"
ADAPTER = "jxia622/SonoMed-VLM-MedGemma-1.5-4B-LoRA"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to an ultrasound image")
    parser.add_argument("--prompt", default="Describe the ultrasound image.")
    parser.add_argument("--adapter", default=ADAPTER)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    base = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, args.adapter).eval()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(args.image).convert("RGB")},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    input_length = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    print(processor.decode(generated[0][input_length:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
