---
license: other
license_name: health-ai-developer-foundations-terms-of-use
license_link: https://developers.google.com/health-ai-developer-foundations/terms
base_model: google/medgemma-1.5-4b-it
base_model_relation: adapter
datasets:
  - Ssdaizi/SonoInstruct
library_name: peft
pipeline_tag: image-text-to-text
tags:
  - medical
  - ultrasound
  - vision-language-model
  - lora
  - peft
  - visual-grounding
---

# SonoMed-VLM — MedGemma 1.5 4B LoRA

SonoMed-VLM is a rank-16 LoRA adapter for ultrasound understanding, question
answering, report-style generation, multiple-choice tasks, and visual grounding.
It adapts `google/medgemma-1.5-4b-it` on 190,625 training examples from
SonoInstruct while keeping the vision tower and multimodal projector frozen.

The released [Hugging Face model repository](https://huggingface.co/jxia58/SonoMed-VLM-MedGemma-1.5-4B-LoRA) contains adapter weights only. This GitHub repository contains code,
configs and aggregate research results, not model weights or raw dataset content.

Source code and reproducibility materials are available at
[jxia622/SonoMed-VLM](https://github.com/jxia622/SonoMed-VLM).

> Research use only. This model is not a medical device and is not approved for
> diagnosis, treatment, triage, or clinical decision-making. Outputs require
> independent expert verification and task-, population-, site-, and
> device-specific validation.

## Model details

| Field | Value |
|---|---|
| Base model | `google/medgemma-1.5-4b-it` |
| Pinned base revision | `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b` |
| Adapter method | BF16 LoRA SFT |
| LoRA | rank 16, alpha 32, dropout 0.05 |
| Targets | decoder q/k/v/o and gate/up/down projections |
| Trainable parameters | 29,802,496 across 238 modules |
| Dataset | `Ssdaizi/SonoInstruct` |
| Training examples | 190,625 |
| Validation examples | 10,098 |
| Seed | 42 |
| Effective batch | 8 on 4 GPUs |

## Results

**Updated 27 September 2026:** These values use the common scorer and corrected
task routing. The MCQ prompts omitted choices during training and inference;
semantic matching is diagnostic only, not standard MCQ accuracy. See the
[audit](docs/MCQ_PROMPT_AUDIT_20260927.md) and [Qwen comparison](docs/RESEARCH.md).

| Metric | Result |
|---|---:|
| QA + open-response token F1 | 0.3496 |
| QA + open-response ROUGE-L | 0.2607 |
| Visual-grounding mean IoU | 0.5831 |
| Localization@0.5 | 71.15% |
| Valid-box rate | 100.00% |
| Answer-content matching (choices omitted; diagnostic) | 93.52% |

The split contains 10,098 examples: 565 grounding, 4,628 QA/open, and 4,905
MCQ-labeled diagnostic examples. One epoch, one seed; no uncertainty intervals.

### Scaling study

| Data | Examples | QA/open token F1 | Mean IoU | Loc@0.5 |
|---:|---:|---:|---:|---:|
| 5% | 9,379 | 0.2914 | 0.3020 | 24.96% |
| 10% | 19,035 | 0.3144 | 0.3231 | 32.04% |
| 25% | 47,860 | 0.3237 | 0.3615 | 37.17% |
| 50% | 95,544 | 0.3383 | 0.4419 | 46.90% |
| 100% | 190,625 | 0.3496 | 0.5831 | 71.15% |

### MCQ protocol limitation

Earlier descriptions attributed label/text contradictions to model label binding.
That interpretation is unsupported because the model did not receive the choices.
Presenting choices at inference alone would not repair the training protocol.
The current published adapter retains this limitation. Corrected retraining remains
a follow-up experiment.

### Grounding correction

SonoInstruct boxes use `[0,1000]` coordinates. An initial evaluator accepted
only `[0,1]`; release metrics were recomputed after supporting `[0,1000]` and
normalizing both reference and prediction equivalently for IoU. Original
artifacts were preserved separately.

## Load the adapter

Users must first accept the gated MedGemma terms and authenticate with Hugging
Face. Install `torch`, `transformers`, `accelerate`, `peft`, and `pillow`.

```python
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

base_id = "google/medgemma-1.5-4b-it"
revision = "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b"
adapter_id = "jxia58/SonoMed-VLM-MedGemma-1.5-4B-LoRA"

processor = AutoProcessor.from_pretrained(base_id, revision=revision)
base = AutoModelForImageTextToText.from_pretrained(
    base_id,
    revision=revision,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, adapter_id).eval()

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": Image.open("ultrasound.png").convert("RGB")},
            {"type": "text", "text": "Describe the ultrasound image."},
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
prompt_length = inputs["input_ids"].shape[-1]
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(processor.decode(output[0][prompt_length:], skip_special_tokens=True))
```

## Intended use

- research on ultrasound vision-language modeling;
- reproducible analysis of parameter-efficient medical-model adaptation;
- offline experiments in ultrasound QA, report-style generation, MCQ, and
  bounding-box grounding;
- education and prototyping with qualified expert oversight.

## Out-of-scope use

- diagnosis, treatment, triage, or patient management;
- autonomous or unreviewed healthcare decisions;
- claims of clinical performance or regulatory approval;
- use that violates the HAI-DEF Prohibited Use Policy, applicable law, patient
  privacy, dataset terms, or third-party rights.

## Limitations

- Evaluation is limited to one SonoInstruct split and does not establish
  external clinical validity across sites, devices, protocols, or populations.
- One deterministic run was completed per scale; seed variance was not measured.
- MCQ choices were omitted during training and inference; standard MCQ conclusions are invalid.
- Token F1 and ROUGE-L do not measure factual or clinical safety.
- The adapter can generate incorrect, incomplete, biased, or unsafe responses.

## Training hardware

The released scale runs used one node with 4× NVIDIA RTX PRO 6000 Blackwell
GPUs. In a controlled 50-step benchmark with the same effective batch, this
platform processed 2.959 examples/second versus 1.464 on 4× A100 (2.02×). No
custom Blackwell kernel is claimed.

## License and attribution

This adapter is a MedGemma **Model Derivative** and is distributed under the
[Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms),
including its [Prohibited Use Policy](https://developers.google.com/health-ai-developer-foundations/prohibited-use-policy).
The distribution includes [`HAI_DEF_TERMS.md`](HAI_DEF_TERMS.md) and the
required [`NOTICE`](NOTICE) file.
Modified adapter files are identified as modifications produced by LoRA
supervised fine-tuning on SonoInstruct.

SonoInstruct is released under Apache-2.0. The adapter repository does not
redistribute the dataset. Users must obtain the gated MedGemma base model and
accept its terms separately.

Project code, configs, evaluator, and full methodology:
[github.com/jxia622/SonoMed-VLM](https://github.com/jxia622/SonoMed-VLM)
