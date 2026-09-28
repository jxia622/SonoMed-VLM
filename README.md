# SonoMed-VLM

### Does medical specialization help a model adapt to ultrasound?

A reproducible comparison of **MedGemma 1.5 4B** and **Qwen3-VL 4B** across training-data scales, with decoder-only LoRA and a shared held-out SonoInstruct split.

[Research report](docs/RESEARCH.md) · [Results CSV](results/research_results.csv) · [Results + provenance](results/research_results.json) · [Evaluation protocol](docs/EVALUATION.md) · [Released MedGemma adapter](https://huggingface.co/jxia58/SonoMed-VLM-MedGemma-1.5-4B-LoRA)

![Original and adapted model grounding performance](docs/assets/research/grounding_comparison.png)

**Current finding:** Under our one-epoch recipe, adapted Qwen achieved higher ultrasound grounding scores than adapted MedGemma at every shared tested data fraction. At full data, mean IoU was **0.7449 vs. 0.5831**. At just **5%** of training data, Qwen reached **0.6143**, above MedGemma's full-data observation.

> **Research status:** Preliminary, single-seed results on a validation split, not official SonoBench or external clinical results. An audit found that MCQ choices were omitted from both training and inference. MCQ-derived scores below are diagnostic answer matching, **not valid standard MCQ accuracy**. [Read the audit](docs/MCQ_PROMPT_AUDIT_20260927.md).

**Correction in progress:** The [v2 open-ended QA protocol](docs/OPEN_QA_V2.md) removes option labels from training targets and evaluates answer text without candidate choices. New results are pending; the figures below describe the historical v1 recipe.

## Why compare these models?

Medical knowledge may help ultrasound interpretation, but useful transfer could depend on the task and imaging modality. We ask whether starting from a medically specialized model produces better adaptation and data efficiency than starting from a similarly sized general VLM.

Our initial hypothesis was that specialization in other medical modalities might transfer less effectively to ultrasound than general visual grounding. **We have not demonstrated that MedGemma overfit CT or MRI.** Its documented training spans medical text, QA, and several image modalities; Qwen also cannot be assumed to have had zero medical exposure. [MedGemma model card](https://huggingface.co/google/medgemma-1.5-4b-it).

This experiment compares complete model families, whose architectures, processors and pretraining differ. It cannot isolate the effect of medical QA training. The [research report](docs/RESEARCH.md) explains alternative explanations, related work, and the controls needed to test causality.

## Grounding results

All rows use the same **565 grounding examples** within the 10,098-example held-out split. Mean IoU measures box overlap; Localization@0.5 is the fraction of examples with IoU at least 0.5.

| Model | Adaptation | Mean IoU | Localization@0.5 |
|---|---|---:|---:|
| Qwen3-VL 4B | Original | 0.1444 | 16.81% |
| Qwen3-VL 4B | 100% LoRA | 0.7449 | 86.90% |
| MedGemma 1.5 4B | Original | 0.0894 | 5.31% |
| MedGemma 1.5 4B | 100% LoRA | 0.5831 | 71.15% |

- Qwen improves **5.16×** over its original mean IoU; MedGemma improves **6.52×** over its own lower baseline.
- At full data, Qwen's mean IoU is **27.75% higher** than MedGemma's; Localization@0.5 is **15.75 percentage points higher**.
- These are observed differences without seed repeats, confidence intervals, or significance claims.

## How much adaptation data is needed?

![Grounding performance across training-data fractions](docs/assets/research/grounding_scaling.png)

| Data | Examples | Qwen IoU | MedGemma IoU | Qwen Loc@0.5 | MedGemma Loc@0.5 |
|---:|---:|---:|---:|---:|---:|
| 1% | 1,906 | 0.3673 | — | 33.81% | — |
| 5% | 9,379 | 0.6143 | 0.3020 | 74.16% | 24.96% |
| 10% | 19,035 | 0.6585 | 0.3231 | 81.24% | 32.04% |
| 25% | 47,860 | 0.7006 | 0.3615 | 84.07% | 37.17% |
| 50% | 95,544 | 0.7258 | 0.4419 | 85.66% | 46.90% |
| 100% | 190,625 | 0.7449 | 0.5831 | 86.90% | 71.15% |

At 5%, Qwen uses **9,379 examples**, compared with MedGemma's **190,625** at 100%—approximately **20.3× fewer examples** while exceeding both measured grounding endpoints. This is data efficiency under the tested recipe, not a claim of equivalent compute or a precisely estimated sample-complexity ratio. No full-split 1% MedGemma evaluation is available.

Each fraction starts from the original checkpoint, independently. Subsets are nested, image-group-safe, and shared between model families. One epoch means larger fractions also receive more optimizer updates.

## Language results and the MCQ limitation

![QA overlap and diagnostic answer matching](docs/assets/research/language_diagnostics.png)

| Model | Adaptation | QA/open token F1 | Answer-content matching† |
|---|---|---:|---:|
| Qwen3-VL 4B | Original | 0.1924 | 12.80% |
| Qwen3-VL 4B | 100% LoRA | 0.3535 | 93.84% |
| MedGemma 1.5 4B | Original | 0.1880 | 17.59% |
| MedGemma 1.5 4B | 100% LoRA | 0.3496 | 93.52% |

† Diagnostic only: choices omitted; not standard MCQ accuracy.

The full-data language scores are close: **0.3535 vs. 0.3496 token F1**, and **93.84% vs. 93.52% diagnostic answer matching**. We do not claim statistical equivalence or clinical correctness. The strong grounding gap does not imply superiority on every task.

The shared input builder kept MCQ options in metadata but did not show them to the model. Consequently, earlier descriptions of low letter accuracy as a model “label-binding failure” were unsupported. Corrected MCQ prompts, target checks, and retraining are required before drawing conventional MCQ conclusions. Grounding scores are not directly invalidated by the omission, although the flawed MCQ training mixture remains part of the recipe.

## Experiment at a glance

| Component | Setting |
|---|---|
| Medical model | [`google/medgemma-1.5-4b-it`](https://huggingface.co/google/medgemma-1.5-4b-it) |
| General model | [`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |
| Dataset | [SonoInstruct](https://huggingface.co/datasets/Ssdaizi/SonoInstruct); obtained separately |
| Full training / validation | 190,625 / 10,098 examples |
| Adaptation | One epoch; BF16; decoder LoRA rank 16, alpha 32, dropout 0.05 |
| Frozen components | Vision encoder and multimodal projector |
| Optimization | Seed 42; LR 1e-4; cosine; effective batch 8 on four GPUs |
| Inference | Greedy; maximum 256 new tokens; native processors and chat templates |
| Split protection | Connected groups of identical image-byte hashes; no patient-level guarantee |
| Evaluation counts | 565 grounding; 4,628 QA/open; 4,905 MCQ-labeled diagnostic records |

The 13 reported model/scale combinations were rescored from saved raw generations with one evaluator. Validation IDs, prompts, references, option metadata, and generation settings were checked for consistency. The [aggregate JSON](results/research_results.json) includes metric denominators, source checksums and model revisions. Historical task-routing and pooling differences are corrected in this release.

## Reproduce the study

```bash
git clone https://github.com/jxia622/SonoMed-VLM.git
cd SonoMed-VLM
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,viz,qwen]'
```

Use a compatible CUDA PyTorch/torchvision pair for GPU work. The CRC runs used torch 2.8.0+cu128, torchvision 0.23.0+cu128, transformers 5.14.1 and peft 0.20.0. Accept MedGemma's gated terms and authenticate with `hf auth login`. Obtain SonoInstruct separately; no dataset images or text are distributed here.

```bash
export SONOMED_PROJECT_ROOT="$PWD"
export SONOINSTRUCT_ROOT="/path/to/SonoInstruct"
export SONOMED_OUTPUT_ROOT="/path/to/outputs"
export HF_HOME="/path/to/huggingface-cache"

python scripts/inspect_dataset.py --data-root "$SONOINSTRUCT_ROOT"
python scripts/build_manifests.py --data-root "$SONOINSTRUCT_ROOT"
python scripts/validate_data.py --data-root "$SONOINSTRUCT_ROOT"
```

The following commands reproduce the **historical choices-omitted recipe**. They are not the proposed repaired MCQ experiment. Use the pinned manifests for exact comparisons; see [the CRC guide](docs/QWEN3VL_CRC.md).

```bash
# Full Qwen training; use a fresh output directory.
torchrun --standalone --nproc_per_node=4 \
  scripts/train.py --config configs/qwen3vl_train_100pct.yaml --strict-numerics

# Full MedGemma training.
torchrun --standalone --nproc_per_node=4 \
  scripts/train.py --config configs/rtx6k_train_100pct.yaml

# Example Qwen adapter evaluation.
torchrun --standalone --nproc_per_node=4 \
  scripts/baseline_eval.py --config configs/qwen3vl_eval_finetuned.yaml \
  --adapter "$SONOMED_OUTPUT_ROOT/qwen3vl_100pct/final_adapter"

# Rebuild figures using only the public aggregate results; no GPU needed.
python scripts/plot_research_results.py
```

Qwen scale configs cover 1%, 5%, 10%, 25%, and 50%. Slurm examples and `scripts/submit_qwen3vl_scaling.sh` provide the CRC workflow. [Rebuild aggregate results from archived predictions](docs/RESEARCH.md#reproducibility-and-provenance).

[PNG, SVG and PDF figures](docs/assets/research/) are generated from the same machine-readable source. Raw generations and adapter checkpoints remain in the authorized experiment storage. Only the earlier MedGemma adapter is currently linked as a public model release; Qwen adapter weights are not published by this repository update.

## What would strengthen the research?

1. Repair MCQ inputs and retrain both families with a versioned protocol.
2. Add Gemma 3 4B as a closer parent-family control for MedGemma.
3. Repeat seeds, quantify paired uncertainty, and reserve an untouched external test set.
4. Control visual token budgets, adapter capacity and tuning budgets; test vision unfreezing.
5. Evaluate clinical answer quality beyond token overlap.

SonoBench has **not** been evaluated. We could not locate the official test package in the authors' public releases as of 27 September 2026. The [research report](docs/RESEARCH.md) distinguishes completed experiments from proposed follow-ups and situates this study relative to existing medical-adaptation and SonoInstruct work.

## Repository map

| Path | Contents |
|---|---|
| `docs/RESEARCH.md` | Question, motivation, hypotheses, methods, findings, caveats and related work |
| `results/research_results.*` | Current aggregate metrics and provenance |
| `docs/assets/research/` | Reproducible research figures in PNG/SVG/PDF |
| `configs/` | Pinned model and data-fraction configurations |
| `scripts/` | Training, inference, validation, comparison and plotting |
| `src/sonomed_vlm/` | Dataset, model, training and evaluation implementation |
| `tests/` | CPU checks and gated/GPU integration tests |

## License and use

Research software only; these experiments do not establish clinical validity. Project code is Apache-2.0 licensed ([LICENSE](LICENSE)). MedGemma and its derivatives remain subject to the [HAI-DEF terms](https://developers.google.com/health-ai-developer-foundations/terms). Consult the upstream Qwen and SonoInstruct releases for their terms. See [third-party notices](THIRD_PARTY_NOTICES.md) and the [MedGemma model card](MODEL_CARD.md).

```bibtex
@software{xia2026sonomedvlm,
  author = {Jack Xia},
  title = {SonoMed-VLM: Medical and General Vision-Language Models for Ultrasound Adaptation},
  year = {2026},
  url = {https://github.com/jxia622/SonoMed-VLM}
}
```
