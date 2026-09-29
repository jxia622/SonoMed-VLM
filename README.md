# SonoMed-VLM

### Does medical specialization help a model adapt to ultrasound?

A reproducible comparison of **Qwen3-VL 4B** and **MedGemma 1.5 4B**, using the same
nested ultrasound subsets and decoder LoRA training recipe.

[Research report](docs/RESEARCH.md) · [All 16 results (CSV)](results/research_results.csv) ·
[Aggregate metrics and hashes (JSON)](results/research_results.json) ·
[Corrected protocol](docs/OPEN_QA_V2.md) · [Controlled follow-up](docs/MEDICAL_INTERMEDIATE_TRANSFER.md)

**Corrected release · 29 September 2026.** All 16 open-QA-v2 evaluations are complete:
six freshly trained data fractions per model, two original models, and two historical
full-data adapter bridge controls. The primary figures below use the corrected retraining.
No pending experiment results are included.

![Original and adapted grounding](docs/assets/research/grounding_comparison.png)

**Finding:** Qwen has higher observed grounding scores at all six data fractions.
At full data, mean IoU is **0.7496 vs. 0.5716**. Converted open-QA exact match slightly
favors MedGemma (**91.29% vs. 91.13%**); language and grounding tell different stories.

These are **one-seed, internal validation results**, not official SonoBench scores,
clinical accuracy estimates, or evidence that medical pretraining is useless.

## Why this comparison?

Medical knowledge might help ultrasound adaptation, but transfer can depend on the
modality and task. We initially wondered whether a general model's visual grounding
would transfer better than medical specialization in other imaging modalities.
**We have not demonstrated that MedGemma overfit CT or MRI**, or that those were its
primary training data. Its documented training spans medical text, QA, and images;
Qwen cannot be assumed to lack medical knowledge.
[MedGemma model card](https://huggingface.co/google/medgemma-1.5-4b-it).

This comparison changes architectures, processors and pretraining together. It does
not isolate the causal effect of medical knowledge. Our [next experiment](docs/MEDICAL_INTERMEDIATE_TRANSFER.md)
keeps Qwen fixed and compares direct ultrasound tuning with medical and general QA
intermediate tuning. It is queued as a **one-seed pilot**, with no results yet.

## Original versus adapted models

| Model | Adaptation | Open-QA exact match | Open-QA token F1 | Ordinary QA F1 | Mean IoU | Loc@0.5 |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-VL 4B | Original | 4.56% | 0.0948 | 0.1980 | 0.1444 | 16.81% |
| Qwen3-VL 4B | Corrected 100% | 91.13% | 0.9261 | 0.3511 | 0.7496 | 87.08% |
| MedGemma 1.5 4B | Original | 5.00% | 0.0808 | 0.1928 | 0.0815 | 4.78% |
| MedGemma 1.5 4B | Corrected 100% | 91.29% | 0.9273 | 0.3430 | 0.5716 | 68.67% |

At full data, Qwen reaches **0.7496 mean IoU versus 0.5716** for MedGemma:
an absolute difference of **0.1780**, or **31.14%** relative.
Localization@0.5 is **87.08% versus 68.67%**
(**18.41 percentage points**).
Qwen improves 5.19× over its original mean IoU;
MedGemma improves 7.01× over its lower baseline.
These are observed differences, not significance claims.

## Data efficiency

![Grounding across all six corrected data fractions](docs/assets/research/grounding_scaling.png)

| Data | Examples | Qwen IoU | MedGemma IoU | Qwen Loc@0.5 | MedGemma Loc@0.5 |
|---:|---:|---:|---:|---:|---:|
| 1% | 1,882 | 0.3857 | 0.1871 | 36.11% | 13.27% |
| 5% | 9,251 | 0.5989 | 0.3257 | 68.67% | 27.43% |
| 10% | 18,761 | 0.6745 | 0.3423 | 81.77% | 33.63% |
| 25% | 47,173 | 0.7007 | 0.3878 | 83.36% | 40.71% |
| 50% | 94,118 | 0.7374 | 0.4150 | 87.08% | 45.84% |
| 100% | 187,761 | 0.7496 | 0.5716 | 87.08% | 68.67% |

Qwen at **5% (9,251 examples)** has higher mean IoU than full-data MedGemma
(**0.5989 vs. 0.5716**) and **ties** its Localization@0.5 (**68.67%**).
It uses **20.3× fewer training examples**. This is not a 20× compute-saving claim;
the exact crossing point between tested fractions is unknown.

Fractions branch independently from each original model. Image-connected groups
remain together; filtering preserves the original nested subsets. One epoch means
larger subsets also receive more optimizer updates.

## Corrected language evaluation

![Corrected open-ended language metrics](docs/assets/research/language_diagnostics.png)

The earlier MCQ recipe omitted options from model inputs while retaining option-letter
targets and option-aware scoring. We replaced it with **open-ended QA: no options,
answer text only**, retrained both models, and regenerated all evaluations.
[Correction and data audit](docs/OPEN_QA_V2.md) · [Historical defect](docs/MCQ_PROMPT_AUDIT_20260927.md).

Every run uses the same **9,964 validation examples**: **4,938 converted open-QA**,
**4,461 ordinary QA/open**, and **565 grounding**. A letter-only answer receives zero
converted-QA credit; the scorer never looks up a predicted letter in hidden options.
Open-QA exact match is not MCQ accuracy. Lexical scores are not clinical correctness.

At full data, converted open-QA token F1 is **0.9261 Qwen / 0.9273 MedGemma**;
ordinary QA token F1 is **0.3511 / 0.3430**. Small single-seed differences do not
establish superiority or equivalence. [All 16 rows and bridge analysis](docs/RESEARCH.md#all-16-evaluations).

## Reproduce and inspect

```bash
python -m pip install -e '.[viz]'
python scripts/plot_research_results.py
```

The plotting script reads the public aggregate JSON and rebuilds four figures in
PNG, SVG and PDF. [Training and aggregation instructions](docs/OPEN_QA_V2.md),
[evaluator definitions](docs/EVALUATION.md), and [CRC setup](docs/QWEN3VL_CRC.md)
cover reproduction. Both models use one epoch, seed 42, rank-16 decoder LoRA,
frozen vision/projector, LR 1e-4, and effective batch eight; their native image
processors and adapter parameter counts differ.

| Material | Location |
|---|---|
| Research question, methods, all 16 results, caveats | [Research report](docs/RESEARCH.md) |
| Current results and prediction hashes | [JSON](results/research_results.json), [CSV](results/research_results.csv) |
| Conversion counts and manifest hashes | [Audit](results/openqa_v2_audit.json) |
| Corrected training/evaluation run registry | [Run specifications](configs/openqa_v2_runs.json) |
| Prior v1 numbers, explicitly historical | [Archive](results/archive/README.md) |
| Medical intermediate-training pilot | [Design and compute plan](docs/MEDICAL_INTERMEDIATE_TRANSFER.md) |
| Published historical MedGemma adapter | [Model card](MODEL_CARD.md), [Hugging Face weights](https://huggingface.co/jxia58/SonoMed-VLM-MedGemma-1.5-4B-LoRA) |

The Hugging Face adapter still contains **historical v1 weights**, not the corrected
v2 retraining. GitHub publishes aggregate research results and code, not raw data,
private environment settings, or new model weights.

## Research boundaries and attribution

This is a preliminary transfer/data-scaling study. Cross-family confounding,
one training seed, development on the validation split, native visual budgets,
and lexical scoring limit the conclusions. External/source-disjoint testing,
repeated seeds, matched optimization budgets and expert evaluation remain needed.
There is no patient-disjoint guarantee from identical-image hashing alone.

SonoInstruct's authors already study Qwen ultrasound adaptation; our contribution
is this approximately size-matched comparison, scaling analysis, and documented
protocol correction, not a first-ever Qwen ultrasound claim.
[Dataset and benchmark paper](https://www.nature.com/articles/s41746-026-02930-w).

Research use only. MedGemma derivatives require the [HAI-DEF terms](HAI_DEF_TERMS.md)
and [NOTICE](NOTICE); obtain base models and dataset access under their respective
terms. See [the model card](MODEL_CARD.md) for intended use and limitations.
