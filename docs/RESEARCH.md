# Does medical specialization improve ultrasound adaptation?

**Research report · 27 September 2026 · Preliminary, single-seed empirical study**

## Research question

At a similar nominal model size, does a medically specialized vision-language model
provide better ultrasound adaptation and data efficiency than a general-purpose
vision-language model under the same decoder-LoRA training recipe? Does the answer
differ between spatial grounding and language output?

We compare `google/medgemma-1.5-4b-it` with
`Qwen/Qwen3-VL-4B-Instruct`. “Original” means the released instruction-tuned
checkpoint before our ultrasound adaptation, not an untrained or pretrained-only model.

## Motivation and competing explanations

We began with MedGemma because medical specialization might provide useful clinical
knowledge and image representations. Our motivating concern was whether experience
with other medical modalities transfers to ultrasound's different appearance. A
strong general VLM might instead provide more useful spatial representations for
this specific adaptation task.

**This is a transfer hypothesis, not evidence that MedGemma overfit CT or MRI.**
Google describes broad medical text, question-answer and image training; CT/MRI are
among several modalities. Its public model card does not justify calling CT/MRI
the primary training data. We also cannot establish that Qwen never encountered
medical data. [Google model card](https://huggingface.co/google/medgemma-1.5-4b-it).

Three explanations remain compatible with our observations:

1. Medical specialization may help language knowledge without improving ultrasound localization.
2. General spatial pretraining may transfer well: Qwen3-VL explicitly trains on grounding
   and uses normalized 0–1000 coordinates, compatible with this task's output convention.
   This is a plausible explanation, not an ablation result.
   [Qwen3-VL technical report, §3.2.4](https://arxiv.org/html/2511.21631v1#S3.SS2.SSS4).
3. Architecture, visual token budgets, adapter capacity, native prompts, or optimization
   may explain the gap independently of medical specialization.

Qwen versus MedGemma changes all of these factors together. It does not identify
the causal effect of medical QA pretraining. A closer control is Gemma 3 4B versus
MedGemma, supplemented by controlled pretraining ablations if accessible.

## Study design

| Factor | Protocol |
|---|---|
| Adaptation data | SonoInstruct; 190,625 examples in the full training manifest |
| Evaluation | Same 10,098-example held-out validation split for every reported run |
| Task denominators | 565 grounding; 4,628 QA/open; 4,905 MCQ-labeled diagnostic records |
| Qwen fractions | 1%, 5%, 10%, 25%, 50%, 100%, plus original checkpoint |
| MedGemma fractions | 5%, 10%, 25%, 50%, 100%, plus original checkpoint |
| Initialization | Each fraction starts independently from its family's pinned checkpoint |
| Optimization | One epoch, seed 42, AdamW, LR 1e-4, cosine schedule, warmup 0.03, weight decay 0.01 |
| Effective batch | 8: four GPUs × batch one × two accumulation steps |
| Adaptation | BF16; decoder LoRA rank 16, alpha 32, dropout 0.05; vision/projector frozen |
| LoRA targets | Decoder q/k/v/o and gate/up/down projections |
| Generation | Greedy, maximum 256 new tokens, native model chat templates |
| Qwen image budget | Native processor capped at 1,048,576 pixels |
| Hardware | Four RTX PRO 6000 Blackwell GPUs for the scale runs |

The nominal fractions contain **1,906 / 9,379 / 19,035 / 47,860 / 95,544 /
190,625** examples. Fractions are approximate because image-connected groups stay
together. The training subsets are nested. Original image-byte hashes link records
into components before splitting, preventing identical-image leakage under that
hash definition. This is not a guarantee against near duplicates, shared patients,
or source-level overlap. No patient-disjoint or external-site claim is made.

The 1% Qwen result comes from the new full-evaluation run, not the earlier smoke
check (128 validation examples and eight generated answers). There is no comparable
full-split 1% MedGemma result; we leave that cell empty.

The common recipe controls examples and several hyperparameters, but does not
match FLOPs, visual-token counts, trainable-parameter counts, or model-specific
optimal hyperparameters. At one epoch, more examples also mean more optimizer
updates. This study therefore measures data scaling under a fixed-epoch recipe,
not a pure data effect at fixed compute. Losses use different tokenizers and are
not treated as a cross-model quality ranking. Recorded trainable adapter counts
are 33,030,144 for Qwen and 29,802,496 for MedGemma. The post-injection total
parameter counts are 4,470,845,952 and 4,329,881,968 respectively; “4B” is a
nominal model-size class, not exact parameter equality.

## Results: visual grounding

![Grounding before and after adaptation](assets/research/grounding_comparison.png)

| Model | Adaptation | Mean IoU | Localization@0.5 |
|---|---|---:|---:|
| Qwen3-VL 4B | Original | 0.1444 | 16.81% |
| Qwen3-VL 4B | 100% LoRA | 0.7449 | 86.90% |
| MedGemma 1.5 4B | Original | 0.0894 | 5.31% |
| MedGemma 1.5 4B | 100% LoRA | 0.5831 | 71.15% |

At 100% data, Qwen's mean IoU is **0.7449 versus 0.5831**, an absolute difference
of **0.1618** and a relative increase of **27.75%**. Localization@0.5 is **86.90%
versus 71.15%**, a **15.75 percentage-point** difference. Relative to their own
original checkpoints, Qwen reaches **5.16×** mean IoU and MedGemma reaches **6.52×**.
A larger multiplier from a lower baseline does not imply a higher final score.

![Grounding learning curves](assets/research/grounding_scaling.png)

| Data | Examples | Qwen IoU | MedGemma IoU | Qwen Loc@0.5 | MedGemma Loc@0.5 |
|---:|---:|---:|---:|---:|---:|
| 1% | 1,906 | 0.3673 | — | 33.81% | — |
| 5% | 9,379 | 0.6143 | 0.3020 | 74.16% | 24.96% |
| 10% | 19,035 | 0.6585 | 0.3231 | 81.24% | 32.04% |
| 25% | 47,860 | 0.7006 | 0.3615 | 84.07% | 37.17% |
| 50% | 95,544 | 0.7258 | 0.4419 | 85.66% | 46.90% |
| 100% | 190,625 | 0.7449 | 0.5831 | 86.90% | 71.15% |

Qwen has higher observed mean IoU and Localization@0.5 at every shared tested
training fraction. At **5% (9,379 examples)**, Qwen reaches **0.6143 mean IoU**
and **74.16% Localization@0.5**, exceeding MedGemma's full-data observations
(**0.5831**, **71.15%**, 190,625 examples). This uses approximately **20.3× fewer
training examples**. Five percent is the first tested Qwen fraction that exceeds
that endpoint; the exact crossing point is unknown. This is not a demonstrated
20× compute saving or a statistical significance claim.

## Results: language output and the MCQ audit

![Language metrics and explicitly provisional answer matching](assets/research/language_diagnostics.png)

| Model | Adaptation | QA/open token F1 | Answer-content matching† |
|---|---|---:|---:|
| Qwen3-VL 4B | Original | 0.1924 | 12.80% |
| Qwen3-VL 4B | 100% LoRA | 0.3535 | 93.84% |
| MedGemma 1.5 4B | Original | 0.1880 | 17.59% |
| MedGemma 1.5 4B | 100% LoRA | 0.3496 | 93.52% |

† Diagnostic only: choices omitted; not standard MCQ accuracy.

At full data, pooled QA/open token F1 is **0.3535 for Qwen versus 0.3496 for
MedGemma**. Diagnostic semantic matching is **93.84% versus 93.52%**. These small
observed differences do not establish superiority, equivalence, or clinical quality.
The grounding advantage should not be described as an across-the-board win.

**The MCQ recipe has a known input defect.** Choices were stored in metadata but
not appended to model inputs during either training or inference. An audit of all
4,905 fine-tuned Qwen MCQ prompts found none containing all the literal options.
The evaluator could access choices the model did not see. The semantic score
therefore describes answer-content matching under that protocol, not standard MCQ
accuracy. The approximately 70% label/text contradiction rate cannot be presented
as evidence of weak model label binding. See [the evidence and consequences](MCQ_PROMPT_AUDIT_20260927.md).

Grounding inputs are not directly affected by missing MCQ choices. Nevertheless,
MCQ examples participated in the shared training mixture, so corrected retraining
could change even the grounding results. Existing runs remain a documented historical
recipe; no evaluation-only patch can retroactively repair their training inputs.

## What this study supports

**Within this split and training recipe, medical specialization was not sufficient
to outperform the general Qwen model on ultrasound grounding. Qwen achieved higher
observed grounding scores with fewer training examples.**

This supports testing transfer at the task and modality level. It does not show
that medical QA pretraining is useless, that MedGemma overfit CT/MRI, or that Qwen
will outperform MedGemma on other medical tasks. Qwen already had a stronger
original grounding score, and it improved further after adaptation.

## Related work and the contribution

Jeong et al. compared medically adapted models with their parent models and found
that specialization did not consistently improve medical QA, including after
supervised fine-tuning. The broad question is therefore not new.
[The Limited Impact of Medical Adaptation of Large Language and Vision-Language Models](https://arxiv.org/abs/2411.08870).

The SonoInstruct authors already fine-tuned Qwen3-VL-2B and evaluated it on
SonoBench. Fine-tuning Qwen on this dataset alone is not our novelty claim.
[A multimodal instruction dataset and benchmark for ultrasound understanding](https://www.nature.com/articles/s41746-026-02930-w).

The contribution here is a reproducible, approximately size-matched **ultrasound
adaptation and data-scaling comparison**, separating grounding from language
metrics and documenting a consequential evaluation defect. A stronger research
claim requires the additional controls below. We make no first-in-the-literature
claim and do not present this repository as a completed causal study.

## Limitations and experiments needed

| Open issue | Required follow-up | What it would resolve |
|---|---|---|
| Missing MCQ choices | Version a corrected prompt/target protocol; check labels and option permutations; retrain and evaluate consistently | Valid MCQ conclusions |
| Cross-family confounding | Add Gemma 3 4B; compare against MedGemma under matched tuning budgets | A closer estimate of medical-specialization benefit |
| Single seed and one split | Repeat training seeds; paired uncertainty estimates grouped by image/patient where available | Stability and uncertainty of differences |
| Validation reused throughout development | Freeze protocol, then evaluate untouched source/patient/external data | Generalization beyond this validation set |
| Native visual budgets and different adapters | Measure and control visual tokens, trainable capacity, and compute; test frozen/unfrozen vision | Sources of the grounding advantage |
| One epoch and fixed LR | Comparable per-family tuning budget, epoch sensitivity, fixed-update controls | Recipe robustness and optimization confounding |
| Lexical language metrics | Blinded expert review or validated task-specific scoring | Clinical correctness and hallucination assessment |

**SonoBench status:** none of these results are official SonoBench scores. On
27 September 2026, we checked the [official repository](https://github.com/ShiDaizi/SonoInstruct)
and [Hugging Face release](https://huggingface.co/datasets/Ssdaizi/SonoInstruct/tree/main)
but could not locate the official test package or executable evaluator. Obtaining
those resources and checking overlap is a next step, not an experiment already run.

## Reproducibility and provenance

The public release contains aggregate results and code, not raw prompts, reference
answers, ultrasound images, or model weights. Each run in
[`research_results.json`](../results/research_results.json) records prediction,
metadata and config hashes, model revisions, metric denominators, and per-task,
source and focus aggregates. Raw artifacts remain in the authorized CRC storage.

- Qwen revision: `ebb281ec70b05090aa6165b016eac8ec08e71b17`
- MedGemma revision: `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`
- Validation manifest SHA-256: `1c126056347b97d300f32e3becb0008352f45731c982cdd442ae27f4617c95a0`
- Qwen scale jobs: 4060375 (1%), 4060376 (5%), 4060377 (10%), 4060378 (25%), 4060379 (50%); all completed successfully.
- Full comparison uses the original and 100% Qwen generations from the prior matched evaluation; MedGemma generations are reused and rescored.

The exporter rejects duplicate/missing IDs, mismatched manifests, question/reference/
option/task metadata, and inconsistent generation settings. It recomputes scores
from raw output instead of trusting cached scores. Explicit task labels take
precedence over option metadata; 167 QA examples carrying choices remain QA.
This changes historical task denominators and pooled language metrics.

```bash
# Aggregate archived raw runs locally or on CRC; CPU only.
python scripts/build_research_results.py \
  --runs configs/research_runs.json \
  --outputs-root /path/to/sonomed-vlm-outputs \
  --manifest data/manifests/val.jsonl \
  --output-dir results

# Rebuild every research figure from the public aggregate file.
python -m pip install -e '.[viz]'
python scripts/plot_research_results.py
```

See [training and CRC setup](QWEN3VL_CRC.md), [evaluation details](EVALUATION.md),
and [the matched-comparison checks](QWEN_MEDGEMMA_COMPARISON.md). PNGs are embedded
for GitHub; SVG and PDF versions are alongside them for reuse.
