# Evaluation — corrected open-QA v2

The current [research release](RESEARCH.md) contains 16 verified evaluations on the
same 9,964-example internal validation manifest. See [the protocol](OPEN_QA_V2.md)
for conversion and training, and [the audit](../results/openqa_v2_audit.json) for hashes.

## Task-specific metrics

| Subset | Count | Metrics |
|---|---:|---|
| Converted open-ended QA | 4,938 | Answer-text exact match, token F1, ROUGE-L; empty/label-only rate |
| Ordinary QA/open response | 4,461 | Token F1, ROUGE-L, normalized exact match |
| Visual grounding | 565 | Mean IoU, Localization@0.5, valid/invalid box rate |

Converted questions show no candidate answers. Targets contain verified answer text,
never A/B/C/D labels. Case/whitespace normalization and a final sentence terminator
are tolerated; inequality signs and decimal points are preserved. An explicit option
prefix can be stripped mechanically, but a letter-only prediction receives zero credit.
The scorer never looks up a predicted letter in hidden candidates. These are open-QA
metrics, not standard MCQ accuracy. Lexical overlap is not clinical correctness.

Grounding uses the common parser and normalization for SonoInstruct's [0,1000]
coordinate convention. IoU is intersection area divided by union area; Localization@0.5
is the fraction with IoU at least 0.5. Invalid/missing boxes receive zero IoU and remain
in the denominator. Means are per example, not pixel- or patient-weighted.

All generation is greedy with a 256-new-token cap. Each model retains its native
chat template and image processor. Thus equal output settings do not equalize visual
compute or tokenization. No checkpoint is selected using these generation scores.

## Integrity checks and aggregation

`check_open_qa_run.py` verifies IDs, prompts, targets, options absence for converted
records, manifest hashes and saved adapter provenance. `summarize_open_qa.py` compares
input signatures across all registered runs and recomputes scores from raw output.
The public JSON includes denominators, prediction SHA-256s, task/source/focus breakdowns.
Do not pool converted-QA exact match and ordinary-QA token F1 into a single accuracy.

The 12 corrected adapters, two originals, and two legacy-adapter bridge controls all
use the same v2 validation protocol. Only corrected adapters enter learning curves.
Each full curve uses seed 42; no uncertainty or significance estimate is available.
Identical-image-connected train/validation separation is not patient/site separation.
These are not official SonoBench or external clinical results.

## Historical protocol

The earlier v1 recipe omitted choices while preserving letter targets and option-aware
answer matching. Its archived metrics are diagnostic only. See [the original audit](MCQ_PROMPT_AUDIT_20260927.md)
and [historical exports](../results/archive/README.md). V1 and v2 language denominators
and scorers differ; their percentages are not interchangeable. The original published
Hugging Face adapter remains a v1 adapter, even when evaluated with v2 prompts.

## Completed same-Qwen intermediate-training pilot

The separate [medical-training pilot](MEDICAL_INTERMEDIATE_TRANSFER.md) reuses the
same 9,964-example corrected ultrasound validation set. Three arms × three fractions
produce nine downstream checkpoints; each is also evaluated on 500 held-out medical
and 500 general text questions. Original Qwen and the two intermediate checkpoints
add six diagnostic evaluations, for 33 evaluation records total.

The final aggregation job completed successfully and verified identical inputs within
each domain. The [pilot JSON](../results/medical_transfer_pilot/results.json) is
separate from the 16-run cross-family release. Seed 42 only; standard deviations
remain null. These text diagnostics are open-ended adaptations of MedQA and SQuAD,
not their original benchmark protocols or measures of clinical correctness.
