# Four-model held-out comparison

> **MCQ protocol limitation:** Question-specific choices were omitted from model inputs.
> These metrics are diagnostic only; see [the audit](MCQ_PROMPT_AUDIT_20260927.md).
> The current results and interpretation are in [the research report](RESEARCH.md).

The comparisons are:

1. Fine-tuned Qwen3-VL-4B versus its original instruction-tuned checkpoint.
2. Fine-tuned Qwen3-VL-4B versus the full-data MedGemma 1.5 4B adapter.
3. Original Qwen3-VL-4B versus original MedGemma 1.5 4B.

All four use the same 10,098 validation example IDs, original user prompts,
references and options. Qwen uses its native processor and training-time image
area cap. Generation is greedy with a 256-token maximum. The existing full
MedGemma raw predictions have the same manifest SHA-256 and generation settings,
so they are reused. Scores are recomputed for every model with one scorer version;
historical metric summaries are not copied into this comparison.

The audit found 167 records explicitly labeled QA that retain option metadata.
The task router now respects explicit labels, giving 4,905 MCQ, 4,628 QA/open and
565 grounding examples. MCQ strict label, exact option text, semantic choice,
contradiction and invalid-response rates remain separate. QA/open metrics are
pooled in the main table and also broken down by task type in JSON. Do not compare
pooled scores to historical open-only scores. Historical published strict-label
numbers also require a separate audit; use freshly rescored values for this study.

The comparison script rejects duplicate/missing examples, mismatched prompts,
references, task labels, options, manifest checksums or generation settings. The
report includes all three paired differences, applicable sample counts, and
per-task/source/focus metrics. It records input and scorer checksums. It does not
claim significance or external generalization.

CRC project: `/ix1/kkim/xiac/sonomed-qwen3vl-20260925`.
Output root: `/ix1/kkim/xiac/sonomed-vlm-outputs/qwen-medgemma-comparison-20260926`.

`slurm/compare_qwen_medgemma.slurm` runs original Qwen generation, adapter Qwen
generation, and then builds `comparison/comparison.md` and `comparison.json`.
Each Qwen evaluation keeps raw outputs, metrics, resolved configuration, runtime
metadata, and architecture details in its own directory. Adapter identity and
checksum are recorded for the new fine-tuned Qwen evaluation.

Submission receipt: `submission-comparison.txt`; environment: `comparison.env`.
The job has a 12-hour maximum allocation, not an expected runtime. The report is
automatically generated after successful inference; no recurring monitor is set.
