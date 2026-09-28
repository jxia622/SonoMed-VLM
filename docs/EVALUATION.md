# Evaluation protocol

SonoMed-VLM uses deterministic generation and preserves each prompt, reference,
raw output, parsed output, generation setting, task family, source, and focus.
Saved JSONL outputs can therefore be re-scored without loading a model.

> **2026-09-27 protocol audit:** All published runs used MCQ prompts without the
> question-specific answer choices. These scores are diagnostic answer matching,
> not conventional MCQ accuracy. See [the audit](MCQ_PROMPT_AUDIT_20260927.md).
> Grounding is not directly affected by this omission.

## Multiple choice

The evaluator retains these separate diagnostic metrics for the historical protocol:

- **Strict label accuracy:** the emitted A/B/C/D label matches the gold label.
- **Option-text accuracy:** the emitted answer text matches the correct option
  after lowercase, whitespace, prefix, and harmless-punctuation normalization.
- **Semantic-choice accuracy:** a unique question-specific option phrase is
  resolved first; otherwise a valid explicit label is mapped to the associated
  option. Unresolved outputs count as incorrect.
- **Label/text consistency:** label and text refer to the same option.
- **Contradiction rate:** label and text resolve to different options.
- **Valid-response rate:** at least one supported representation is parseable.

Option text takes precedence only for the separate semantic-choice metric. A
contradiction such as `A: Kidney`, when Kidney is option B, can be semantically
correct while remaining strictly wrong and contradictory. Existing strict and
option-text metrics are never overwritten.

## Open response and QA

Free-response records receive normalized exact match and token F1. Report-like
outputs additionally receive ROUGE-L F1. These lexical metrics measure overlap,
not clinical correctness, factuality, or safety.

## Visual grounding

SonoInstruct serializes bounding boxes in a `[0,1000]` coordinate convention.
The evaluator accepts that convention, normalizes both reference and prediction
to `[0,1]` for mathematically equivalent scoring, and reports:

- valid-box and invalid-box rates;
- mean intersection-over-union (IoU);
- Localization@0.5, the fraction with IoU at least 0.5.

The original evaluator incorrectly accepted only `[0,1]` input coordinates.
Release results were regenerated with the corrected parser and stored separately
from the original artifacts.

## Aggregation

Metrics are aggregated only over applicable records and are available overall,
by task family, task type, focus, and source. The release split contains 10,098
examples: 4,905 MCQ-labeled records, 4,628 QA/open responses, and 565 grounding examples.
Explicit task labels take precedence over option metadata; 167 QA records with
options remain QA. Historical summaries used different routing and pooling; use
[the common-scorer research results](../results/research_results.json).

Generate outputs with `scripts/baseline_eval.py`, or re-score saved outputs:

```bash
python scripts/evaluate.py \
  --predictions /path/to/eval_predictions.jsonl \
  --output-dir /path/to/corrected-evaluation
```

External benchmarks must remain outside training, and test data must not be used
for hyperparameter selection. Licensing and task compatibility must be verified
before adding a benchmark.
