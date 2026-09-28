# Corrected open-ended QA experiment (v2)

This experiment answers ultrasound questions without showing candidate answers.
Converted targets contain answer text, never an A/B/C/D identifier. This is a
new task protocol, not a relabeling of the old MCQ accuracy numbers. Results are
pending; the published v1 results remain explicitly historical.

## Correction

- Identify explicit MCQs and other choice-bearing records with an answer label.
  Some source records labeled `qa` still have lettered choice targets.
- Resolve the source reference using the annotated gold choice and verify that
  its answer text agrees. Preserve clinical wording, measurements and symbols.
- Use a concise answer-text instruction instead of the source option-only system
  instruction. Do not add candidate answers to the user prompt.
- Exclude annotations with conflicting references, incomplete questions, or
  questions/answers that rely on seeing choices (e.g. “all of the above”).
  These are conservative rules, not a clinical-quality certification.
- Apply identical, versioned conversion rules to both models. Filter the original
  nested manifests without resampling, retaining the train/validation separation.
  Runtime conversion checks the source again and fails closed on inconsistencies.
- Keep existing grounding and ordinary free-response tasks. Preserve example IDs
  and record the original task type so converted questions can be analyzed separately.

Explicit option prefixes may be removed from predictions for formatting tolerance.
A letter-only prediction gets zero answer credit: the scorer never looks up a
predicted letter in candidate answers. Exact match is case/whitespace normalized
and ignores a final sentence terminator while preserving inequality signs and
decimal points. Token F1 and ROUGE-L are separately named lexical scores, not
clinical-correctness estimates. There is no semantic matching against distractors.

## Runs

Both `Qwen/Qwen3-VL-4B-Instruct` and `google/medgemma-1.5-4b-it` restart from their
pinned original checkpoints at each nominal 1%, 5%, 10%, 25%, 50%, and 100% scale.
They use one epoch, seed 42, decoder rank-16 LoRA, frozen vision/projector,
learning rate 1e-4, and effective batch eight. Both families evaluate only at the
end and use the same corrected validation manifest. Nominal fractions refer to
the original nested subsets; actual counts after filtering are recorded in the audit.

The two 1% runs include full held-out generation and act as gates for larger runs.
Each gate validates finite losses, a completed epoch, frozen vision parameters,
saved adapter identity, all expected evaluation IDs, the new prompt protocol, and
absence of choices/labels in converted records. Gates test execution and protocol,
not a minimum accuracy threshold.

Four additional evaluations use the new protocol: each original model and each
historical full-data adapter. The latter measure an evaluation-only correction;
they do not erase the original adapters' flawed training recipe. Comparing these
with freshly trained adapters separates the effects of the new evaluation prompt
from changes to training targets and filtering, but does not isolate those
training changes from one another.

A final aggregation job on the same Slurm cluster depends on every run.
Scoring is CPU-only, but CRC requires a minimum one-GPU reservation on this cluster;
its time limit is 30 minutes.
It verifies matched input text, references, IDs, protocols and generation settings,
then writes `summary/results.json`, `results.csv`, and `results.md`. It does not
publish new scores to GitHub automatically.

## Audited data counts

| Original fraction | Retained training examples | Converted open-QA examples |
|---:|---:|---:|
| 1% | 1,882 | 983 |
| 5% | 9,251 | 4,630 |
| 10% | 18,761 | 9,566 |
| 25% | 47,173 | 23,739 |
| 50% | 94,118 | 47,307 |
| 100% | 187,761 | 93,548 |

Validation retains **9,964** examples: **4,938** converted open-QA, **4,461**
ordinary QA/open, and **565** grounding. Filtering excludes 2,864 full-training
and 134 validation examples. See [the aggregate audit](../results/openqa_v2_audit.json)
for exclusion reasons and manifest hashes. V2 denominators must not be mixed
with historical v1 language scores.

## Reproduce

```bash
python scripts/prepare_open_qa.py \
  --source-manifests /path/to/original/manifests \
  --data-root "$SONOINSTRUCT_ROOT" \
  --output-dir data/manifests-openqa-v2

# Export project/data/output/Python paths; see the existing CRC guide.
# For evaluation-only bridge runs, also provide both historical adapter paths:
export OPENQA_LEGACY_QWEN_ADAPTER=/path/to/qwen/final_adapter
export OPENQA_LEGACY_MEDGEMMA_ADAPTER=/path/to/medgemma/final_adapter
bash scripts/submit_open_qa_v2.sh
```

Use a fresh staging/output directory. `submission-openqa-v2.tsv` is the submission
receipt. `data/manifests-openqa-v2/audit.json` records source and filtered manifest
hashes, counts and exclusion reasons. Excluded IDs and a small private annotation
review packet stay in CRC storage. Model outputs are also retained there.

## Remaining research needs

This correction enables a well-defined open-ended task but does not complete the
causal research design. Repeated training seeds, paired/group-aware uncertainty,
an untouched external/source-disjoint test set, a Gemma 3 parent-family control,
and a matched tuning/compute-budget sensitivity study remain needed. Single-reference
lexical scoring can penalize valid synonyms and cannot establish clinical safety.
The official SonoBench test package has not been obtained or evaluated.
