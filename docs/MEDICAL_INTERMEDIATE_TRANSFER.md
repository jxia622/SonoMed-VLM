# Does additional medical instruction tuning help ultrasound adaptation?

This experiment tests **additional** clinical QA training, starting from the same
`Qwen/Qwen3-VL-4B-Instruct` checkpoint. It does not assume Qwen lacks medical knowledge,
and does not reproduce MedGemma's pretraining. This is decoder LoRA intermediate
instruction tuning, not full-parameter continued pretraining.

## Current status and budget

The active deployment is a **single-seed pilot (42)** with three sequential arm
bundles and one final summary, reusing the existing smoke test: **five jobs total**
instead of 36. Seeds 43 and 44 are deferred, not silently treated as completed.
No intermediate-training results have been reported. The corrected cross-family
Qwen/MedGemma results are a separate [completed study](RESEARCH.md).

This reduces new ultrasound training from 24 runs to six (**75% fewer**); direct
seed-42 adapters are reused in both plans. Intermediate runs fall from six to two.
The expected total training reduction is roughly 70–75%, not a measured billing
saving. Bundling alone saves little GPU time; dropping two pilot seeds is the
main saving. The pilot cannot estimate between-seed variance.

## Fixed design

| Arm | Intermediate stage | Downstream stage |
|---|---|---|
| Direct | None | Corrected SonoInstruct open-QA training |
| Medical | MedQA clinical vignette → answer text | Identical ultrasound training |
| General | SQuAD passage/question → answer text | Identical ultrasound training |

Each arm uses the same nested 1%, 10%, and 100% ultrasound subsets and seed 42.
The original three-seed plan (42, 43, 44) remains available for a later extension. All three fractions branch independently from the same intermediate adapter
for that seed. The direct seed-42 adapters and ultrasound evaluations are reused
from the corrected open-QA-v2 experiment only after manifest, protocol, and adapter
hash checks. No new direct-arm ultrasound training is needed for this pilot.

The model revision is `ebb281ec70b05090aa6165b016eac8ec08e71b17`.
Both stages use rank-16 decoder LoRA (alpha 32, dropout 0.05), frozen vision and
projector, BF16, AdamW, LR 1e-4, cosine schedule, warmup 0.03, weight decay 0.01,
and effective batch 8 across four RTX PRO 6000 GPUs. Each stage is one epoch.
The downstream stage continues the **same** adapter weights with a fresh optimizer
and schedule. It neither stacks adapters nor increases trainable parameter count.
No checkpoint is selected by evaluation score. Longer intermediate training or
different learning rates would be separate experiments.

## Intermediate data and contamination checks

Sources are pinned and their downloaded files SHA-256 hashed:

- `GBaker/MedQA-USMLE-4-options`, revision `0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18`.
- `rajpurkar/squad`, revision `7b6d24c440a36b6815f21b70d25016731768db1f`.

Only the upstream training splits contribute training examples. Medical questions
receive a mechanical “which of the following” → “what” rewrite. Remaining
choice-dependent, negative-selection, and image-dependent questions are excluded;
correct answer text is verified against the source annotation. Candidate answers
and option letters are never shown to a model. The same concise answer-only system
instruction is used for medical and general intermediate data.

SQuAD contexts containing selected explicit medical terms are excluded. This is a
general reading-QA control, not a guarantee of zero biological knowledge. Medical
and general examples are paired with identical supervised-answer token counts and
total sequence lengths within 5%. Aggregate prompt, answer, and total token budgets
must match within 2%; example counts and optimizer updates are identical. Pair
counts are rounded down to multiples of eight to avoid distributed padding.
128 matched pairs are reserved for fixed end-of-training loss evaluation.

All released SonoInstruct questions are indexed, including material outside our
selected splits. Intermediate training and medical/general diagnostic questions
are excluded if they match a SonoInstruct question exactly after normalization,
or trigger the documented lexical near-duplicate screen. Training is also screened
against the external diagnostic pools and duplicate questions within each source.
The near-duplicate screen retrieves candidates with up to 20 rare shared word
5-grams, then requires at least ten shared 5-grams and 80% containment. This is not
an assurance against paraphrases, translations, or unknown Qwen pretraining exposure.

The authoritative `data/medical-transfer/audit.json` on CRC records selected counts,
exclusions, source revisions, token totals, and every experiment manifest hash.
Data contents remain on CRC; no clinical question data are committed to Git.

## Evaluation and interpretation

The unchanged corrected ultrasound validation manifest has 9,964 examples:
4,938 converted open-QA, 4,461 other QA/open responses, and 565 grounding examples.
This is internal validation, not SonoBench or an untouched external ultrasound test.
Report converted QA, original QA, task/source groups, and grounding separately.

Medical and general held-out diagnostics use up to 500 eligible examples each from
the upstream MedQA test and SQuAD validation splits. They are evaluated on original
Qwen, each intermediate checkpoint, and all nine pilot downstream checkpoints. Development
loss uses the separate 128-pair set, never these diagnostic examples. All generation
is greedy with a 256-new-token cap; scoring uses answer-text exact match, token F1,
and ROUGE-L without option lookup. These are open-ended adaptations of the source
benchmarks, not their published MCQ accuracy protocols or clinical adjudications.

Primary contrasts are medical minus direct and medical minus general at each
ultrasound fraction. The final job preserves raw predictions, validates identical
inputs, and writes per-run values, paired arm differences, task/source breakdowns, CSV,
and Markdown. With one seed, sample SD is **null**, not zero, and the output is
explicitly marked `single_seed_pilot`. A later multi-seed summary can report sample
SD, which is still not a significance test.

The general control matches format, tokens, updates, and trainable capacity, but
SQuAD passage extraction differs from clinical reasoning. A gain over this control
supports this particular medical curriculum, not a pure causal isolation of
“knowledge.” No improvement after one fixed intermediate recipe does not establish
that every medical curriculum is useless. Check knowledge acquisition and retention
before interpreting ultrasound transfer. Text-only intermediate adaptation may
also change multimodal alignment despite a frozen vision encoder.

## Deployment

`scripts/prepare_medical_transfer.py` freezes audited data. Source the private CRC
`run.env`; submit a smoke test using `slurm/medical_transfer.slurm smoke`, then:

```bash
python scripts/submit_medical_transfer_pilot.py --smoke-job YOUR_SMOKE_JOB_ID
```

The append-only `submission-medical-transfer-pilot.tsv` receipt refuses duplicate
submission. The smoke must belong to the same frozen stage and output directory;
each experiment process independently checks its success marker and data-audit hash.
The original 36-job launcher remains available for a deliberate future full repeat;
do not use both launchers against the same outputs.

A four-GPU smoke trains both text arms for four steps, reloads adapters, continues
each into four ultrasound steps, and generates text/image responses. Its adapters
are excluded from the experiment. The pilot then runs:

| Job | Sequential work | GPU request | Wall-time limit |
|---|---|---:|---:|
| Direct bundle | Original knowledge diagnostics; reuse 1/10/100% direct adapters and add knowledge diagnostics | 4 | 4h |
| Medical bundle | Medical intermediate training/diagnostics; independent 1/10/100% ultrasound runs/diagnostics | 4 | 24h |
| General bundle | General intermediate training/diagnostics; independent 1/10/100% ultrasound runs/diagnostics | 4 | 24h |
| Summary | Validate all nine downstream comparisons and six original/intermediate diagnostic evaluations | 1 (CRC minimum) | 30m |

At most 12 GPUs are requested concurrently by the three bundles. Within each
bundle, stages execute sequentially and stop on failure. **The 10% run does not
continue the 1% adapter:** each fraction starts independently from the same
intermediate adapter (or the original model for direct tuning). All diagnostic
comparisons and token-matched general controls are preserved.

The summary requires all three bundles to complete. Invalid dependencies cancel
it instead of emitting an incomplete result. One seed is passed explicitly to the
summarizer. If a bundle fails, inspect and recover its completed outputs before
resubmitting; the runner intentionally refuses to overwrite an existing run.

The most recent corrected full-data Qwen train/evaluation took about 10.5 hours.
A medical/general bundle is estimated at **14–20 hours after it starts**, including
intermediate training, smaller fractions and additional diagnostics. Queue delay
is unknown, and a 24h reservation is a limit, not an ETA. All bundles running
concurrently would finish within roughly that range plus the summary; scheduling
may spread them over longer. No monitoring automation is installed.
