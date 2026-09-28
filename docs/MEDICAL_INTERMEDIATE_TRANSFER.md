# Does additional medical instruction tuning help ultrasound adaptation?

This experiment tests **additional** clinical QA training, starting from the same
`Qwen/Qwen3-VL-4B-Instruct` checkpoint. It does not assume Qwen lacks medical knowledge,
and does not reproduce MedGemma's pretraining. This is decoder LoRA intermediate
instruction tuning, not full-parameter continued pretraining.

## Fixed design

| Arm | Intermediate stage | Downstream stage |
|---|---|---|
| Direct | None | Corrected SonoInstruct open-QA training |
| Medical | MedQA clinical vignette → answer text | Identical ultrasound training |
| General | SQuAD passage/question → answer text | Identical ultrasound training |

Each arm uses the same nested 1%, 10%, and 100% ultrasound subsets and seeds 42, 43,
and 44. All three fractions branch independently from the same intermediate adapter
for that seed. The direct seed-42 adapters and ultrasound evaluations are reused
from the corrected open-QA-v2 experiment only after manifest, protocol, and adapter
hash checks. Its other two seeds are trained afresh.

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
Qwen, each intermediate checkpoint, and all 27 downstream checkpoints. Development
loss uses the separate 128-pair set, never these diagnostic examples. All generation
is greedy with a 256-new-token cap; scoring uses answer-text exact match, token F1,
and ROUGE-L without option lookup. These are open-ended adaptations of the source
benchmarks, not their published MCQ accuracy protocols or clinical adjudications.

Primary contrasts are medical minus direct and medical minus general at each
ultrasound fraction. The final job preserves raw predictions, validates identical
inputs, and writes per-seed values, means, sample SDs, paired seed differences,
task/source breakdowns, CSV, and Markdown. Seed SD is not a significance test.

The general control matches format, tokens, updates, and trainable capacity, but
SQuAD passage extraction differs from clinical reasoning. A gain over this control
supports this particular medical curriculum, not a pure causal isolation of
“knowledge.” No improvement after one fixed intermediate recipe does not establish
that every medical curriculum is useless. Check knowledge acquisition and retention
before interpreting ultrasound transfer. Text-only intermediate adaptation may
also change multimodal alignment despite a frozen vision encoder.

## Deployment

`scripts/prepare_medical_transfer.py` freezes audited data. Then source the private
CRC `run.env` and run `scripts/submit_medical_transfer.py` once. The append-only TSV
receipt prevents silent duplicate deployment after interruption.

A four-GPU smoke job trains both text-only arms for four optimizer steps, reloads
their adapters, continues each into four ultrasound training steps, and generates
text and image-conditioned responses. Successful completion gates six intermediate
jobs, 27 downstream/diagnostic jobs, and an original-Qwen diagnostic job. A final
summary depends on every job succeeding. Failed dependencies are cancelled rather
than allowing an incomplete comparison to appear complete. Smoke adapters are
discarded from the scientific experiment. The submission consists of 36 jobs total.

No monitoring automation is installed. The user will request result collection
after the jobs finish.
