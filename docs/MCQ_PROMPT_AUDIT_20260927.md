# MCQ prompt audit — 2026-09-27

During research-question assessment, a read-only audit found that MCQ choices
are retained in metadata but are not appended to the model's input. This affects
the shared training and inference message builder. It is separate from the
previous correction to task-routing denominators.

Evidence:

- `src/sonomed_vlm/data/sonoinstruct.py` sets `user_prompt` from the question
  and stores `options` separately in metadata.
- `src/sonomed_vlm/data/collator.py:21` supplies `example.user_prompt` as user
  text without adding the options. Both training and evaluation use this builder.
- All 4,905 saved Qwen fine-tuned MCQ user prompts were checked. Zero contained
  all their literal option strings; zero contained both A and B choice markers.
- The first three MCQ source examples were also inspected. Their system prompts
  requested an option-only answer but contained no choices.
- One saved example asks which body area is shown. The separately stored choices
  are A: Liver, B: Uterus, C: Kidney, D: cervix. The reference is C: Kidney; the
  generated answer is A: Kidney. The model never receives that option mapping in
  the inspected example.

Consequences:

- Current strict-label accuracy and label/text contradiction rates must not be
  interpreted as ordinary MCQ performance or evidence of label-binding failure.
- Current semantic-choice results measure generated answer content resolved using
  choices available to the evaluator; they are not conventional MCQ accuracy.
- The omission applies to the shared recipe used for the scaling experiments.
- Grounding scores are not directly invalidated by this particular omission.
- Earlier assistant statements attributing the approximately 70% contradiction
  rate to model behavior need this correction.

Recommended follow-up: define and version a corrected MCQ protocol that presents
the full question-specific choices, checks target-label consistency and tests
option permutations. Evaluate all original and adapted models consistently.
Training with omitted choices remains a confound even if inference prompts are
fixed; a corrected training comparison should be distinguished from an
evaluation-only repair. Preserve the existing runs as the original recipe.

No training code or queued CRC jobs were changed during this research assessment.
