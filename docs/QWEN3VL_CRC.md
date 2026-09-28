# Qwen3-VL 4B ultrasound comparison

The general vision-language comparator is `Qwen/Qwen3-VL-4B-Instruct`, pinned to
`ebb281ec70b05090aa6165b016eac8ec08e71b17`. Plain Qwen3-4B is text-only.
Official model card: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

Use the original SonoInstruct manifests without rebuilding them: 1,906 examples
for smoke training, 190,625 for full training, and 10,098 held-out examples.
Both stages start independently from the base checkpoint. The full run uses the
same seed 42, one epoch, BF16, decoder-only rank-16 LoRA (alpha 32, dropout 0.05),
learning rate 1e-4, cosine scheduler, weight decay 0.01, warmup 0.03, and effective
batch eight (four GPUs, batch one, accumulation two) as the MedGemma experiment.

Qwen uses its native chat template and image processor, with image area capped at
1,048,576 pixels (up to approximately 1,024 image tokens). This cap bounds training
memory without truncating answers. Tokenization, image processing, architecture,
total parameters and adapter parameter counts differ; this is a comparison of
similarly sized model families, not a causal isolation of medical pretraining.
Medical pretraining does not itself demonstrate overfitting. Record actual
parameter counts from architecture.json and compare task metrics rather than
cross-tokenizer losses. Untouched Qwen and full-adapter held-out generation
evaluations should be run after training, using the existing evaluator.

The initial deployment submitted smoke and full training. Smoke runs one epoch on the original
1% training manifest, validation on the first 128 held-out examples, then reloads
the saved adapter and generates eight answers. Non-finite metric logs, incomplete
training, incorrect trainable parameters, missing adapter files or empty
generations fail the gate. Full training depends on smoke success, starts fresh,
checkpoints every 500 steps and evaluates the full validation split at the end.
A subsequent deployment completed independent 1%, 5%, 10%, 25%, and 50% runs,
each followed by full 10,098-example evaluation. Use `scripts/submit_qwen3vl_scaling.sh`
and the `qwen3vl_scaling_*` configs to reproduce that historical recipe.
The original smoke evaluation is not the 1% research result.
No scheduled monitoring was configured.

CRC staging: `/ix1/kkim/xiac/sonomed-qwen3vl-20260925`.
Outputs: `/ix1/kkim/xiac/sonomed-vlm-outputs/qwen3vl-20260925`.
The staged source is isolated from the MedGemma project; manifests are symlinked.
The existing Python environment is reused unchanged (torch 2.8.0+cu128,
transformers 5.14.1, peft 0.20.0); torchvision 0.23.0+cu128 is installed into the
deployment's private `runtime-deps` overlay because Qwen's processor requires it.
Exact package versions and resolved configs are recorded by each training job.

From the staging directory with the environment exported:

```bash
bash scripts/submit_qwen3vl.sh
```

Submission IDs are saved in `submission.txt`. Full training reserves up to three
days on four RTX PRO 6000 GPUs, smoke up to four hours. A smoke failure cancels the
dependent full job. Resume/retry requires inspecting logs and explicitly selecting
a checkpoint or a fresh output directory; reruns never silently overwrite results.

All reported runs retain the historical MCQ choice-omission limitation. See
[the audit](MCQ_PROMPT_AUDIT_20260927.md) before launching new research training.
The configs reproduce existing runs; they do not implement the proposed corrected protocol.
