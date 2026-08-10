# Training protocol

## Final configuration

The released adapter uses BF16 supervised fine-tuning with decoder-only LoRA.
The vision tower and multimodal projector remain frozen. LoRA targets are
discovered from the installed model architecture, filtered to decoder modules,
and persisted in `architecture.json`.

| Setting | Value |
|---|---:|
| Base model | `google/medgemma-1.5-4b-it` |
| Base revision | `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b` |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target suffixes | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Trainable parameters | 29,802,496 across 238 decoder modules |
| Epochs | 1 |
| Optimizer / schedule | AdamW / cosine |
| Learning rate / weight decay | `1e-4` / `0.01` |
| Per-device batch / accumulation | 1 / 2 |
| GPUs / effective batch | 4 / 8 |
| Precision | BF16 with TF32 enabled |
| Seed | 42 |

Every scale starts independently from the same base revision. No smaller-scale
adapter or checkpoint initializes a larger-scale run.

## Prompts and assistant-only loss

The official processor `apply_chat_template` path formats all messages. The
collator separately tokenizes the prompt-only conversation and the full
conversation, then masks the prompt prefix and padding with `-100`. Only
assistant tokens contribute to loss. Images receive only the processor-required
decode, resize, and normalization operations; no random augmentation is used.

## Distributed execution

The pipeline uses Transformers Trainer and PyTorch DDP through `torchrun`.
Effective batch size is:

```text
per-device batch × gradient accumulation × distributed world size
```

For the final four-GPU run:

```bash
torchrun --standalone --nproc_per_node=4 \
  scripts/train.py --config configs/rtx6k_train_100pct.yaml
```

Slurm examples are portable templates: set `SONOMED_PROJECT_ROOT`,
`SONOINSTRUCT_ROOT`, `SONOMED_OUTPUT_ROOT`, and optionally `SONOMED_PYTHON` and
`SONOMED_CONFIG` in the submitting environment. No cluster identity or private
storage path is embedded in the public repository.

## Checkpointing and provenance

Trainer checkpoints contain adapter state, optimizer, scheduler, RNG, and
trainer state. Checkpoint discovery accepts only numerically named, complete
checkpoints whose recorded global step matches their directory. Stable run
directories support explicit `latest` or `auto-if-present` resume behavior.

Each run saves the resolved configuration, architecture and trainable-parameter
report, environment/package snapshots, manifest hashes, seed, model and dataset
revision, JSONL training log, final metrics, wall time, throughput, and compute
accounting. Generated outputs, checkpoints, logs, and weights are ignored by Git.

## Controlled hardware benchmark

The A100 and RTX PRO 6000 comparison used four GPUs, the same fixed subset,
effective batch eight, and 50 optimizer steps. Measured throughput was 1.464 and
2.959 examples/second respectively (2.02×). This comparison did not use or claim
a custom Blackwell kernel.
