#!/bin/bash
#SBATCH --job-name=sonomed-rtx6k-probe
#SBATCH --cluster=gpu
#SBATCH --partition=rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${SONOMED_PROJECT_ROOT:?Set SONOMED_PROJECT_ROOT}"
SONOMED_PYTHON="${SONOMED_PYTHON:-python}"
GPU_COUNT="${SLURM_GPUS_ON_NODE:-4}"
GPU_COUNT="${GPU_COUNT##*:}"
PROBE_OUTPUT="${SONOMED_OUTPUT_ROOT:-$SONOMED_PROJECT_ROOT/outputs}/distributed-probe-${SLURM_JOB_ID}"
cd "$SONOMED_PROJECT_ROOT"
nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader
"$SONOMED_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
  scripts/rtx6k_probe.py \
  --output-dir "$PROBE_OUTPUT" \
  --medgemma-config configs/base.yaml
crc-job-stats || true
