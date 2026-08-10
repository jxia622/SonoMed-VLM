#!/bin/bash
#SBATCH --job-name=sonomed-2gpu
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${SONOMED_PROJECT_ROOT:?Set SONOMED_PROJECT_ROOT}"
: "${SONOINSTRUCT_ROOT:?Set SONOINSTRUCT_ROOT}"
: "${SONOMED_OUTPUT_ROOT:?Set SONOMED_OUTPUT_ROOT}"
SONOMED_PYTHON="${SONOMED_PYTHON:-python}"
SONOMED_CONFIG="${SONOMED_CONFIG:-configs/train_1pct.yaml}"
GPU_COUNT="${SLURM_GPUS_ON_NODE:-2}"
GPU_COUNT="${GPU_COUNT##*:}"
cd "$SONOMED_PROJECT_ROOT"
nvidia-smi
"$SONOMED_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
  scripts/train.py --config "$SONOMED_CONFIG"
crc-job-stats || true
