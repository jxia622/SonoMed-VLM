#!/bin/bash
#SBATCH --job-name=sonomed-preempt
#SBATCH --cluster=gpu
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${SONOMED_PROJECT_ROOT:?Set SONOMED_PROJECT_ROOT}"
: "${SONOINSTRUCT_ROOT:?Set SONOINSTRUCT_ROOT}"
: "${SONOMED_OUTPUT_ROOT:?Set SONOMED_OUTPUT_ROOT}"
SONOMED_PYTHON="${SONOMED_PYTHON:-python}"
cd "$SONOMED_PROJECT_ROOT"
nvidia-smi
"$SONOMED_PYTHON" scripts/train.py --config configs/preempt.yaml \
  --resume-from-checkpoint auto-if-present
crc-job-stats || true
