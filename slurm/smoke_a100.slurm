#!/bin/bash
#SBATCH --job-name=sonomed-smoke
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${SONOMED_PROJECT_ROOT:?Set SONOMED_PROJECT_ROOT}"
: "${SONOINSTRUCT_ROOT:?Set SONOINSTRUCT_ROOT}"
: "${SONOMED_OUTPUT_ROOT:?Set SONOMED_OUTPUT_ROOT}"
SONOMED_PYTHON="${SONOMED_PYTHON:-python}"
cd "$SONOMED_PROJECT_ROOT"
nvidia-smi
"$SONOMED_PYTHON" scripts/train.py --config configs/smoke.yaml
"$SONOMED_PYTHON" scripts/baseline_eval.py \
  --config configs/smoke_eval.yaml \
  --max-examples 1 \
  --adapter "$SONOMED_OUTPUT_ROOT/smoke/final_adapter"
crc-job-stats || true
