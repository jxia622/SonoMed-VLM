#!/bin/bash
#SBATCH --job-name=sonomed-1gpu
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${SONOMED_PROJECT_ROOT:?Set SONOMED_PROJECT_ROOT}"
: "${SONOINSTRUCT_ROOT:?Set SONOINSTRUCT_ROOT}"
: "${SONOMED_OUTPUT_ROOT:?Set SONOMED_OUTPUT_ROOT}"
SONOMED_PYTHON="${SONOMED_PYTHON:-python}"
SONOMED_CONFIG="${SONOMED_CONFIG:-configs/train_1pct.yaml}"
cd "$SONOMED_PROJECT_ROOT"
nvidia-smi
monitor_prefix="logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-gpu"
bash scripts/gpu_monitor.sh "$monitor_prefix" &
monitor_pid=$!
cleanup_monitor() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup_monitor EXIT
"$SONOMED_PYTHON" scripts/train.py --config "$SONOMED_CONFIG"
crc-job-stats || true
