#!/bin/bash
#SBATCH --job-name=sonomed-rtx6k-eval
#SBATCH --cluster=gpu
#SBATCH --partition=rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${SONOMED_PROJECT_ROOT:?Set SONOMED_PROJECT_ROOT}"
: "${SONOINSTRUCT_ROOT:?Set SONOINSTRUCT_ROOT}"
: "${SONOMED_OUTPUT_ROOT:?Set SONOMED_OUTPUT_ROOT}"
: "${SONOMED_CONFIG:?Set SONOMED_CONFIG}"
: "${SONOMED_ADAPTER:?Set SONOMED_ADAPTER}"
SONOMED_PYTHON="${SONOMED_PYTHON:-python}"
GPU_COUNT="${SLURM_GPUS_ON_NODE:-4}"
GPU_COUNT="${GPU_COUNT##*:}"
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
"$SONOMED_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
  scripts/baseline_eval.py --config "$SONOMED_CONFIG" --adapter "$SONOMED_ADAPTER"
crc-job-stats || true
