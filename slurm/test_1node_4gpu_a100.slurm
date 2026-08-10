#!/bin/bash
#SBATCH --job-name=sonomed-nccl-4
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${SONOMED_PROJECT_ROOT:?Set SONOMED_PROJECT_ROOT}"
SONOMED_PYTHON="${SONOMED_PYTHON:-python}"
GPU_COUNT="${SLURM_GPUS_ON_NODE:-4}"
GPU_COUNT="${GPU_COUNT##*:}"
PROBE_OUTPUT="${SONOMED_OUTPUT_ROOT:-$SONOMED_PROJECT_ROOT/outputs}/distributed-probe-${SLURM_JOB_ID}"
cd "$SONOMED_PROJECT_ROOT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"$SONOMED_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
  scripts/distributed_probe.py --output-dir "$PROBE_OUTPUT"
crc-job-stats || true
