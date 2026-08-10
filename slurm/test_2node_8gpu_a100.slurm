#!/bin/bash
#SBATCH --job-name=sonomed-nccl-8
#SBATCH --cluster=gpu
#SBATCH --partition=a100_multi
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${SONOMED_PROJECT_ROOT:?Set SONOMED_PROJECT_ROOT}"
SONOMED_PYTHON="${SONOMED_PYTHON:-python}"
NNODES="${SLURM_NNODES:?SLURM_NNODES is unavailable}"
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE##*:}"
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST:?}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
PROBE_OUTPUT="${SONOMED_OUTPUT_ROOT:-$SONOMED_PROJECT_ROOT/outputs}/distributed-probe-${SLURM_JOB_ID}"
export SONOMED_PYTHON NNODES GPUS_PER_NODE WORLD_SIZE MASTER_ADDR MASTER_PORT PROBE_OUTPUT
cd "$SONOMED_PROJECT_ROOT"
printf 'job_id=%s node_list=%s world_size=%s master=%s:%s output=%s\n' \
  "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST" "$WORLD_SIZE" "$MASTER_ADDR" "$MASTER_PORT" "$PROBE_OUTPUT"
srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
  bash -c '
    set -euo pipefail
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    "$SONOMED_PYTHON" -m torch.distributed.run \
      --nnodes="$NNODES" \
      --nproc_per_node="$GPUS_PER_NODE" \
      --node_rank="$SLURM_PROCID" \
      --master_addr="$MASTER_ADDR" \
      --master_port="$MASTER_PORT" \
      scripts/distributed_probe.py --output-dir "$PROBE_OUTPUT"
  '
crc-job-stats || true
