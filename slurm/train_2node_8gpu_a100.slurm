#!/bin/bash
#SBATCH --job-name=sonomed-8gpu
#SBATCH --cluster=gpu
#SBATCH --partition=a100_multi
#SBATCH --nodes=2
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
SONOMED_PYTHON="${SONOMED_PYTHON:-python}"
SONOMED_CONFIG="${SONOMED_CONFIG:-configs/train_1pct.yaml}"
NNODES="${SLURM_NNODES:?SLURM_NNODES is unavailable}"
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE##*:}"
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST:?}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
export SONOMED_PYTHON SONOMED_CONFIG NNODES GPUS_PER_NODE WORLD_SIZE MASTER_ADDR MASTER_PORT

cd "$SONOMED_PROJECT_ROOT"
printf 'job_id=%s nodes=%s node_list=%s gpus_per_node=%s world_size=%s master=%s:%s config=%s\n' \
  "$SLURM_JOB_ID" "$NNODES" "$SLURM_JOB_NODELIST" "$GPUS_PER_NODE" "$WORLD_SIZE" \
  "$MASTER_ADDR" "$MASTER_PORT" "$SONOMED_CONFIG"
srun --overlap --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 \
  bash scripts/gpu_monitor.sh "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-gpu" &
monitor_pid=$!
cleanup_monitor() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup_monitor EXIT
srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
  bash -c '
    set -euo pipefail
    printf "launcher_host=%s node_rank=%s visible_devices=%s\n" \
      "$(hostname)" "$SLURM_PROCID" "${CUDA_VISIBLE_DEVICES:-unset}"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    "$SONOMED_PYTHON" -m torch.distributed.run \
      --nnodes="$NNODES" \
      --nproc_per_node="$GPUS_PER_NODE" \
      --node_rank="$SLURM_PROCID" \
      --master_addr="$MASTER_ADDR" \
      --master_port="$MASTER_PORT" \
      scripts/train.py --config "$SONOMED_CONFIG" --resume-from-checkpoint auto-if-present
  '
crc-job-stats || true
