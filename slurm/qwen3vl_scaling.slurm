#!/bin/bash
#SBATCH --job-name=qwen-scaling
#SBATCH --cluster=gpu
#SBATCH --partition=rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${SONOMED_PROJECT_ROOT:?Set SONOMED_PROJECT_ROOT}"
: "${SONOMED_OUTPUT_ROOT:?Set SONOMED_OUTPUT_ROOT}"
: "${SONOINSTRUCT_ROOT:?Set SONOINSTRUCT_ROOT}"
: "${SONOMED_PYTHON:?Set SONOMED_PYTHON}"
scale="${1:?Expected scale: 1, 5, 10, 25, or 50}"
case "$scale" in 1|5|10|25|50) ;; *) exit 2 ;; esac
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTHONPATH="$SONOMED_PROJECT_ROOT/runtime-deps${PYTHONPATH:+:$PYTHONPATH}"
cd "$SONOMED_PROJECT_ROOT"
for run in "qwen3vl_train_${scale}pct" "qwen3vl_eval_${scale}pct"; do
  if [[ -e "$SONOMED_OUTPUT_ROOT/$run" ]]; then
    echo "Refusing to overwrite $SONOMED_OUTPUT_ROOT/$run" >&2
    exit 1
  fi
done
nvidia-smi
# Every scale starts from the pinned base, never from another scale's adapter.
# Unlike the previous 1% smoke run, validation and generation use all 10,098 examples.
"$SONOMED_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/train.py --config "configs/qwen3vl_scaling_${scale}pct.yaml" --strict-numerics
"$SONOMED_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/baseline_eval.py --config "configs/qwen3vl_scaling_eval_${scale}pct.yaml" \
  --adapter "$SONOMED_OUTPUT_ROOT/qwen3vl_train_${scale}pct/final_adapter"
