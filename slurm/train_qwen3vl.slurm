#!/bin/bash
#SBATCH --job-name=qwen3vl
#SBATCH --cluster=gpu
#SBATCH --partition=rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${SONOMED_PROJECT_ROOT:?Set SONOMED_PROJECT_ROOT}"
: "${SONOINSTRUCT_ROOT:?Set SONOINSTRUCT_ROOT}"
: "${SONOMED_OUTPUT_ROOT:?Set SONOMED_OUTPUT_ROOT}"
: "${SONOMED_PYTHON:?Set SONOMED_PYTHON}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
cd "$SONOMED_PROJECT_ROOT"
# CRC uses the existing torch 2.8/cu128 environment with a private torchvision overlay.
export PYTHONPATH="$SONOMED_PROJECT_ROOT/runtime-deps${PYTHONPATH:+:$PYTHONPATH}"
mode="${1:?Expected smoke or full}"
case "$mode" in
  smoke)
    config=configs/qwen3vl_train_1pct.yaml
    extra=(--max-eval-examples 128)
    run=qwen3vl_1pct_smoke
    ;;
  full)
    "$SONOMED_PYTHON" scripts/check_training_smoke.py "$SONOMED_OUTPUT_ROOT"
    config=configs/qwen3vl_train_100pct.yaml
    extra=()
    run=qwen3vl_100pct
    ;;
  *) echo "Expected smoke or full" >&2; exit 2 ;;
esac
# Both runs start independently; never overwrite or resume an earlier experiment.
if [[ -e "$SONOMED_OUTPUT_ROOT/$run" ]]; then
  echo "Run output already exists: $SONOMED_OUTPUT_ROOT/$run" >&2
  exit 1
fi
nvidia-smi
"$SONOMED_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/train.py --config "$config" --strict-numerics "${extra[@]}"
if [[ "$mode" == smoke ]]; then
  "$SONOMED_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
    scripts/baseline_eval.py --config configs/qwen3vl_eval_smoke.yaml \
    --adapter "$SONOMED_OUTPUT_ROOT/$run/final_adapter" --max-examples 8
  "$SONOMED_PYTHON" scripts/check_training_smoke.py "$SONOMED_OUTPUT_ROOT"
fi
