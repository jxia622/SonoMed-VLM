#!/bin/bash
#SBATCH --job-name=qwen-comparison
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
: "${SONOMED_OUTPUT_ROOT:?Set SONOMED_OUTPUT_ROOT}"
: "${SONOMED_PYTHON:?Set SONOMED_PYTHON}"
: "${QWEN_ADAPTER:?Set QWEN_ADAPTER}"
: "${MEDGEMMA_EVAL_ROOT:?Set MEDGEMMA_EVAL_ROOT}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTHONPATH="$SONOMED_PROJECT_ROOT/runtime-deps${PYTHONPATH:+:$PYTHONPATH}"
cd "$SONOMED_PROJECT_ROOT"
for run in qwen_original qwen_finetuned comparison; do
  if [[ -e "$SONOMED_OUTPUT_ROOT/$run" ]]; then
    echo "Refusing to overwrite $SONOMED_OUTPUT_ROOT/$run" >&2
    exit 1
  fi
done
nvidia-smi
"$SONOMED_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/baseline_eval.py --config configs/qwen3vl_eval_base.yaml
"$SONOMED_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/baseline_eval.py --config configs/qwen3vl_eval_finetuned.yaml --adapter "$QWEN_ADAPTER"
"$SONOMED_PYTHON" scripts/compare_model_predictions.py \
  --manifest data/manifests/val.jsonl \
  --predictions "qwen_original=$SONOMED_OUTPUT_ROOT/qwen_original/eval_predictions.jsonl" \
  --predictions "qwen_finetuned=$SONOMED_OUTPUT_ROOT/qwen_finetuned/eval_predictions.jsonl" \
  --predictions "medgemma_original=$MEDGEMMA_EVAL_ROOT/eval_base_medgemma_rtx6k/eval_predictions.jsonl" \
  --predictions "medgemma_finetuned=$MEDGEMMA_EVAL_ROOT/eval_100pct_rtx6k/eval_predictions.jsonl" \
  --output-dir "$SONOMED_OUTPUT_ROOT/comparison"
