#!/bin/bash
#SBATCH --job-name=openqa-v2
#SBATCH --cluster=gpu
#SBATCH --partition=rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
: "${SONOMED_PROJECT_ROOT:?Set SONOMED_PROJECT_ROOT}"
: "${SONOMED_OUTPUT_ROOT:?Set SONOMED_OUTPUT_ROOT}"
: "${SONOMED_PYTHON:?Set SONOMED_PYTHON}"
family="${1:?qwen or medgemma}"
scale="${2:?1, 5, 10, 25, 50, 100, or baselines}"
case "$family" in qwen|medgemma) ;; *) exit 2 ;; esac
cd "$SONOMED_PROJECT_ROOT"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTHONPATH="$SONOMED_PROJECT_ROOT/runtime-deps${PYTHONPATH:+:$PYTHONPATH}"
manifest=data/manifests-openqa-v2/val.jsonl
if [[ "$scale" == baselines ]]; then
  for kind in original legacy100; do
    name="openqa_v2_${family}_${kind}"
    [[ ! -e "$SONOMED_OUTPUT_ROOT/$name" ]] || exit 1
    adapter=()
    if [[ "$kind" == legacy100 ]]; then
      if [[ "$family" == qwen ]]; then
        adapter=(--adapter "${OPENQA_LEGACY_QWEN_ADAPTER:?Set legacy Qwen adapter path}")
      else
        adapter=(--adapter "${OPENQA_LEGACY_MEDGEMMA_ADAPTER:?Set legacy MedGemma adapter path}")
      fi
    fi
    "$SONOMED_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
      scripts/baseline_eval.py --config "configs/${name}.yaml" "${adapter[@]}"
    "$SONOMED_PYTHON" scripts/check_open_qa_run.py --evaluation "$SONOMED_OUTPUT_ROOT/$name" --manifest "$manifest"
  done
  exit 0
fi
case "$scale" in 1|5|10|25|50|100) ;; *) exit 2 ;; esac
name="openqa_v2_${family}_${scale}pct"
for out in "$name" "${name}_eval"; do
  [[ ! -e "$SONOMED_OUTPUT_ROOT/$out" ]] || exit 1
done
if [[ "$scale" != 1 ]]; then
  "$SONOMED_PYTHON" scripts/check_open_qa_run.py \
    --evaluation "$SONOMED_OUTPUT_ROOT/openqa_v2_${family}_1pct_eval" \
    --training "$SONOMED_OUTPUT_ROOT/openqa_v2_${family}_1pct" --manifest "$manifest"
fi
"$SONOMED_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/train.py --config "configs/${name}.yaml" --strict-numerics
"$SONOMED_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/baseline_eval.py --config "configs/${name}_eval.yaml" \
  --adapter "$SONOMED_OUTPUT_ROOT/$name/final_adapter"
"$SONOMED_PYTHON" scripts/check_open_qa_run.py --training "$SONOMED_OUTPUT_ROOT/$name" \
  --evaluation "$SONOMED_OUTPUT_ROOT/${name}_eval" --manifest "$manifest"
