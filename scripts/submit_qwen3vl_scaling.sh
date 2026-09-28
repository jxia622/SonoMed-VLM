#!/bin/bash
set -euo pipefail
: "${SONOMED_PROJECT_ROOT:?Set SONOMED_PROJECT_ROOT}"
cd "$SONOMED_PROJECT_ROOT"
mkdir -p logs
receipt=submission-scaling-20260927.tsv
if [[ -e "$receipt" ]]; then
  echo "Submission receipt exists; inspect it before retrying: $receipt" >&2
  exit 1
fi
printf 'scale_pct\tjob_id_cluster\ttime_limit\n' > "$receipt"
for scale in 1 5 10 25 50; do
  case "$scale" in
    1|5) limit=02:00:00 ;;
    10) limit=03:00:00 ;;
    25) limit=05:00:00 ;;
    50) limit=08:00:00 ;;
  esac
  job=$(sbatch --parsable --qos=gpu-rtx6k-s --job-name="qwen-scale-${scale}pct" \
    --time="$limit" slurm/qwen3vl_scaling.slurm "$scale")
  printf '%s\t%s\t%s\n' "$scale" "$job" "$limit" | tee -a "$receipt"
done
