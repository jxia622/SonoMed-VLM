#!/bin/bash
# Run from the staged CRC project after exporting the SONOMED paths.
set -euo pipefail
: "${SONOMED_PROJECT_ROOT:?Set SONOMED_PROJECT_ROOT}"
cd "$SONOMED_PROJECT_ROOT"
if [[ -e submission.txt ]]; then
  echo "This deployment has already been submitted; see submission.txt" >&2
  exit 1
fi
mkdir -p logs
smoke=$(sbatch --parsable --job-name=qwen3vl-1pct --time=04:00:00 slurm/train_qwen3vl.slurm smoke)
smoke_id=${smoke%%;*}
full=$(sbatch --parsable --job-name=qwen3vl-full --dependency="afterok:$smoke_id" \
  --kill-on-invalid-dep=yes slurm/train_qwen3vl.slurm full)
printf 'smoke=%s\nfull=%s\n' "$smoke" "$full" | tee submission.txt
