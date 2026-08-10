#!/bin/bash
#SBATCH --job-name=sonomed-download
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${SONOMED_PROJECT_ROOT:?Set SONOMED_PROJECT_ROOT}"
: "${SONOINSTRUCT_ROOT:?Set SONOINSTRUCT_ROOT}"
: "${HF_HOME:?Set HF_HOME}"

cd "$SONOMED_PROJECT_ROOT"
mkdir -p "$SONOINSTRUCT_ROOT" "$HF_HOME" logs
.venv/bin/hf download Ssdaizi/SonoInstruct \
  --repo-type dataset \
  --include 'parquet_full_train/*' \
  --include README.md \
  --include train_sample.parquet \
  --local-dir "$SONOINSTRUCT_ROOT"
find "$SONOINSTRUCT_ROOT/parquet_full_train" -maxdepth 1 -name '*.parquet' -type f | wc -l
du -sh "$SONOINSTRUCT_ROOT" "$HF_HOME"
crc-job-stats || true
