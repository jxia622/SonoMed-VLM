#!/bin/bash
#SBATCH --job-name=sonomed-data
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${SONOMED_PROJECT_ROOT:?Set SONOMED_PROJECT_ROOT}"
: "${SONOINSTRUCT_ROOT:?Set SONOINSTRUCT_ROOT}"
SONOMED_PYTHON="${SONOMED_PYTHON:-$SONOMED_PROJECT_ROOT/.venv/bin/python}"
cd "$SONOMED_PROJECT_ROOT"
mkdir -p artifacts data/manifests logs
"$SONOMED_PYTHON" scripts/inspect_dataset.py \
  --data-root "$SONOINSTRUCT_ROOT" --sample-rows 16 --output-dir artifacts
"$SONOMED_PYTHON" scripts/build_manifests.py \
  --data-root "$SONOINSTRUCT_ROOT" --output-dir data/manifests
"$SONOMED_PYTHON" scripts/check_manifests.py \
  --manifest-dir data/manifests --output artifacts/manifest_validation.json
"$SONOMED_PYTHON" -m pytest -q
"$SONOMED_PROJECT_ROOT/.venv/bin/ruff" check .
crc-job-stats || true
