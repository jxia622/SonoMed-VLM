#!/bin/bash
#SBATCH --job-name=sonomed-env
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${SONOMED_PROJECT_ROOT:?Set SONOMED_PROJECT_ROOT}"

module purge
module load python/3.11.9

cd "$SONOMED_PROJECT_ROOT"
mkdir -p logs
if [[ ! -x .venv/bin/python ]]; then
  python -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --upgrade 'torch==2.8.0' \
  --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python scripts/summarize_run.py --environment-only
crc-job-stats || true
