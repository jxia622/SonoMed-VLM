#!/bin/bash
#SBATCH --job-name=medical-transfer
#SBATCH --cluster=gpu
#SBATCH --partition=rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SONOMED_PROJECT_ROOT:?}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTHONPATH="$SONOMED_PROJECT_ROOT/runtime-deps${PYTHONPATH:+:$PYTHONPATH}"
"${SONOMED_PYTHON:?}" scripts/run_medical_transfer.py "$@"
