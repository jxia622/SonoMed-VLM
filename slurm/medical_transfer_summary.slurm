#!/bin/bash
#SBATCH --job-name=med-transfer-summary
#SBATCH --cluster=gpu
#SBATCH --partition=rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SONOMED_PROJECT_ROOT:?}"
"${SONOMED_PYTHON:?}" scripts/summarize_medical_transfer.py "$@"
