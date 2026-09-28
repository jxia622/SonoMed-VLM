#!/bin/bash
#SBATCH --job-name=openqa-summary
#SBATCH --cluster=gpu
#SBATCH --partition=rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SONOMED_PROJECT_ROOT:?}"
"${SONOMED_PYTHON:?}" scripts/summarize_open_qa.py \
  --outputs-root "${SONOMED_OUTPUT_ROOT:?}" --manifest data/manifests-openqa-v2/val.jsonl
