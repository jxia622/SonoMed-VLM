#!/bin/bash
#SBATCH --job-name=sonomed-cuda
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${SONOMED_PROJECT_ROOT:?Set SONOMED_PROJECT_ROOT}"
SONOMED_PYTHON="${SONOMED_PYTHON:-python}"
cd "$SONOMED_PROJECT_ROOT"
nvidia-smi
"$SONOMED_PYTHON" - <<'PY'
import json

import torch

if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable to Torch')
device = torch.device('cuda', 0)
left = torch.ones((256, 256), device=device)
right = torch.ones((256, 256), device=device)
result = left @ right
torch.cuda.synchronize()
print(json.dumps({
    'torch': torch.__version__,
    'torch_cuda': torch.version.cuda,
    'device': torch.cuda.get_device_name(device),
    'result': float(result[0, 0]),
}, sort_keys=True))
PY
crc-job-stats || true
