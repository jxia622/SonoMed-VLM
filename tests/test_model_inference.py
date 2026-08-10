from __future__ import annotations

import os

import pytest


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("RUN_MEDGEMMA_GPU_TESTS") != "1",
    reason="Set RUN_MEDGEMMA_GPU_TESTS=1 only on an A100 with accepted gated access",
)
def test_base_inference_gate_is_explicit() -> None:
    import torch

    assert torch.cuda.is_available()
    assert torch.cuda.is_bf16_supported()
