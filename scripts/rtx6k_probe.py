#!/usr/bin/env python3
"""Validate Blackwell CUDA/BF16, NCCL, and MedGemma compatibility."""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from sonomed_vlm.utils.io import write_json


def _move_inputs(inputs: Any, device: Any, dtype: Any) -> dict[str, Any]:
    moved = {}
    for key, value in dict(inputs).items():
        if hasattr(value, "to"):
            value = value.to(device)
            if value.is_floating_point():
                value = value.to(dtype=dtype)
        moved[key] = value
    return moved


def _check_medgemma(config_path: Path, device: Any) -> dict[str, Any]:
    import torch

    from sonomed_vlm.config import load_config
    from sonomed_vlm.models.medgemma import load_medgemma

    config = load_config(config_path)
    model, processor = load_medgemma(config, for_training=True)
    model = model.to(device).eval()
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Reply with OK."}],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = _move_inputs(inputs, device, torch.bfloat16)
    with torch.inference_mode():
        outputs = model(**inputs)
    last_logits = outputs.logits[:, -1, :]
    if not torch.isfinite(last_logits).all():
        raise RuntimeError("MedGemma produced non-finite logits on the target GPU")
    result = {
        "model_class": type(model).__name__,
        "processor_class": type(processor).__name__,
        "model_dtype": str(next(parameter.dtype for parameter in model.parameters())),
        "model_forward_finite": True,
        "model_logits_shape": list(outputs.logits.shape),
    }
    del outputs, inputs, model, processor
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--medgemma-config", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("The RTX6K probe requires CUDA")
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(f"Missing torchrun environment variables: {missing}")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    try:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The target GPU does not support BF16")
        left = torch.randn((256, 256), device=device, dtype=torch.bfloat16)
        right = torch.randn((256, 256), device=device, dtype=torch.bfloat16)
        product = left @ right
        if not torch.isfinite(product).all():
            raise RuntimeError("BF16 matrix multiplication produced non-finite values")

        value = torch.tensor(float(rank), device=device)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        expected = world_size * (world_size - 1) / 2
        if value.item() != expected:
            raise RuntimeError(f"all_reduce returned {value.item()}, expected {expected}")
        diagnostic = {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "hostname": socket.gethostname(),
            "cuda_device_index": local_rank,
            "cuda_device_name": torch.cuda.get_device_name(local_rank),
            "cuda_device_count": torch.cuda.device_count(),
            "bf16_supported": True,
            "bf16_matmul_finite": True,
            "all_reduce_sum": value.item(),
            "medgemma": _check_medgemma(args.medgemma_config, device),
        }
        dist.barrier()
        gathered: list[dict[str, object] | None] = [None] * world_size
        dist.all_gather_object(gathered, diagnostic)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / f"rank-{rank:02d}.json", diagnostic)
        dist.barrier()
        if rank == 0:
            hosts: dict[str, list[int]] = {}
            for item in gathered:
                if item is None:
                    continue
                hosts.setdefault(str(item["hostname"]), []).append(int(item["rank"]))
            summary = {
                "passed": len(gathered) == world_size
                and all(item is not None for item in gathered),
                "backend": dist.get_backend(),
                "world_size": world_size,
                "hosts": {host: sorted(ranks) for host, ranks in sorted(hosts.items())},
                "ranks": gathered,
            }
            write_json(args.output_dir / "distributed_summary.json", summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
