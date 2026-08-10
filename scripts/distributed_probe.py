#!/usr/bin/env python3
"""Validate CUDA/NCCL initialization and collect rank diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

import _bootstrap  # noqa: F401

from sonomed_vlm.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("The distributed probe requires CUDA")
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(f"Missing torchrun environment variables: {missing}")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    try:
        value = torch.tensor(float(rank), device=f"cuda:{local_rank}")
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
            "all_reduce_sum": value.item(),
        }
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
