#!/usr/bin/env python3
"""Print the numerically latest complete Trainer checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from sonomed_vlm.training.checkpointing import find_latest_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_root", type=Path)
    args = parser.parse_args()
    latest = find_latest_checkpoint(args.checkpoint_root)
    if latest is None:
        raise SystemExit(f"No valid checkpoint found under {args.checkpoint_root}")
    print(latest.resolve())


if __name__ == "__main__":
    main()
