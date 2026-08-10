#!/usr/bin/env python3
"""Summarize run provenance/metrics or print the current package environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from sonomed_vlm.training.metadata import gpu_metadata, package_versions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, nargs="?")
    parser.add_argument("--environment-only", action="store_true")
    args = parser.parse_args()
    if args.environment_only:
        print(json.dumps({"packages": package_versions(), "gpu": gpu_metadata()}, indent=2))
        return
    if args.run_dir is None:
        parser.error("run_dir is required unless --environment-only is used")
    output = {}
    for name in ("metadata.json", "metrics.json", "architecture.json"):
        path = args.run_dir / name
        output[name.removesuffix(".json")] = json.loads(path.read_text()) if path.exists() else None
    print(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
