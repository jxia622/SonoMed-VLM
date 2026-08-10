#!/usr/bin/env python3
"""Validate manifests against source data and prove split isolation."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from sonomed_vlm.data.sonoinstruct import ManifestDataset
from sonomed_vlm.data.splits import assert_no_image_leakage
from sonomed_vlm.data.validation import ValidationReport, validate_example
from sonomed_vlm.utils.io import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/data_validation.json"))
    args = parser.parse_args()

    train_records = list(read_jsonl(args.manifest_dir / "train.jsonl"))
    val_records = list(read_jsonl(args.manifest_dir / "val.jsonl"))
    assert_no_image_leakage(train_records, val_records)
    report = ValidationReport()
    for manifest in (args.manifest_dir / "train.jsonl", args.manifest_dir / "val.jsonl"):
        dataset = ManifestDataset(manifest, args.data_root)
        limit = len(dataset) if args.max_examples is None else min(len(dataset), args.max_examples)
        for index in range(limit):
            validate_example(dataset[index], report)
    output = {"split_leakage": False, "validation": report.to_dict()}
    write_json(args.output, output)
    print(output)
    if report.exclusions:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
