#!/usr/bin/env python3
"""Verify manifest leakage isolation and nested scaling subsets without re-decoding data."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from sonomed_vlm.data.splits import assert_no_image_leakage
from sonomed_vlm.utils.io import read_jsonl, sha256_file, write_json


def image_ids(records: list[dict]) -> set[str]:
    return {
        image_id for record in records for image_id in record.get("image_ids", [record["image_id"]])
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/manifest_validation.json"))
    args = parser.parse_args()

    train = list(read_jsonl(args.manifest_dir / "train.jsonl"))
    val = list(read_jsonl(args.manifest_dir / "val.jsonl"))
    smoke_val = list(read_jsonl(args.manifest_dir / "val_smoke.jsonl"))
    if not train or not val:
        raise ValueError("Both train and validation manifests must be non-empty")
    if not smoke_val:
        raise ValueError("Smoke validation manifest must be non-empty")
    assert_no_image_leakage(train, val)
    val_ids = image_ids(val)
    if not image_ids(smoke_val) <= val_ids:
        raise ValueError("val_smoke.jsonl contains images outside val.jsonl")
    fractions = (1, 5, 10, 25, 50)
    previous: set[str] = set()
    subsets = {}
    for fraction in fractions:
        path = args.manifest_dir / f"train_{fraction}pct.jsonl"
        records = list(read_jsonl(path))
        current = image_ids(records)
        if not previous <= current:
            raise ValueError(f"train_{fraction}pct is not nested over the preceding subset")
        if current & val_ids:
            raise ValueError(f"train_{fraction}pct overlaps validation images")
        if not current <= image_ids(train):
            raise ValueError(f"train_{fraction}pct contains images outside train.jsonl")
        previous = current
        subsets[f"{fraction}pct"] = {"examples": len(records), "images": len(current)}

    files = [
        args.manifest_dir / "train.jsonl",
        args.manifest_dir / "val.jsonl",
        args.manifest_dir / "val_smoke.jsonl",
    ] + [args.manifest_dir / f"train_{fraction}pct.jsonl" for fraction in fractions]
    output = {
        "passed": True,
        "train_examples": len(train),
        "validation_examples": len(val),
        "smoke_validation_examples": len(smoke_val),
        "train_images": len(image_ids(train)),
        "validation_images": len(val_ids),
        "train_validation_image_overlap": 0,
        "subsets": subsets,
        "manifest_sha256": {path.name: sha256_file(path) for path in files},
    }
    write_json(args.output, output)
    print(output)


if __name__ == "__main__":
    main()
