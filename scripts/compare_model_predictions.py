#!/usr/bin/env python3
"""Build an audited four-model report from saved raw generations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from sonomed_vlm.eval.comparison import compare_predictions, markdown_report
from sonomed_vlm.utils.io import atomic_write_text, read_jsonl, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sources = {}
    predictions = {}
    manifest_hash = sha256_file(args.manifest)
    for spec in args.predictions:
        name, path_string = spec.split("=", 1)
        path = Path(path_string)
        if name in predictions:
            raise ValueError(f"Duplicate model name: {name}")
        predictions[name] = list(read_jsonl(path))
        metadata = json.loads((path.parent / "metadata.json").read_text())
        if metadata["dataset_manifest_sha256"] != manifest_hash:
            raise ValueError(f"{name}: different manifest checksum")
        sources[name] = {
            "path": str(path.resolve()), "sha256": sha256_file(path),
            "model_revision": metadata["model_revision"],
            "packages": metadata["packages"],
            "config": str(path.parent / "config.yaml"),
        }
    report = compare_predictions(predictions, [row["example_id"] for row in read_jsonl(args.manifest)])
    scorer_root = Path(__file__).resolve().parents[1] / "src" / "sonomed_vlm" / "eval"
    report["provenance"] = {
        "manifest_sha256": manifest_hash, "sources": sources,
        "scorer_sha256": {path.name: sha256_file(path) for path in sorted(scorer_root.glob("*.py"))},
    }
    write_json(args.output_dir / "comparison.json", report)
    atomic_write_text(args.output_dir / "comparison.md", markdown_report(report))
    print(markdown_report(report))


if __name__ == "__main__":
    main()
