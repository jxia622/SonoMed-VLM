#!/usr/bin/env python3
"""Legacy v1 exporter only; use summarize_open_qa.py for corrected v2 results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from sonomed_vlm.eval.comparison import MATCH_FIELDS, compare_predictions
from sonomed_vlm.utils.io import read_jsonl, sha256_file, sha256_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (args.output_dir / "research_results.json").exists():
        existing = json.loads((args.output_dir / "research_results.json").read_text())
        if existing.get("protocol") == "open_qa_v2":
            raise ValueError(
                "Refusing to overwrite corrected v2 results with historical v1 metrics"
            )
    manifest = list(read_jsonl(args.manifest))
    ids = [row["example_id"] for row in manifest]
    manifest_hash = sha256_file(args.manifest)
    reference = None
    models = []
    specs = json.loads(args.runs.read_text())
    if len({spec["name"] for spec in specs}) != len(specs):
        raise ValueError("Run names must be unique")
    for spec in specs:
        folder = args.outputs_root / spec["evaluation_dir"]
        path = folder / "eval_predictions.jsonl"
        metadata = json.loads((folder / "metadata.json").read_text())
        if metadata["dataset_manifest_sha256"] != manifest_hash:
            raise ValueError(f"{spec['name']}: manifest mismatch")
        rows = list(read_jsonl(path))
        signatures = {
            row["example_id"]: sha256_json({key: row.get(key) for key in MATCH_FIELDS})
            for row in rows
        }
        if reference is not None and signatures != reference:
            raise ValueError(f"{spec['name']}: prompts/references/options/settings differ")
        reference = signatures
        # Enforces unique complete IDs, greedy 256-token generation, finite metrics;
        # scores raw outputs, ignoring historical cached score fields.
        scored = compare_predictions({spec["name"]: rows}, ids)["models"][spec["name"]]
        record = {
            **spec,
            **scored,
            "provenance": {
                "predictions_sha256": sha256_file(path),
                "metadata_sha256": sha256_file(folder / "metadata.json"),
                "config_sha256": sha256_file(folder / "config.yaml"),
                "model_revision": metadata["model_revision"],
                "adapter_sha256": metadata.get("adapter_sha256"),
                "packages": metadata.get("packages", {}),
            },
        }
        if spec.get("training_dir"):
            training = args.outputs_root / spec["training_dir"]
            record["training_metrics"] = json.loads((training / "metrics.json").read_text())
            record["provenance"]["training_metrics_sha256"] = sha256_file(training / "metrics.json")
        models.append(record)
        print(spec["name"], scored["examples"], scored["overall"]["iou"], flush=True)
    scorer_root = Path(__file__).resolve().parents[1] / "src/sonomed_vlm/eval"
    report = {
        "protocol": "sonoinstruct-heldout-v1-choices-omitted-rescored-v2",
        "evaluation_examples": len(ids),
        "manifest_sha256": manifest_hash,
        "mcq_status": "Choices were omitted from training and inference; semantic matching is diagnostic, not standard MCQ accuracy.",
        "uncertainty": "One seed per scale; no confidence intervals or significance claims.",
        "scorer_sha256": {p.name: sha256_file(p) for p in sorted(scorer_root.glob("*.py"))},
        "models": models,
    }
    write_json(args.output_dir / "research_results.json", report)
    metric_keys = list(models[0]["metric_counts"])
    columns = ["name", "family", "scale_pct", "train_examples", *metric_keys]
    with (args.output_dir / "research_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for record in models:
            writer.writerow(
                {
                    **{k: record[k] for k in columns[:4]},
                    **{k: record["overall"].get(k) for k in metric_keys},
                }
            )


if __name__ == "__main__":
    main()
