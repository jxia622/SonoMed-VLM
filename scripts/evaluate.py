#!/usr/bin/env python3
"""Re-score stored raw predictions without rerunning the model."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from sonomed_vlm.eval.aggregate import aggregate_scores
from sonomed_vlm.eval.parsing import score_prediction
from sonomed_vlm.utils.io import read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    rows = []
    for row in read_jsonl(args.predictions):
        row["score"] = score_prediction(row)
        row["parsed_prediction"] = row["score"].get("parsed_prediction")
        rows.append(row)
    output_dir = args.output_dir or args.predictions.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "eval_predictions.jsonl", rows)
    metrics = aggregate_scores(rows)
    write_json(output_dir / "metrics.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
