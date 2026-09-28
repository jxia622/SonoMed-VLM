#!/usr/bin/env python3
"""Aggregate corrected runs only after matching protocol, inputs and example IDs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
from check_open_qa_run import check

from sonomed_vlm.eval.aggregate import aggregate_scores
from sonomed_vlm.eval.comparison import MATCH_FIELDS
from sonomed_vlm.eval.parsing import score_prediction
from sonomed_vlm.utils.io import read_jsonl, sha256_file, sha256_json, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--runs", type=Path, default=Path("configs/openqa_v2_runs.json"))
    args = p.parse_args()
    result = {
        "protocol": "open_qa_v2",
        "manifest_sha256": sha256_file(args.manifest),
        "models": [],
        "limitations": "One seed; internal validation; open-QA exact match and lexical overlap do not measure clinical correctness.",
    }
    reference = None
    keys = [
        "answer_exact_match",
        "answer_token_f1",
        "answer_rouge_l",
        "answer_empty_or_label_only_rate",
        "token_f1",
        "rouge_l_f1",
        "iou",
        "localization_at_0_5",
    ]
    for spec in json.loads(args.runs.read_text()):
        folder = args.outputs_root / spec["evaluation_dir"]
        check(
            folder,
            args.manifest,
            args.outputs_root / spec["training_dir"] if spec["training_dir"] else None,
        )
        rows = list(read_jsonl(folder / "eval_predictions.jsonl"))
        signatures = {
            row["example_id"]: sha256_json(
                {
                    k: row.get(k)
                    for k in (
                        *MATCH_FIELDS,
                        "system_prompt",
                        "instruction_protocol",
                        "original_task_type",
                    )
                }
            )
            for row in rows
        }
        if reference is not None and signatures != reference:
            raise ValueError("Unmatched evaluation inputs: " + spec["name"])
        reference = signatures
        for row in rows:
            row["score"] = score_prediction(row)
        metrics = aggregate_scores(rows)
        result["models"].append(
            {
                **spec,
                **metrics,
                "metric_counts": dict(
                    Counter(k for row in rows for k in keys if k in row["score"])
                ),
                "predictions_sha256": sha256_file(folder / "eval_predictions.jsonl"),
            }
        )
    output = args.outputs_root / "summary"
    output.mkdir(exist_ok=True)
    write_json(output / "results.json", result)
    with (output / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["name", "family", "scale_pct", "kind", *keys], lineterminator="\n"
        )
        writer.writeheader()
        for r in result["models"]:
            writer.writerow(
                {
                    **{k: r[k] for k in ["name", "family", "scale_pct", "kind"]},
                    **{k: r["overall"].get(k) for k in keys},
                }
            )
    lines = [
        "# Corrected open-ended QA results",
        "",
        "One seed, same filtered validation split. These are not MCQ accuracy or SonoBench scores.",
        "",
        "| Model/run | Open-QA exact match | Open-QA token F1 | Original QA/open F1 | Mean IoU | Loc@0.5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in result["models"]:
        m = r["overall"]
        lines.append(
            f"| {r['name']} | {m['answer_exact_match']:.2%} | {m['answer_token_f1']:.4f} | {m['token_f1']:.4f} | {m['iou']:.4f} | {m['localization_at_0_5']:.2%} |"
        )
    (output / "results.md").write_text("\n".join(lines) + "\n")
    print("Verified and summarized", len(result["models"]), "matched runs.")


if __name__ == "__main__":
    main()
