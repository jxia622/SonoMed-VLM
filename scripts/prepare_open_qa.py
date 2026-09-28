#!/usr/bin/env python3
"""Audit annotations and derive nested, shared open-QA manifests without reading images."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
import pyarrow.parquet as pq

from sonomed_vlm.data.open_qa import PROTOCOL, ConversionError, canonical_answer, is_choice_example
from sonomed_vlm.utils.io import read_jsonl, sha256_file, sha256_json, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifests", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    names = [f"train_{n}pct.jsonl" for n in [1, 5, 10, 25, 50]] + ["train.jsonl", "val.jsonl"]
    manifests = {name: list(read_jsonl(args.source_manifests / name)) for name in names}
    all_rows = manifests["train.jsonl"] + manifests["val.jsonl"]
    if len({r["example_id"] for r in all_rows}) != len(all_rows):
        raise ValueError("Source train/validation IDs overlap or repeat")
    if {r["leakage_group_id"] for r in manifests["train.jsonl"]} & {
        r["leakage_group_id"] for r in manifests["val.jsonl"]
    }:
        raise ValueError("Source train/validation image groups overlap")
    by_shard = defaultdict(list)
    for row in all_rows:
        by_shard[Path(row["shard"]).name].append(row)
    exclusions = {}
    converted = {}
    examples = []
    for name, records in sorted(by_shard.items()):
        source = args.data_root / "parquet_full_train" / name
        rows = pq.read_table(source, columns=["QA"]).to_pylist()
        for row in records:
            items = rows[row["row_index"]]["QA"]
            items = json.loads(items) if isinstance(items, str) else items
            qa = items[row["qa_index"]]
            if not is_choice_example(row.get("task_type"), qa):
                continue
            try:
                gold = canonical_answer(
                    str(qa.get("answer") or ""),
                    qa.get("answer_label"),
                    qa.get("options"),
                    re.sub(
                        r"\s*<image>\s*", "\n", str(qa.get("question") or ""), flags=re.I
                    ).strip(),
                )
                converted[row["example_id"]] = sha256_json(
                    {"question": qa["question"], "answer": gold}
                )
            except ConversionError as exc:
                exclusions[row["example_id"]] = str(exc)
                if len(examples) < 30:
                    examples.append(
                        {
                            "example_id": row["example_id"],
                            "reason": str(exc),
                            "question": qa.get("question"),
                            "answer": qa.get("answer"),
                        }
                    )
        print(name, "audited", len(records), flush=True)
    report = {
        "protocol": PROTOCOL,
        "source_manifests": {},
        "output_manifests": {},
        "excluded_reasons": dict(Counter(exclusions.values())),
        "converted_reference_sha256": sha256_json(converted),
        "conversion_code_sha256": sha256_file(
            Path(__file__).resolve().parents[1] / "src/sonomed_vlm/data/open_qa.py"
        ),
    }
    previous = set()
    for name, rows in manifests.items():
        kept = [row for row in rows if row["example_id"] not in exclusions]
        ids = {row["example_id"] for row in kept}
        if name != "val.jsonl":
            if not previous <= ids:
                raise ValueError("Filtered training subsets are not nested")
            previous = ids
        count = write_jsonl(args.output_dir / name, kept)
        report["source_manifests"][name] = {
            "examples": len(rows),
            "sha256": sha256_file(args.source_manifests / name),
        }
        report["output_manifests"][name] = {
            "examples": count,
            "sha256": sha256_file(args.output_dir / name),
            "converted_open_qa": sum(row["example_id"] in converted for row in kept),
            "excluded": len(rows) - count,
            "exclusion_reasons": dict(
                Counter(
                    exclusions[row["example_id"]] for row in rows if row["example_id"] in exclusions
                )
            ),
        }
    write_json(args.output_dir / "audit.json", report)
    write_json(args.output_dir / "excluded_review_examples.private.json", examples)
    write_jsonl(
        args.output_dir / "excluded_ids.jsonl",
        ({"example_id": k, "reason": v} for k, v in sorted(exclusions.items())),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
