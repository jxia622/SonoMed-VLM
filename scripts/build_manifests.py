#!/usr/bin/env python3
"""Build deterministic, image-overlap-aware SonoInstruct manifests."""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

from sonomed_vlm.data.sonoinstruct import iter_raw_rows, normalize_row
from sonomed_vlm.data.splits import assert_no_image_leakage, nested_subsets, split_records
from sonomed_vlm.data.validation import ValidationReport, duplicate_qa_key, validate_example
from sonomed_vlm.utils.io import write_json, write_jsonl
from sonomed_vlm.utils.logging import configure_logging

LOGGER = logging.getLogger("build_manifests")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ood-source", action="append", default=[])
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--skip-image-validation", action="store_true")
    args = parser.parse_args()
    configure_logging()
    root = args.data_root.expanduser().resolve()
    records = []
    exclusions = []
    report = ValidationReport()
    duplicate_keys: set[tuple[str, str, str]] = set()
    duplicate_images: Counter[str] = Counter()

    for raw_index, raw in enumerate(iter_raw_rows(root)):
        if args.max_rows is not None and raw_index >= args.max_rows:
            break
        try:
            examples = normalize_row(raw, root)
        except Exception as exc:
            exclusions.append(
                {
                    "shard": str(raw.shard),
                    "row_index": raw.row_index,
                    "reason": "normalization_error",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        for example in examples:
            key = duplicate_qa_key(example)
            if key in duplicate_keys:
                report.examined += 1
                report.exclude("duplicate_exact_qa", example.example_id)
                exclusions.append(
                    {"example_id": example.example_id, "reason": "duplicate_exact_qa"}
                )
                continue
            duplicate_keys.add(key)
            for image_id in example.image_ids:
                duplicate_images[image_id] += 1
            valid = True
            if args.skip_image_validation:
                report.examined += 1
                if not example.user_prompt.strip():
                    report.exclude("empty_question", example.example_id)
                    valid = False
                if not example.assistant_response.strip():
                    report.exclude("empty_response", example.example_id)
                    valid = False
                if valid:
                    report.valid += 1
            else:
                valid = validate_example(example, report)
            if not valid:
                exclusions.append(
                    {"example_id": example.example_id, "reason": "integrity_validation"}
                )
                continue
            record = example.to_manifest_record()
            shard = Path(record["shard"])
            try:
                record["shard"] = str(shard.relative_to(root))
            except ValueError:
                raise ValueError(f"Shard escaped data root: {shard}") from None
            records.append(record)
        if (raw_index + 1) % 1000 == 0:
            LOGGER.info("processed_rows=%d valid_examples=%d", raw_index + 1, len(records))

    ood_sources = set(args.ood_source)
    if not records:
        raise RuntimeError(
            "No valid examples were produced. Inspect exclusions and verify the observed schema/data root."
        )
    ood = [record for record in records if record.get("source_dataset") in ood_sources]
    internal = [record for record in records if record.get("source_dataset") not in ood_sources]
    train, val = split_records(internal, val_fraction=args.val_fraction, seed=args.seed)
    assert_no_image_leakage(train, val)
    if ood:
        assert_no_image_leakage(train + val, ood)
    subsets = nested_subsets(train, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "val.jsonl", val)
    smoke_validation = val[: min(16, len(val))]
    write_jsonl(args.output_dir / "val_smoke.jsonl", smoke_validation)
    if ood:
        write_jsonl(args.output_dir / "ood_val.jsonl", ood)
    for fraction, subset in subsets.items():
        name = "train.jsonl" if fraction == 100 else f"train_{fraction}pct.jsonl"
        if fraction != 100:
            write_jsonl(args.output_dir / name, subset)
    write_jsonl(args.output_dir / "exclusions.jsonl", exclusions)
    summary = {
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "train_examples": len(train),
        "val_examples": len(val),
        "smoke_val_examples": len(smoke_validation),
        "ood_examples": len(ood),
        "subset_examples": {f"{key}pct": len(value) for key, value in subsets.items()},
        "validation": report.to_dict(),
        "exact_image_hashes_reused": sum(count > 1 for count in duplicate_images.values()),
        "ood_sources": sorted(ood_sources),
    }
    write_json(args.output_dir / "manifest_summary.json", summary)
    LOGGER.info("manifest_summary=%s", summary)


if __name__ == "__main__":
    main()
