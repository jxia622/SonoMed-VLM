#!/usr/bin/env python3
"""Inspect every Parquet shard and sample SonoInstruct records/images safely."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import pyarrow.parquet as pq

from sonomed_vlm.data.sonoinstruct import discover_parquet_files, iter_raw_rows, normalize_row
from sonomed_vlm.utils.io import write_json


def _safe_value(value: Any, limit: int = 240) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<bytes:{len(value)}>"
    if isinstance(value, str):
        compact = value.replace("\n", "\\n")
        return compact if len(compact) <= limit else compact[:limit] + "…"
    if isinstance(value, list):
        return [_safe_value(item, limit) for item in value[:3]] + (["…"] if len(value) > 3 else [])
    if isinstance(value, dict):
        return {key: _safe_value(item, limit) for key, item in value.items()}
    return value


def inspect(data_root: Path, sample_rows: int) -> dict[str, Any]:
    shards = discover_parquet_files(data_root)
    shard_reports: list[dict[str, Any]] = []
    total_rows = 0
    schema_strings: set[str] = set()
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        arrow_schema = parquet.schema_arrow
        schema_text = str(arrow_schema)
        schema_strings.add(schema_text)
        total_rows += parquet.metadata.num_rows
        shard_reports.append(
            {
                "path": str(shard),
                "rows": parquet.metadata.num_rows,
                "row_groups": parquet.num_row_groups,
                "size_bytes": shard.stat().st_size,
                "created_by": parquet.metadata.created_by,
                "columns": [
                    {"name": field.name, "arrow_type": str(field.type), "nullable": field.nullable}
                    for field in arrow_schema
                ],
            }
        )

    samples: list[dict[str, Any]] = []
    task_values: Counter[str] = Counter()
    source_values: Counter[str] = Counter()
    image_counts: Counter[int] = Counter()
    qa_counts: Counter[int] = Counter()
    dimensions: list[dict[str, Any]] = []
    image_hashes: Counter[str] = Counter()
    corrupt = 0
    sampled_rows = 0
    if sample_rows:
        for raw in iter_raw_rows(data_root, batch_size=min(64, sample_rows)):
            try:
                examples = normalize_row(raw, data_root)
                raw_qa = raw.value.get("QA")
                qa_value = json.loads(raw_qa) if isinstance(raw_qa, str) else raw_qa
                qa_counts[len(qa_value or [])] += 1
                image_counts[len(examples[0].images) if examples else 0] += 1
                task_values[str(raw.value.get("type"))] += 1
                source_values[str(raw.value.get("dataset"))] += 1
                for image in examples[0].images if examples else []:
                    image_id = image.compute_id()
                    image_hashes[image_id] += 1
                    try:
                        decoded = image.open()
                        dimensions.append(
                            {
                                "image_id": image_id,
                                "width": decoded.width,
                                "height": decoded.height,
                                "mode": decoded.mode,
                                "format": decoded.format,
                            }
                        )
                    except ValueError:
                        corrupt += 1
                samples.append(
                    {
                        "shard": str(raw.shard),
                        "row_index": raw.row_index,
                        "raw": _safe_value(raw.value),
                        "normalized_examples": [
                            {
                                "example_id": example.example_id,
                                "image_id": example.image_id,
                                "source_dataset": example.source_dataset,
                                "task_family": example.task_family,
                                "task_type": example.task_type,
                                "focus": example.focus,
                                "user_prompt": _safe_value(example.user_prompt),
                                "assistant_response": _safe_value(example.assistant_response),
                            }
                            for example in examples[:3]
                        ],
                    }
                )
            except Exception as exc:
                samples.append(
                    {
                        "shard": str(raw.shard),
                        "row_index": raw.row_index,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            sampled_rows += 1
            if sampled_rows >= sample_rows:
                break

    return {
        "data_root": str(data_root.resolve()),
        "shard_count": len(shards),
        "total_rows": total_rows,
        "distinct_arrow_schemas": len(schema_strings),
        "shards": shard_reports,
        "sample": {
            "rows": sampled_rows,
            "task_values": dict(task_values),
            "source_values": dict(source_values),
            "images_per_row": {str(key): value for key, value in image_counts.items()},
            "qa_items_per_row": {str(key): value for key, value in qa_counts.items()},
            "image_dimensions": dimensions,
            "corrupted_images": corrupt,
            "duplicate_image_hashes": {
                key: count for key, count in image_hashes.items() if count > 1
            },
            "examples": samples,
        },
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SonoInstruct schema inspection",
        "",
        f"- Data root: `{report['data_root']}`",
        f"- Parquet shards: {report['shard_count']}",
        f"- Source rows: {report['total_rows']:,}",
        f"- Distinct Arrow schemas: {report['distinct_arrow_schemas']}",
        f"- Sampled rows: {report['sample']['rows']}",
        f"- Corrupted sampled images: {report['sample']['corrupted_images']}",
        "",
        "## Observed columns",
        "",
        "| Column | Arrow type | Nullable |",
        "|---|---|---|",
    ]
    if report["shards"]:
        for column in report["shards"][0]["columns"]:
            lines.append(f"| {column['name']} | `{column['arrow_type']}` | {column['nullable']} |")
    lines.extend(["", "## Shards", "", "| Shard | Rows | Size (bytes) |", "|---|---:|---:|"])
    for shard in report["shards"]:
        lines.append(
            f"| `{Path(shard['path']).name}` | {shard['rows']:,} | {shard['size_bytes']:,} |"
        )
    lines.extend(
        [
            "",
            "## Sample distributions",
            "",
            f"- Images per row: `{json.dumps(report['sample']['images_per_row'], sort_keys=True)}`",
            f"- QA items per row: `{json.dumps(report['sample']['qa_items_per_row'], sort_keys=True)}`",
            f"- Task values: `{json.dumps(report['sample']['task_values'], ensure_ascii=False, sort_keys=True)}`",
            f"- Source values: `{json.dumps(report['sample']['source_values'], ensure_ascii=False, sort_keys=True)}`",
            "",
            "Medical text examples are truncated in the JSON artifact. Embedded image bytes are never serialized.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sample-rows", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    report = inspect(args.data_root, max(0, args.sample_rows))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "dataset_schema.json", report)
    (args.output_dir / "dataset_schema.md").write_text(to_markdown(report), encoding="utf-8")
    print(to_markdown(report))


if __name__ == "__main__":
    main()
