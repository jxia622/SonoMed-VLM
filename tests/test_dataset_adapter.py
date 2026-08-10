from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sonomed_vlm.data.sonoinstruct import RawRow, iter_raw_rows, normalize_row


def _full_row(image: bytes) -> dict:
    return {
        "QA": json.dumps(
            [
                {
                    "question": "What structure?\n<image>",
                    "answer": "Kidney",
                    "answer_label": "B",
                    "options": ["A: Liver", "B: Kidney"],
                    "qa_type": "mcq",
                    "system_prompt": "Answer with one option.",
                },
                {
                    "question": "Describe it.",
                    "answer": "Normal kidney ultrasound.",
                    "answer_label": "",
                    "options": [],
                    "qa_type": "open",
                    "system_prompt": "",
                },
            ]
        ),
        "dataset": "synthetic",
        "focus": "Kidney",
        "info": json.dumps({"source_id": "fixture"}),
        "type": "AR",
        "images": [{"bytes": image}],
    }


def test_full_schema_flattens_qa_and_hashes_images(tmp_path: Path, png_bytes: bytes) -> None:
    shard = tmp_path / "parquet_full_train" / "train_00000.parquet"
    shard.parent.mkdir()
    pq.write_table(pa.Table.from_pylist([_full_row(png_bytes)]), shard)
    raw = next(iter_raw_rows(tmp_path))
    examples = normalize_row(raw, tmp_path)
    assert len(examples) == 2
    assert examples[0].image_id == examples[1].image_id
    assert examples[0].image_ids == examples[1].image_ids
    assert "<image>" not in examples[0].user_prompt
    assert examples[0].metadata["qa_index"] == 0
    assert examples[1].metadata["qa_index"] == 1
    assert examples[0].images[0].open().size == (8, 6)
    manifest = examples[0].to_manifest_record()
    assert "assistant_response" not in manifest
    assert "image_bytes" not in manifest


def test_preview_path_schema_is_supported(tmp_path: Path, png_bytes: bytes) -> None:
    image_path = tmp_path / "images" / "sample.png"
    image_path.parent.mkdir()
    image_path.write_bytes(png_bytes)
    row = {
        "image_path": ["images\\sample.png"],
        "dataset": "preview",
        "focus": "Fetus",
        "QA": [{"question": "Plane? <image>", "answer": "Four chamber", "type": "qa"}],
        "type": "AR",
    }
    examples = normalize_row(RawRow(tmp_path / "train_sample.parquet", 0, row), tmp_path)
    assert examples[0].images[0].image_path == image_path.resolve()
    assert examples[0].task_type == "qa"
