from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from sonomed_vlm.data.sonoinstruct import ManifestDataset
from sonomed_vlm.data.splits import assert_no_image_leakage
from sonomed_vlm.utils.io import read_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _png(value: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("L", (4, 4), color=value).save(buffer, format="PNG")
    return buffer.getvalue()


def test_inspect_build_validate_manifest_cli_end_to_end(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    shard = data_root / "parquet_full_train" / "train_00000.parquet"
    shard.parent.mkdir(parents=True)
    rows = []
    for index in range(40):
        rows.append(
            {
                "QA": json.dumps(
                    [
                        {
                            "question": f"Question {index}?",
                            "answer": f"Answer {index}",
                            "answer_label": "",
                            "options": [],
                            "qa_type": "qa",
                            "system_prompt": "",
                        }
                    ]
                ),
                "dataset": f"source-{index % 2}",
                "focus": "fixture",
                "info": "{}",
                "type": "AR",
                "images": [{"bytes": _png(index)}],
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), shard)

    artifacts = tmp_path / "artifacts"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "inspect_dataset.py"),
            "--data-root",
            str(data_root),
            "--sample-rows",
            "2",
            "--output-dir",
            str(artifacts),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads((artifacts / "dataset_schema.json").read_text())["total_rows"] == 40

    manifests = tmp_path / "manifests"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_manifests.py"),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(manifests),
            "--val-fraction",
            "0.2",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    train = list(read_jsonl(manifests / "train.jsonl"))
    val = list(read_jsonl(manifests / "val.jsonl"))
    assert len(train) + len(val) == 40
    assert_no_image_leakage(train, val)
    dataset = ManifestDataset(manifests / "train.jsonl", data_root)
    assert dataset[0].assistant_response.startswith("Answer")
    for name in (
        "train_1pct.jsonl",
        "train_5pct.jsonl",
        "train_10pct.jsonl",
        "train_25pct.jsonl",
        "train_50pct.jsonl",
    ):
        assert (manifests / name).is_file()
    manifest_report = artifacts / "manifest_validation.json"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "check_manifests.py"),
            "--manifest-dir",
            str(manifests),
            "--output",
            str(manifest_report),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(manifest_report.read_text())["train_validation_image_overlap"] == 0
