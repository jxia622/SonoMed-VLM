#!/usr/bin/env python3
"""Fail a corrected run unless full training, adapter reload and protocol checks pass."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import _bootstrap  # noqa: F401

from sonomed_vlm.data.open_qa import PROTOCOL, SYSTEM_PROMPT
from sonomed_vlm.utils.io import read_jsonl, sha256_file, write_json


def check(evaluation: Path, manifest: Path, training: Path | None = None) -> dict:
    expected = [r["example_id"] for r in read_jsonl(manifest)]
    rows = list(read_jsonl(evaluation / "eval_predictions.jsonl"))
    if len(rows) != len(expected) or len({r["example_id"] for r in rows}) != len(rows):
        raise ValueError("Incomplete/duplicate evaluation predictions")
    if {r["example_id"] for r in rows} != set(expected):
        raise ValueError("Evaluation IDs differ from the corrected manifest")
    metadata = json.loads((evaluation / "metadata.json").read_text())
    if metadata.get("instruction_protocol") != PROTOCOL or metadata[
        "dataset_manifest_sha256"
    ] != sha256_file(manifest):
        raise ValueError("Wrong evaluation protocol or manifest")
    converted = [r for r in rows if r["task_type"] == "open_qa"]
    if not converted:
        raise ValueError("No converted open-QA examples evaluated")
    for row in rows:
        if row.get("instruction_protocol") != PROTOCOL:
            raise ValueError("Missing protocol marker")
        if row["generation_settings"] != {"do_sample": False, "max_new_tokens": 256}:
            raise ValueError("Wrong generation protocol")
    for row in converted:
        if (
            row.get("options")
            or row.get("answer_label") is not None
            or row.get("system_prompt") != SYSTEM_PROMPT
        ):
            raise ValueError("Leaked choices/labels or old instruction in converted evaluation")
        if not row["ground_truth"].strip() or "semantic_choice_accuracy" in row["score"]:
            raise ValueError("Wrong target or choice-based scoring")
    metrics = json.loads((evaluation / "metrics.json").read_text())
    if any(not math.isfinite(v) for v in metrics["overall"].values()):
        raise ValueError("Nonfinite evaluation metrics")
    if training:
        values = json.loads((training / "metrics.json").read_text())
        if values.get("epoch", 0) < 0.99 or any(
            not math.isfinite(values[k]) or values[k] <= 0 for k in ["train_loss", "eval_loss"]
        ):
            raise ValueError("Training did not finish with finite positive losses")
        arch = json.loads((training / "architecture.json").read_text())
        if (
            not arch["vision_frozen"]
            or arch["projector_trainable"]
            or arch["trainable_parameters"] <= 0
        ):
            raise ValueError("Wrong adaptation settings")
        for name in ["adapter_config.json", "adapter_model.safetensors"]:
            if not (training / "final_adapter" / name).stat().st_size:
                raise ValueError("Missing saved adapter")
        if metadata.get("adapter_sha256") != sha256_file(
            training / "final_adapter/adapter_model.safetensors"
        ):
            raise ValueError("Evaluated adapter differs from saved adapter")
    return {
        "passed": True,
        "protocol": PROTOCOL,
        "examples": len(rows),
        "open_qa_examples": len(converted),
        "manifest_sha256": sha256_file(manifest),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--training", type=Path)
    args = p.parse_args()
    result = check(args.evaluation, args.manifest, args.training)
    write_json(args.evaluation / "protocol_passed.json", result)
    print(json.dumps(result))
