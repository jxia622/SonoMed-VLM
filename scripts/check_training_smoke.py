#!/usr/bin/env python3
"""Gate full training on completed, finite smoke training and adapter reload."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def check_smoke(root: Path) -> dict:
    run = root / "qwen3vl_1pct_smoke"
    metrics = json.loads((run / "metrics.json").read_text())
    for key in ("train_loss", "eval_loss"):
        if not math.isfinite(metrics[key]) or metrics[key] <= 0:
            raise ValueError(f"Invalid smoke metric {key}: {metrics[key]}")
    if metrics.get("epoch", 0) < 0.99:
        raise ValueError("Smoke training did not complete one epoch")
    architecture = json.loads((run / "architecture.json").read_text())
    names = architecture["trainable_parameter_names"]
    if not names or any("lora_" not in name or "visual" in name for name in names):
        raise ValueError("Expected decoder-only LoRA with frozen vision")
    adapter = run / "final_adapter"
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter / name).is_file() or not (adapter / name).stat().st_size:
            raise ValueError(f"Missing saved adapter file: {name}")
    predictions = root / "qwen3vl_reload_smoke" / "eval_predictions.jsonl"
    rows = [json.loads(line) for line in predictions.read_text().splitlines() if line]
    if len(rows) != 8:
        raise ValueError("Expected eight successful generations after adapter reload")
    if any(not row.get("raw_model_output", "").strip() for row in rows):
        raise ValueError("Adapter reload generated an empty answer")
    return {"passed": True, "train_loss": metrics["train_loss"], "eval_loss": metrics["eval_loss"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    result = check_smoke(args.output_root)
    (args.output_root / "smoke_passed.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
