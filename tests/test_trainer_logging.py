from __future__ import annotations

import json
from types import SimpleNamespace

from sonomed_vlm.training.trainer import build_metrics_callback


def test_metrics_callback_writes_jsonl(tmp_path) -> None:
    path = tmp_path / "train_log.jsonl"
    callback = build_metrics_callback(path)
    state = SimpleNamespace(is_world_process_zero=True, global_step=3)
    callback.on_log(None, state, None, logs={"loss": 1.25, "grad_norm": 2.5})
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["step"] == 3
    assert record["loss"] == 1.25
    assert record["grad_norm"] == 2.5
