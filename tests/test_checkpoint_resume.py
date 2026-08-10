from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

from sonomed_vlm.training.checkpointing import (
    find_latest_checkpoint,
    resolve_resume_checkpoint,
)


def _checkpoint(root: Path, step: int, state_step: int | None = None) -> Path:
    path = root / f"checkpoint-{step}"
    path.mkdir()
    (path / "trainer_state.json").write_text(
        json.dumps({"global_step": step if state_step is None else state_step})
    )
    return path


def test_latest_checkpoint_is_numeric_and_complete(tmp_path: Path) -> None:
    _checkpoint(tmp_path, 9)
    latest = _checkpoint(tmp_path, 100)
    _checkpoint(tmp_path, 200, state_step=199)
    (tmp_path / "checkpoint-1000").mkdir()
    assert find_latest_checkpoint(tmp_path) == latest
    assert resolve_resume_checkpoint("latest", tmp_path) == str(latest)
    assert resolve_resume_checkpoint("auto-if-present", tmp_path) == str(latest)


def test_auto_if_present_starts_only_without_checkpoint(tmp_path: Path) -> None:
    assert resolve_resume_checkpoint("auto-if-present", tmp_path) is None


class _TinyTokenDataset(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        values = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        return {"input_ids": values, "attention_mask": torch.ones_like(values), "labels": values}


def _tiny_model() -> LlamaForCausalLM:
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=16,
        )
    )


def _arguments(output: Path, max_steps: int) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output),
        max_steps=max_steps,
        per_device_train_batch_size=2,
        save_strategy="steps",
        save_steps=1,
        logging_strategy="no",
        report_to=[],
        use_cpu=True,
        disable_tqdm=True,
    )


def test_transformers_resume_progresses_global_step(tmp_path: Path) -> None:
    first = Trainer(
        model=_tiny_model(),
        args=_arguments(tmp_path, max_steps=2),
        train_dataset=_TinyTokenDataset(),
    )
    first.train()
    checkpoint = find_latest_checkpoint(tmp_path)
    assert checkpoint is not None
    assert first.state.global_step == 2

    resumed = Trainer(
        model=_tiny_model(),
        args=_arguments(tmp_path, max_steps=3),
        train_dataset=_TinyTokenDataset(),
    )
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 3
