#!/usr/bin/env python3
"""Deterministic base or adapter evaluation with raw-output preservation."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from sonomed_vlm.config import load_config, save_resolved_config
from sonomed_vlm.data.collator import build_messages
from sonomed_vlm.data.sonoinstruct import ManifestDataset
from sonomed_vlm.eval.aggregate import aggregate_scores
from sonomed_vlm.eval.parsing import score_prediction
from sonomed_vlm.models.lora import apply_lora, find_decoder_lora_targets, parameter_report
from sonomed_vlm.models.medgemma import (
    architecture_report,
    load_medgemma,
    validate_training_device,
)
from sonomed_vlm.training.metadata import (
    collect_metadata,
    finalize_compute,
    resolve_dataset_revision,
    save_environment_files,
)
from sonomed_vlm.utils.io import read_jsonl, sha256_file, write_json, write_jsonl
from sonomed_vlm.utils.reproducibility import seed_everything


def _move_inputs(inputs: Any, model: Any) -> dict[str, Any]:
    device = next(model.parameters()).device
    dtype = next(
        parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()
    )
    moved = {}
    for key, value in dict(inputs).items():
        if hasattr(value, "to"):
            value = value.to(device)
            if value.is_floating_point():
                value = value.to(dtype=dtype)
        moved[key] = value
    return moved


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--adapter", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    validate_training_device(config)
    seed_everything(config.training.seed)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        import torch.distributed as dist

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    run_id = config.output.run_name or datetime.now(UTC).strftime("eval-%Y%m%dT%H%M%SZ")
    run_dir = config.output.root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    dataset = ManifestDataset(
        config.data.val_manifest,
        config.data.root,
        task_filters=config.data.task_filters,
        source_filters=config.data.source_filters,
        instruction_protocol=config.data.instruction_protocol,
    )
    model, processor = load_medgemma(config, for_training=distributed)
    if distributed:
        model = model.to(torch.device("cuda", local_rank))
        model.config.use_cache = True
        model.gradient_checkpointing_disable()
    resolved_model_class = type(model).__name__
    resolved_config_class = type(model.config).__name__
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    limit = len(dataset) if args.max_examples is None else min(len(dataset), args.max_examples)
    rows = []

    for index in range(rank, limit, world_size):
        example = dataset[index]
        messages = build_messages(example, include_answer=False)
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_length = inputs["input_ids"].shape[-1]
        inputs = _move_inputs(inputs, model)
        generation_kwargs = {
            "max_new_tokens": config.generation.max_new_tokens,
            "do_sample": config.generation.do_sample,
        }
        if config.generation.do_sample and config.generation.temperature is not None:
            generation_kwargs["temperature"] = config.generation.temperature
        with torch.inference_mode():
            generated = model.generate(**inputs, **generation_kwargs)
        output_tokens = generated[0][input_length:]
        raw_output = processor.decode(output_tokens, skip_special_tokens=True)
        record = {
            "_dataset_index": index,
            "example_id": example.example_id,
            "image_id": example.image_id,
            "task_family": example.task_family,
            "task_type": example.task_type,
            "source_dataset": example.source_dataset,
            "focus": example.focus,
            "prompt": example.user_prompt,
            "system_prompt": example.system_prompt,
            "instruction_protocol": config.data.instruction_protocol,
            "original_task_type": example.metadata.get("original_task_type", example.task_type),
            "ground_truth": example.assistant_response,
            "raw_model_output": raw_output,
            "answer_label": example.metadata.get("answer_label"),
            "options": example.metadata.get("options", []),
            "generation_settings": generation_kwargs,
        }
        record["score"] = score_prediction(record)
        record["parsed_prediction"] = record["score"].get("parsed_prediction")
        rows.append(record)
        if len(rows) % 100 == 0:
            print(f"rank={rank} completed={len(rows)} elapsed={time.monotonic() - started:.1f}s", flush=True)

    shard_path = run_dir / f"eval_predictions.rank-{rank:02d}.jsonl"
    write_jsonl(shard_path, rows)
    if distributed:
        dist.barrier()
    if rank != 0:
        dist.destroy_process_group()
        return
    if distributed:
        rows = [
            row
            for shard_rank in range(world_size)
            for row in read_jsonl(run_dir / f"eval_predictions.rank-{shard_rank:02d}.jsonl")
        ]
    rows.sort(key=lambda row: int(row.pop("_dataset_index")))
    write_jsonl(run_dir / "eval_predictions.jsonl", rows)
    metrics = aggregate_scores(rows)
    write_json(run_dir / "metrics.json", metrics)
    base_total = sum(parameter.numel() for parameter in model.parameters())
    if args.adapter:
        inspected_model = model
        injected_targets = sorted(
            {
                target
                for peft_config in model.peft_config.values()
                for target in (peft_config.target_modules or [])
            }
        )
        candidate_targets = injected_targets
    else:
        candidate_targets = find_decoder_lora_targets(model, config.lora.target_modules)
        inspected_model, injected_targets = apply_lora(model, config.lora)
    adapter_parameters = parameter_report(inspected_model)
    architecture = architecture_report(inspected_model, processor)
    architecture.update(
        {
            "base_total_parameters": base_total,
            "resolved_model_class": resolved_model_class,
            "resolved_config_class": resolved_config_class,
            "candidate_lora_target_modules": candidate_targets,
            "lora_target_modules": injected_targets,
            "vision_frozen_after_lora": True,
            "projector_trainable": config.lora.train_projector,
            **adapter_parameters.to_dict(),
        }
    )
    write_json(run_dir / "architecture.json", architecture)
    project_root = Path.cwd()
    artifact_dir = run_dir / "artifacts"
    write_json(artifact_dir / "medgemma_architecture.json", architecture)
    markdown_lines = [
        "# MedGemma runtime architecture",
        "",
        f"- Model class: `{architecture['model_class']}`",
        f"- Resolved base model class: `{architecture['resolved_model_class']}`",
        f"- Config class: `{architecture['config_class']}`",
        f"- Processor class: `{architecture['processor_class']}`",
        f"- Base parameters: {base_total:,}",
        f"- LoRA trainable parameters: {adapter_parameters.trainable:,}",
        f"- LoRA trainable percent: {adapter_parameters.trainable_percent:.6f}%",
        f"- Vision frozen: {architecture['vision_frozen_after_lora']}",
        f"- Projector trainable: {architecture['projector_trainable']}",
        "",
        "## Exact LoRA targets",
        "",
        *[f"- `{name}`" for name in injected_targets],
        "",
        "## Projector/connector candidates",
        "",
        *[f"- `{name}`" for name in architecture["projector_module_candidates"]],
        "",
    ]
    (artifact_dir / "medgemma_architecture.md").write_text(
        "\n".join(markdown_lines), encoding="utf-8"
    )
    save_resolved_config(config, run_dir / "config.yaml")
    metadata = collect_metadata(
        run_id=run_id,
        project_root=project_root,
        train_manifest=config.data.val_manifest,
        seed=config.training.seed,
        model_revision=getattr(model.config, "_commit_hash", None) or config.model.revision,
        dataset_revision=resolve_dataset_revision(config.data.root, config.data.val_manifest),
    )
    metadata["evaluated_examples"] = len(rows)
    metadata["instruction_protocol"] = config.data.instruction_protocol
    metadata["model_name"] = config.model.name
    metadata["adapter_path"] = str(args.adapter.resolve()) if args.adapter else None
    if args.adapter:
        metadata["adapter_sha256"] = sha256_file(args.adapter / "adapter_model.safetensors")
    save_environment_files(run_dir, metadata)
    finalize_compute(run_dir / "metadata.json", started, examples=len(rows))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
