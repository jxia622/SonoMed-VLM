#!/usr/bin/env python3
"""Single/DDP MedGemma LoRA training entry point."""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import _bootstrap  # noqa: F401

from sonomed_vlm.config import load_config, save_resolved_config
from sonomed_vlm.data.collator import AssistantOnlyMultimodalCollator
from sonomed_vlm.data.sonoinstruct import ManifestDataset
from sonomed_vlm.models.lora import apply_lora, parameter_report
from sonomed_vlm.models.medgemma import architecture_report, load_medgemma, validate_training_device
from sonomed_vlm.training.checkpointing import resolve_resume_checkpoint
from sonomed_vlm.training.distributed import world_info
from sonomed_vlm.training.metadata import (
    collect_metadata,
    finalize_compute,
    resolve_dataset_revision,
    save_environment_files,
)
from sonomed_vlm.training.trainer import build_trainer
from sonomed_vlm.utils.io import write_json
from sonomed_vlm.utils.logging import configure_logging
from sonomed_vlm.utils.reproducibility import seed_everything

LOGGER = logging.getLogger("sonomed.train")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--max-eval-examples", type=int, default=None,
                        help="Limit validation for an engineering smoke test only")
    parser.add_argument("--strict-numerics", action="store_true",
                        help="Fail the job if loss or gradient logs are non-finite")
    args = parser.parse_args()
    if args.max_eval_examples is not None and args.max_eval_examples < 1:
        parser.error("--max-eval-examples must be positive")
    config = load_config(args.config)
    if args.resume_from_checkpoint:
        config.training.resume_from_checkpoint = args.resume_from_checkpoint
    validate_training_device(config)
    seed_everything(config.training.seed)
    world = world_info()
    run_id = config.output.run_name or datetime.now(UTC).strftime("run-%Y%m%dT%H%M%SZ")
    run_dir = config.output.root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(run_dir / "run.log")
    started = time.monotonic()

    train_dataset = ManifestDataset(
        config.data.train_manifest,
        config.data.root,
        task_filters=config.data.task_filters,
        source_filters=config.data.source_filters,
        instruction_protocol=config.data.instruction_protocol,
    )
    eval_dataset = ManifestDataset(
        config.data.val_manifest,
        config.data.root,
        task_filters=config.data.task_filters,
        source_filters=config.data.source_filters,
        instruction_protocol=config.data.instruction_protocol,
    )
    if args.max_eval_examples is not None:
        eval_dataset.records = eval_dataset.records[:args.max_eval_examples]
    model, processor = load_medgemma(config, for_training=True)
    architecture = architecture_report(model)
    model, targets = apply_lora(model, config.lora)
    parameters = parameter_report(model)
    if parameters.trainable == 0:
        raise RuntimeError("No trainable parameters remain after LoRA injection")
    architecture.update(
        {
            "lora_target_modules": targets,
            "vision_frozen": not config.lora.train_vision,
            "projector_trainable": config.lora.train_projector,
            **parameters.to_dict(),
        }
    )
    if world.rank == 0:
        save_resolved_config(config, run_dir / "config.yaml")
        write_json(run_dir / "architecture.json", architecture)
        revision = getattr(model.config, "_commit_hash", None) or config.model.revision
        metadata = collect_metadata(
            run_id=run_id,
            project_root=Path(os.environ.get("SONOMED_PROJECT_ROOT", Path.cwd())).resolve(),
            train_manifest=config.data.train_manifest,
            seed=config.training.seed,
            model_revision=revision,
            dataset_revision=resolve_dataset_revision(config.data.root, config.data.train_manifest),
        )
        metadata.update(
            {
                "train_examples": len(train_dataset),
                "instruction_protocol": config.data.instruction_protocol,
                "max_eval_examples": args.max_eval_examples,
                "validation_examples": len(eval_dataset),
                "effective_batch_size": config.training.per_device_batch_size
                * config.training.gradient_accumulation_steps
                * world.world_size,
            }
        )
        save_environment_files(run_dir, metadata)
        LOGGER.info(
            "model_class=%s trainable=%d total=%d percent=%.6f targets=%d",
            architecture["model_class"],
            parameters.trainable,
            parameters.total,
            parameters.trainable_percent,
            len(targets),
        )

    collator = AssistantOnlyMultimodalCollator(processor, max_length=config.data.max_length)
    trainer = build_trainer(
        model=model,
        processor=processor,
        collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
        run_dir=run_dir,
    )
    if args.strict_numerics:
        from transformers import TrainerCallback

        class FiniteMetricsCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                for key, value in (logs or {}).items():
                    if isinstance(value, (int, float)) and not math.isfinite(value):
                        raise FloatingPointError(f"Non-finite training metric {key}={value}")

        trainer.args.logging_nan_inf_filter = False
        trainer.add_callback(FiniteMetricsCallback())
    resume = resolve_resume_checkpoint(
        config.training.resume_from_checkpoint, run_dir / "checkpoints"
    )
    result = trainer.train(resume_from_checkpoint=resume)
    eval_metrics = next(
        (
            record
            for record in reversed(trainer.state.log_history)
            if record.get("step") == trainer.state.global_step and "eval_loss" in record
        ),
        None,
    )
    if eval_metrics is None:
        eval_metrics = trainer.evaluate()
    if world.rank == 0:
        trainer.save_model(run_dir / "final_adapter")
        processor.save_pretrained(run_dir / "final_adapter")
        metrics = dict(result.metrics)
        metrics.update(eval_metrics)
        write_json(run_dir / "metrics.json", metrics)
        finalize_compute(run_dir / "metadata.json", started, examples=len(train_dataset))


if __name__ == "__main__":
    main()
