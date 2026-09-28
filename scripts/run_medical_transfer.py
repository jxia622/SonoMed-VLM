#!/usr/bin/env python3
"""One Slurm node in the predeclared medical intermediate-training experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml
from check_open_qa_run import check as check_ultrasound

from sonomed_vlm.config import load_config
from sonomed_vlm.data.text_qa import SYSTEM
from sonomed_vlm.utils.io import read_jsonl, sha256_file, write_json

ROOT = Path(os.environ.get("SONOMED_PROJECT_ROOT", ".")).resolve()
OUT = Path(os.environ.get("SONOMED_OUTPUT_ROOT", "outputs")).resolve()
DATA = ROOT / "data/medical-transfer"
SONO = ROOT / "data/manifests-openqa-v2"
LEGACY = Path("/ix1/kkim/xiac/sonomed-vlm-outputs/openqa-v2-20260927")
PYTHON = os.environ.get("SONOMED_PYTHON", "python")


def run(script, *args):
    subprocess.run(
        [
            PYTHON,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            "scripts/" + script,
            *map(str, args),
        ],
        check=True,
        cwd=ROOT,
    )


def config(name, seed, *, text=False, train=None, val=None, smoke=False):
    path = ROOT / "configs/medical_transfer_generated" / (name + ".yaml")
    values = {
        "extends": "../qwen3vl_base.yaml",
        "data": {
            "format": "text_qa" if text else "sonoinstruct",
            "instruction_protocol": "open_qa_v2",
            "train_manifest": str(train or SONO / "train_1pct.jsonl"),
            "val_manifest": str(val or SONO / "val.jsonl"),
            "max_length": None,
        },
        "training": {
            "seed": seed,
            "epochs": 1,
            "eval_steps": 100000000,
            "save_steps": 100000000,
            "logging_steps": 1 if smoke else 10,
            "max_steps": 4 if smoke else -1,
        },
        "output": {"run_name": name},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(values, sort_keys=False))
    load_config(path)
    return path


def check_training(folder, initial=None, smoke=False):
    metrics = json.loads((folder / "metrics.json").read_text())
    metadata = json.loads((folder / "metadata.json").read_text())
    architecture = json.loads((folder / "architecture.json").read_text())
    if not smoke and metrics.get("epoch", 0) < 0.99:
        raise ValueError("Incomplete one-epoch training")
    for field in ["train_loss", "eval_loss"]:
        if not math.isfinite(metrics[field]) or metrics[field] <= 0:
            raise ValueError("Invalid training loss")
    if not architecture["vision_frozen"] or architecture["projector_trainable"]:
        raise ValueError("Unexpected vision or projector adaptation")
    if architecture["trainable_parameters"] != 33030144:
        raise ValueError("Trainable capacity differs from the frozen Qwen recipe")
    expected = sha256_file(initial / "adapter_model.safetensors") if initial else None
    if metadata.get("initial_adapter_sha256") != expected:
        raise ValueError("Wrong intermediate adapter provenance")
    if not (folder / "final_adapter/adapter_model.safetensors").stat().st_size:
        raise ValueError("Empty adapter")


def train(name, seed, *, text=False, manifest=None, val=None, initial=None, smoke=False):
    folder = OUT / name
    if folder.exists():
        raise FileExistsError(folder)
    cfg = config(name, seed, text=text, train=manifest, val=val, smoke=smoke)
    extra = ["--initialize-adapter", initial] if initial else []
    run("train.py", "--config", cfg, "--strict-numerics", *extra)
    check_training(folder, initial, smoke)
    return folder / "final_adapter"


def evaluate(name, seed, adapter=None, *, text=False, manifest=None):
    folder = OUT / name
    if folder.exists():
        raise FileExistsError(folder)
    manifest = manifest or SONO / "val.jsonl"
    cfg = config(name, seed, text=text, val=manifest)
    extra = ["--adapter", adapter] if adapter else []
    run("baseline_eval.py", "--config", cfg, *extra)
    if text:
        rows = list(read_jsonl(folder / "eval_predictions.jsonl"))
        source = {r["example_id"]: r for r in read_jsonl(manifest)}
        if len(rows) != len(source) or {r["example_id"] for r in rows} != set(source):
            raise ValueError("Incomplete text knowledge evaluation")
        for row in rows:
            gold = source[row["example_id"]]
            if (
                row["system_prompt"] != SYSTEM
                or row["options"]
                or row["answer_label"] is not None
                or row["prompt"] != gold["question"]
                or row["ground_truth"] != gold["answer"]
                or "answer_exact_match" not in row["score"]
            ):
                raise ValueError("Invalid text QA evaluation protocol")
        metadata = json.loads((folder / "metadata.json").read_text())
        expected = sha256_file(adapter / "adapter_model.safetensors") if adapter else None
        if metadata.get("adapter_sha256") != expected or metadata[
            "dataset_manifest_sha256"
        ] != sha256_file(manifest):
            raise ValueError("Wrong evaluation provenance")
        write_json(
            folder / "protocol_passed.json",
            {"passed": True, "examples": len(rows), "format": "text_qa"},
        )
    else:
        result = check_ultrasound(folder, manifest)
        write_json(folder / "protocol_passed.json", result)


def diagnostics(name, seed, adapter=None):
    for domain in ["medical", "general"]:
        evaluate(
            name + "_" + domain + "_test",
            seed,
            adapter,
            text=True,
            manifest=DATA / (domain + "_test.jsonl"),
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=["smoke", "intermediate", "downstream", "original"])
    p.add_argument("--arm", choices=["direct", "medical", "general"], default="direct")
    p.add_argument("--seed", type=int, choices=[42, 43, 44], default=42)
    p.add_argument("--fraction", type=int, choices=[1, 10, 100], default=1)
    args = p.parse_args()
    # Verify frozen data bytes before ANY experiment job starts.
    audit = json.loads((DATA / "audit.json").read_text())
    for name, entry in audit["manifests"].items():
        if sha256_file(DATA / (name + ".jsonl")) != entry["sha256"]:
            raise ValueError("Changed intermediate dataset")
    for name, digest in audit["sono_manifests"].items():
        if sha256_file(SONO / name) != digest:
            raise ValueError("Changed ultrasound dataset")
    if args.kind == "smoke":
        for arm in ["medical", "general"]:
            name = "smoke_" + arm
            adapter = train(
                name,
                42,
                text=True,
                manifest=DATA / (arm + "_smoke_train.jsonl"),
                val=DATA / (arm + "_smoke_eval.jsonl"),
                smoke=True,
            )
            evaluate(
                name + "_text_eval",
                42,
                adapter,
                text=True,
                manifest=DATA / (arm + "_smoke_eval.jsonl"),
            )
            adapted = train(
                name + "_ultrasound",
                42,
                initial=adapter,
                val=DATA / "ultrasound_smoke_eval.jsonl",
                smoke=True,
            )
            evaluate(
                name + "_ultrasound_eval",
                42,
                adapted,
                manifest=DATA / "ultrasound_smoke_eval.jsonl",
            )
        write_json(
            OUT / "smoke_passed.json",
            {"passed": True, "data_audit_sha256": sha256_file(DATA / "audit.json")},
        )
        return
    gate = json.loads((OUT / "smoke_passed.json").read_text())
    if not gate["passed"] or gate["data_audit_sha256"] != sha256_file(DATA / "audit.json"):
        raise ValueError("Smoke gate does not match this experiment")
    if args.kind == "original":
        diagnostics("original", 42)
        return
    name = f"{args.arm}_seed{args.seed}"
    if args.kind == "intermediate":
        if args.arm == "direct":
            raise ValueError("Direct arm has no intermediate stage")
        adapter = train(
            name + "_intermediate",
            args.seed,
            text=True,
            manifest=DATA / (args.arm + "_train.jsonl"),
            val=DATA / (args.arm + "_dev.jsonl"),
        )
        diagnostics(name + "_intermediate", args.seed, adapter)
        return
    name += f"_{args.fraction}pct"
    if args.arm == "direct" and args.seed == 42:
        old = LEGACY / f"openqa_v2_qwen_{args.fraction}pct"
        check_ultrasound(Path(str(old) + "_eval"), SONO / "val.jsonl", old)
        oldmeta = json.loads((old / "metadata.json").read_text())
        manifest = SONO / (
            "train.jsonl" if args.fraction == 100 else f"train_{args.fraction}pct.jsonl"
        )
        if (
            oldmeta["dataset_manifest_sha256"] != sha256_file(manifest)
            or oldmeta["seed"] != args.seed
        ):
            raise ValueError("Reused direct baseline training does not match")
        adapter = old / "final_adapter"
        (OUT / name).mkdir(parents=True, exist_ok=False)
        write_json(
            OUT / name / "reused.json",
            {
                "training": str(old),
                "evaluation": str(old) + "_eval",
                "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
            },
        )
    else:
        initial = (
            None
            if args.arm == "direct"
            else OUT / f"{args.arm}_seed{args.seed}_intermediate/final_adapter"
        )
        manifest = SONO / (
            "train.jsonl" if args.fraction == 100 else f"train_{args.fraction}pct.jsonl"
        )
        adapter = train(name, args.seed, manifest=manifest, initial=initial)
        evaluate(name + "_ultrasound_eval", args.seed, adapter)
        check_ultrasound(OUT / (name + "_ultrasound_eval"), SONO / "val.jsonl", OUT / name)
    diagnostics(name, args.seed, adapter)
    write_json(
        OUT / name / "run_passed.json",
        {
            "passed": True,
            "arm": args.arm,
            "seed": args.seed,
            "fraction": args.fraction,
            "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        },
    )


if __name__ == "__main__":
    main()
