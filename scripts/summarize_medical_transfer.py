#!/usr/bin/env python3
"""Validate all experiment outputs and write per-run and paired seed comparisons."""

import csv
import json
from collections import defaultdict
from statistics import mean, stdev

import _bootstrap  # noqa: F401
from check_open_qa_run import check
from run_medical_transfer import DATA, LEGACY, OUT, SONO, check_training

from sonomed_vlm.eval.aggregate import aggregate_scores
from sonomed_vlm.eval.parsing import score_prediction
from sonomed_vlm.utils.io import read_jsonl, sha256_file, sha256_json, write_json


def main():
    results = []
    reference = {}
    for seed in [42, 43, 44]:
        for arm in ["direct", "medical", "general"]:
            for fraction in [1, 10, 100]:
                name = f"{arm}_seed{seed}_{fraction}pct"
                if not json.loads((OUT / name / "run_passed.json").read_text())["passed"]:
                    raise ValueError("Incomplete run " + name)
                reused = arm == "direct" and seed == 42
                folder = (
                    LEGACY / f"openqa_v2_qwen_{fraction}pct_eval"
                    if reused
                    else OUT / (name + "_ultrasound_eval")
                )
                training = LEGACY / f"openqa_v2_qwen_{fraction}pct" if reused else OUT / name
                check(folder, SONO / "val.jsonl", training)
                initial = (
                    None
                    if arm == "direct"
                    else OUT / f"{arm}_seed{seed}_intermediate/final_adapter"
                )
                check_training(training, initial)
                evaluations = [("ultrasound", folder)] + [
                    (domain, OUT / (name + "_" + domain + "_test"))
                    for domain in ["medical", "general"]
                ]
                for domain, directory in evaluations:
                    if not json.loads((directory / "protocol_passed.json").read_text())["passed"]:
                        raise ValueError("Invalid evaluation " + str(directory))
                    rows = list(read_jsonl(directory / "eval_predictions.jsonl"))
                    signatures = {
                        r["example_id"]: sha256_json(
                            {
                                k: r[k]
                                for k in [
                                    "prompt",
                                    "system_prompt",
                                    "ground_truth",
                                    "generation_settings",
                                    "task_type",
                                    "options",
                                ]
                            }
                        )
                        for r in rows
                    }
                    if domain in reference and signatures != reference[domain]:
                        raise ValueError("Unmatched evaluation inputs: " + name)
                    reference[domain] = signatures
                    for r in rows:
                        r["score"] = score_prediction(r)
                    results.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "fraction": fraction,
                            "domain": domain,
                            "name": name,
                            "predictions_sha256": sha256_file(directory / "eval_predictions.jsonl"),
                            **aggregate_scores(rows),
                        }
                    )
    diagnostics = []
    for name in ["original"] + [
        f"{arm}_seed{seed}_intermediate" for seed in [42, 43, 44] for arm in ["medical", "general"]
    ]:
        for domain in ["medical", "general"]:
            directory = OUT / (name + "_" + domain + "_test")
            if not json.loads((directory / "protocol_passed.json").read_text())["passed"]:
                raise ValueError("Incomplete knowledge diagnostic")
            rows = list(read_jsonl(directory / "eval_predictions.jsonl"))
            signatures = {
                r["example_id"]: sha256_json(
                    {
                        k: r[k]
                        for k in [
                            "prompt",
                            "system_prompt",
                            "ground_truth",
                            "generation_settings",
                            "task_type",
                            "options",
                        ]
                    }
                )
                for r in rows
            }
            if signatures != reference[domain]:
                raise ValueError("Changed diagnostic inputs")
            for r in rows:
                r["score"] = score_prediction(r)
            diagnostics.append({"name": name, "domain": domain, **aggregate_scores(rows)})
    grouped = defaultdict(list)
    for row in results:
        for metric, value in row["overall"].items():
            grouped[row["arm"], row["fraction"], row["domain"], metric].append(value)
    summary = [
        {
            "arm": a,
            "fraction": f,
            "domain": d,
            "metric": m,
            "mean": mean(v),
            "seed_sd": stdev(v),
            "n_seeds": len(v),
        }
        for (a, f, d, m), v in sorted(grouped.items())
    ]
    paired = []
    lookup = {(r["arm"], r["seed"], r["fraction"], r["domain"]): r["overall"] for r in results}
    for comparator in ["direct", "general"]:
        for fraction in [1, 10, 100]:
            for domain in ["ultrasound", "medical", "general"]:
                for metric in lookup["medical", 42, fraction, domain]:
                    differences = [
                        lookup["medical", s, fraction, domain][metric]
                        - lookup[comparator, s, fraction, domain][metric]
                        for s in [42, 43, 44]
                    ]
                    paired.append(
                        {
                            "comparison": "medical-minus-" + comparator,
                            "fraction": fraction,
                            "domain": domain,
                            "metric": metric,
                            "mean_difference": mean(differences),
                            "seed_sd": stdev(differences),
                            "seed_differences": differences,
                        }
                    )
    target = OUT / "summary"
    target.mkdir(exist_ok=False)
    write_json(
        target / "results.json",
        {
            "runs": results,
            "knowledge_diagnostics": diagnostics,
            "seed_summary": summary,
            "paired_comparisons": paired,
            "data_audit_sha256": sha256_file(DATA / "audit.json"),
            "limitations": "Internal ultrasound validation; three seeds; lexical QA metrics are not clinical accuracy; no significance claim from seed SD.",
        },
    )
    with (target / "seed_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "# Medical intermediate-training experiment",
        "",
        "Mean across three seeds ± sample SD. Internal validation; open-ended lexical scores, not MCQ accuracy.",
        "",
        "| Fraction | Arm | Open-QA exact match | Open-QA token F1 | Grounding IoU |",
        "|---|---|---:|---:|---:|",
    ]
    for fraction in [1, 10, 100]:
        for arm in ["direct", "general", "medical"]:
            cells = []
            for metric in ["answer_exact_match", "answer_token_f1", "iou"]:
                values = grouped[arm, fraction, "ultrasound", metric]
                cells.append(f"{mean(values):.4f} ± {stdev(values):.4f}")
            lines.append(f"| {fraction}% | {arm} | " + " | ".join(cells) + " |")
    (target / "results.md").write_text("\n".join(lines) + "\n")
    print("Verified 27 downstream comparisons and all knowledge diagnostics.")


if __name__ == "__main__":
    main()
