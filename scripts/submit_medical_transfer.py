#!/usr/bin/env python3
"""Submit a smoke-gated DAG, with an append-only receipt after each submission."""

import argparse
import csv
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-job", type=int, help="Previously submitted smoke in the same frozen stage"
    )
    args = parser.parse_args()
    root = Path(os.environ["SONOMED_PROJECT_ROOT"])
    os.chdir(root)
    (root / "logs").mkdir(exist_ok=True)
    baseline_state = (
        subprocess.check_output(
            ["sacct", "-M", "gpu", "-j", "4079527", "-X", "-n", "--format=State"],
            text=True,
        )
        .strip()
        .split()
    )
    if len(baseline_state) != 1 or baseline_state[0] not in {
        "COMPLETED",
        "RUNNING",
        "PENDING",
        "COMPLETING",
    }:
        raise RuntimeError(f"Reused full baseline is unavailable or failed: {baseline_state}")
    # Refuse accidental duplicate deployment, even after a partial submit failure.
    receipt = open(root / "submission-medical-transfer.tsv", "x", buffering=1)
    writer = csv.writer(receipt, delimiter="\t", lineterminator="\n")
    writer.writerow(["name", "job_id", "cluster", "dependencies", "time_limit"])
    jobs = []

    def submit(name, arguments, limit, deps=(), summary=False):
        command = [
            "sbatch",
            "--parsable",
            "--qos=gpu-rtx6k-s",
            "--job-name=" + name,
            "--time=" + limit,
            "--kill-on-invalid-dep=yes",
        ]
        if deps:
            command.append("--dependency=afterok:" + ":".join(map(str, deps)))
        command += [
            "slurm/medical_transfer_summary.slurm" if summary else "slurm/medical_transfer.slurm",
            *arguments,
        ]
        raw = subprocess.check_output(command, text=True).strip()
        jid, cluster = raw.split(";")
        if cluster != "gpu":
            raise ValueError("Unexpected cluster: " + raw)
        writer.writerow([name, jid, cluster, ":".join(map(str, deps)), limit])
        receipt.flush()
        os.fsync(receipt.fileno())
        jobs.append(jid)
        print(name, raw, flush=True)
        return jid

    if args.smoke_job:
        smoke = str(args.smoke_job)
        writer.writerow(["smoke-existing", smoke, "gpu", "", "02:00:00"])
    else:
        smoke = submit("mt-smoke", ["smoke"], "02:00:00")
    submit("mt-original", ["original"], "02:00:00", [smoke])
    for seed in [42, 43, 44]:
        for arm in ["direct", "medical", "general"]:
            dependency = smoke
            if arm != "direct":
                dependency = submit(
                    f"mt-{arm}-s{seed}-mid",
                    ["intermediate", "--arm", arm, "--seed", str(seed)],
                    "04:00:00",
                    [smoke],
                )
            for fraction in [1, 10, 100]:
                deps = [dependency]
                # Existing corrected seed-42 full baseline may still be running.
                if (
                    arm == "direct"
                    and seed == 42
                    and fraction == 100
                    and baseline_state[0] != "COMPLETED"
                ):
                    deps.append("4079527")
                limit = (
                    "03:00:00" if fraction == 1 else "05:00:00" if fraction == 10 else "16:00:00"
                )
                submit(
                    f"mt-{arm}-s{seed}-{fraction}",
                    ["downstream", "--arm", arm, "--seed", str(seed), "--fraction", str(fraction)],
                    limit,
                    deps,
                )
    submit("mt-summary", [], "00:30:00", list(jobs), summary=True)
    receipt.close()


if __name__ == "__main__":
    main()
