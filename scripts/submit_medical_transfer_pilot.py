#!/usr/bin/env python3
"""Submit three seed-42 arm bundles and a smoke-gated pilot summary."""

import argparse
import csv
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-job", type=int, required=True, help="Smoke from this same frozen stage/output root"
    )
    args = parser.parse_args()
    root = Path(os.environ["SONOMED_PROJECT_ROOT"])
    os.chdir(root)
    # Reused direct adapters and evaluations must have completed successfully.
    for job in [4079527, args.smoke_job]:
        state = (
            subprocess.check_output(
                ["sacct", "-M", "gpu", "-j", str(job), "-X", "-n", "--format=State"], text=True
            )
            .strip()
            .split()
        )
        allowed = (
            {"COMPLETED"} if job == 4079527 else {"COMPLETED", "RUNNING", "PENDING", "COMPLETING"}
        )
        if len(state) != 1 or state[0] not in allowed:
            raise RuntimeError(f"Job {job} is unavailable/failed: {state}")
        if job == args.smoke_job:
            smoke_complete = state[0] == "COMPLETED"
    receipt = open(root / "submission-medical-transfer-pilot.tsv", "x", buffering=1)
    writer = csv.writer(receipt, delimiter="\t", lineterminator="\n")
    writer.writerow(["name", "job_id", "cluster", "dependencies", "time_limit"])
    writer.writerow(["smoke-existing", args.smoke_job, "gpu", "", "02:00:00"])
    jobs = []
    for arm in ["direct", "medical", "general", "summary"]:
        summary = arm == "summary"
        deps = jobs if summary else ([] if smoke_complete else [str(args.smoke_job)])
        limit = "00:30:00" if summary else "04:00:00" if arm == "direct" else "24:00:00"
        name = "mt-pilot-" + arm
        cmd = [
            "sbatch",
            "--parsable",
            "--qos=gpu-rtx6k-s",
            "--job-name=" + name,
            "--time=" + limit,
            "--kill-on-invalid-dep=yes",
        ]
        if deps:
            cmd.append("--dependency=afterok:" + ":".join(deps))
        cmd += (
            ["slurm/medical_transfer_summary.slurm", "--seeds", "42"]
            if summary
            else ["slurm/medical_transfer_bundle.slurm", arm]
        )
        raw = subprocess.check_output(cmd, text=True).strip()
        jid, cluster = raw.split(";")
        if cluster != "gpu":
            raise ValueError("Unexpected cluster: " + raw)
        writer.writerow([name, jid, cluster, ":".join(deps), limit])
        receipt.flush()
        os.fsync(receipt.fileno())
        jobs.append(jid)
        print(name, raw, flush=True)
    receipt.close()


if __name__ == "__main__":
    main()
