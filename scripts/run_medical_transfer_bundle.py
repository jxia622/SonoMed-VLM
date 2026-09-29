#!/usr/bin/env python3
"""Run one pilot arm sequentially; fractions remain independent experiments."""

import argparse
import subprocess
import sys
from pathlib import Path


def commands(arm):
    runner = str(Path(__file__).with_name("run_medical_transfer.py"))
    prefix = [sys.executable, runner]
    stages = [["original"]] if arm == "direct" else [["intermediate", "--arm", arm, "--seed", "42"]]
    stages += [
        ["downstream", "--arm", arm, "--seed", "42", "--fraction", str(f)] for f in [1, 10, 100]
    ]
    return [prefix + stage for stage in stages]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=["direct", "medical", "general"])
    args = parser.parse_args()
    for command in commands(args.arm):
        print("Bundle stage:", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
