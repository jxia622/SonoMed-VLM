#!/usr/bin/env python3
"""Plot the completed single-seed pilot and export all aggregate evaluation rows."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, ScalarFormatter

COLORS = {"direct": "#126DCE", "medical": "#C05C36", "general": "#138578"}
LABELS = {
    "direct": "Direct ultrasound",
    "medical": "Medical QA first",
    "general": "General QA first",
}
ARMS = ["direct", "medical", "general"]
FRACTIONS = [1, 10, 100]
INK = "#172B46"
MUTED = "#516174"


def load_results(source):
    data = json.loads(source.read_text())
    if data["status"] != "single_seed_pilot" or data["seeds"] != [42]:
        raise ValueError("This release requires the completed seed-42 pilot")
    expected = {
        (a, f, d) for a in ARMS for f in FRACTIONS for d in ["ultrasound", "medical", "general"]
    }
    runs = {(r["arm"], r["fraction"], r["domain"]): r for r in data["runs"]}
    if len(data["runs"]) != 27 or set(runs) != expected:
        raise ValueError("Missing or duplicated pilot runs")
    for (_, _, domain), row in runs.items():
        if row["seed"] != 42 or row["examples"] != (9964 if domain == "ultrasound" else 500):
            raise ValueError("Unexpected seed or evaluation denominator")
    diagnostics = {(r["name"], r["domain"]): r for r in data["knowledge_diagnostics"]}
    expected_diagnostics = {
        (n, d)
        for n in ["original", "medical_seed42_intermediate", "general_seed42_intermediate"]
        for d in ["medical", "general"]
    }
    if len(data["knowledge_diagnostics"]) != 6 or set(diagnostics) != expected_diagnostics:
        raise ValueError("Missing or duplicated knowledge diagnostics")
    if any(r["examples"] != 500 for r in diagnostics.values()):
        raise ValueError("Unexpected diagnostic denominator")
    if any(r["n_seeds"] != 1 or r["seed_sd"] is not None for r in data["seed_summary"]):
        raise ValueError("Pilot uncertainty must remain unspecified")
    return data, runs, diagnostics


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=root / "results/medical_transfer_pilot/results.json"
    )
    parser.add_argument("--output-dir", type=Path, default=root / "docs/assets/medical-transfer")
    parser.add_argument(
        "--csv", type=Path, default=root / "results/medical_transfer_pilot/evaluation_metrics.csv"
    )
    args = parser.parse_args()
    data, runs, diagnostics = load_results(args.source)
    metric_keys = sorted(
        {k for r in [*data["runs"], *data["knowledge_diagnostics"]] for k in r["overall"]}
    )
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stage",
                "name",
                "arm",
                "fraction",
                "seed",
                "domain",
                "examples",
                *metric_keys,
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for r in data["knowledge_diagnostics"]:
            arm = "original" if r["name"] == "original" else r["name"].split("_")[0]
            writer.writerow(
                {
                    "stage": "original" if arm == "original" else "intermediate",
                    "name": r["name"],
                    "arm": arm,
                    "seed": "" if arm == "original" else 42,
                    "domain": r["domain"],
                    "examples": r["examples"],
                    **r["overall"],
                }
            )
        for r in data["runs"]:
            writer.writerow(
                {
                    "stage": "downstream",
                    **{k: r[k] for k in ["name", "arm", "fraction", "seed", "domain", "examples"]},
                    **r["overall"],
                }
            )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": "#D5DEE7",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.titlepad": 16,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def frame(title, subtitle, footer, *, diagnostics_plot=False):
        fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.8))
        fig.subplots_adjust(
            left=0.17 if diagnostics_plot else 0.08, right=0.955, top=0.71, bottom=0.21, wspace=0.36
        )
        fig.text(
            0.06,
            0.94,
            "SONOMED-VLM  /  COMPLETED MEDICAL-TRAINING PILOT",
            fontsize=10,
            weight="bold",
            color=COLORS["direct"],
        )
        fig.text(0.06, 0.865, title, fontsize=22, weight="bold")
        fig.text(0.06, 0.80, subtitle, fontsize=11, color=MUTED)
        fig.text(0.06, 0.065, footer, fontsize=10, color=MUTED, linespacing=1.6)
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="x" if diagnostics_plot else "y", color="#E4EAF0", linewidth=0.8)
            ax.set_axisbelow(True)
        return fig, axes

    def save(fig, name):
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(
                args.output_dir / f"{name}.{ext}", dpi=200, metadata={"Creator": "SonoMed-VLM"}
            )
        svg = args.output_dir / f"{name}.svg"
        svg.write_text("\n".join(s.rstrip() for s in svg.read_text().splitlines()) + "\n")
        plt.close(fig)

    fig, axes = frame(
        "Similar ultrasound performance across training paths",
        "Same Qwen3-VL 4B backbone  |  Independent 1%, 10% and 100% ultrasound adaptations",
        "One seed (42) · Same internal validation set · No confidence intervals or significance claims\nOpen-QA exact match: 4,938 examples. Grounding IoU: 565 examples. Lines connect measured runs.",
    )
    for ax, metric, title in zip(
        axes,
        ["answer_exact_match", "iou"],
        ["Converted open-QA exact match", "Mean grounding intersection-over-union"],
        strict=True,
    ):
        for arm, marker, style in zip(ARMS, ["o", "s", "^"], ["-", "--", ":"], strict=True):
            values = [runs[arm, f, "ultrasound"]["overall"][metric] for f in FRACTIONS]
            ax.plot(
                FRACTIONS,
                values,
                marker=marker,
                linestyle=style,
                linewidth=2.2,
                markersize=7,
                fillstyle="none" if arm == "direct" else "full",
                color=COLORS[arm],
                label=LABELS[arm],
            )
        ax.set_xscale("log")
        ax.set_xticks(FRACTIONS)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.set_xlim(0.8, 125)
        ax.set_ylim(0, 1)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Ultrasound training-data fraction (%)", labelpad=9)
        ax.legend(loc="lower right", frameon=False, fontsize=10)
        if metric == "answer_exact_match":
            ax.yaxis.set_major_formatter(PercentFormatter(1))
    save(fig, "ultrasound_scaling")

    fig, axes = frame(
        "Medical training yields small, mixed changes",
        "Medical-first minus each control  |  Positive values favor medical-first",
        "Differences between single-seed observations; no uncertainty estimates. Axis ranges differ between panels.\nMedical QA first does not consistently improve both answer matching and spatial overlap over direct tuning.",
    )
    for ax, metric, title, scale in zip(
        axes,
        ["answer_exact_match", "iou"],
        ["Open-QA exact-match difference (pp)", "Mean grounding IoU difference"],
        [100, 1],
        strict=True,
    ):
        for i, comparator in enumerate(["direct", "general"]):
            diffs = [
                (
                    runs["medical", f, "ultrasound"]["overall"][metric]
                    - runs[comparator, f, "ultrasound"]["overall"][metric]
                )
                * scale
                for f in FRACTIONS
            ]
            positions = [x + (i - 0.5) * 0.34 for x in range(3)]
            ax.bar(
                positions,
                diffs,
                width=0.29,
                color=COLORS[comparator],
                label="Medical minus " + ("direct" if comparator == "direct" else "general"),
            )
            for x, val in zip(positions, diffs, strict=True):
                ax.annotate(
                    f"{val:+.2f}" if scale == 100 else f"{val:+.4f}",
                    (x, val),
                    xytext=(0, 6 if val >= 0 else -8),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=10,
                )
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_ylim((-0.8, 2.1) if scale == 100 else (-0.011, 0.019))
        ax.set_xticks([0, 1, 2], ["1%", "10%", "100%"])
        ax.set_title(title, loc="left")
        ax.set_xlabel("Ultrasound training-data fraction", labelpad=9)
        ax.legend(loc="upper right", frameon=False, fontsize=9)
    save(fig, "paired_differences")

    states = [
        "Original Qwen",
        "Medical QA only",
        "General QA only",
        "Direct → US 100%",
        "Medical → US 100%",
        "General → US 100%",
    ]
    colors = [
        "#8A9CAF",
        COLORS["medical"],
        COLORS["general"],
        COLORS["direct"],
        COLORS["medical"],
        COLORS["general"],
    ]
    fig, axes = frame(
        "Text-task gains do not fully persist after ultrasound",
        "Held-out open-ended diagnostics  |  500 medical questions and 500 general questions",
        "Exact match measures reference agreement, not clinical reasoning. No options shown. Panels use different x-axis ranges.\nIntermediate-only checkpoints and full-data downstream checkpoints use the same fixed diagnostic questions.",
        diagnostics_plot=True,
    )
    for ax, domain, title, limit in zip(
        axes,
        ["medical", "general"],
        ["Medical QA exact match", "General QA exact match"],
        [0.125, 0.82],
        strict=True,
    ):
        values = [
            diagnostics[name, domain]["overall"]["answer_exact_match"]
            for name in ["original", "medical_seed42_intermediate", "general_seed42_intermediate"]
        ]
        values += [runs[arm, 100, domain]["overall"]["answer_exact_match"] for arm in ARMS]
        ax.barh(range(6), values, color=colors, height=0.62)
        ax.set_yticks(range(6), states if domain == "medical" else [])
        ax.invert_yaxis()
        ax.set_xlim(0, limit)
        ax.set_title(title, loc="left")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        for y, val in enumerate(values):
            ax.text(val + limit * 0.02, y, f"{val:.1%}", va="center", fontsize=11)
    save(fig, "knowledge_diagnostics")
    print("Validated 33 evaluation records; exported CSV and three PNG/SVG/PDF figures.")


if __name__ == "__main__":
    main()
