#!/usr/bin/env python3
"""Rebuild research figures from aggregate results; no model/data access required."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, ScalarFormatter

COLORS = {"qwen": "#126DCE", "medgemma": "#C05C36"}
NAMES = {"qwen": "Qwen3-VL 4B", "medgemma": "MedGemma 1.5 4B"}
INK = "#172B46"
MUTED = "#516174"


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "results/research_results.json")
    parser.add_argument("--output-dir", type=Path, default=root / "docs/assets/research")
    args = parser.parse_args()
    data = json.loads(args.source.read_text())
    if data.get("protocol") != "open_qa_v2" or len(data["models"]) != 16:
        raise ValueError("Expected the complete 16-run corrected open-QA-v2 release")
    models = {
        f"{m['family']}_{m['scale_pct']}pct": m for m in data["models"] if m["kind"] != "legacy100"
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
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

    def frame(title, subtitle, footer):
        fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.8))
        fig.subplots_adjust(left=0.08, right=0.965, top=0.73, bottom=0.21, wspace=0.28)
        fig.text(
            0.06,
            0.94,
            "SONOMED-VLM  /  CORRECTED OPEN-QA V2",
            fontsize=10,
            weight="bold",
            color=COLORS["qwen"],
        )
        fig.text(0.06, 0.865, title, fontsize=23, weight="bold")
        fig.text(0.06, 0.80, subtitle, fontsize=11, color=MUTED)
        fig.text(0.06, 0.065, footer, fontsize=10, color=MUTED, linespacing=1.6)
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#E4EAF0", linewidth=0.8)
            ax.set_axisbelow(True)
        return fig, axes

    def save(fig, name):
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(
                args.output_dir / f"{name}.{ext}", dpi=200, metadata={"Creator": "SonoMed-VLM"}
            )
        # Matplotlib SVG paths contain trailing spaces; normalize for clean diffs.
        svg = args.output_dir / f"{name}.svg"
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        plt.close(fig)

    fig, axes = frame(
        "Ultrasound adaptation improves both models",
        "Original instruction checkpoints vs. full-data LoRA  |  Same 565 grounding examples",
        "One epoch · Seed 42 · Frozen vision encoders · No uncertainty intervals\n"
        "Held-out SonoInstruct validation split; these are not official SonoBench results.",
    )
    for ax, metric, title in zip(
        axes,
        ["iou", "localization_at_0_5"],
        ["Mean intersection-over-union", "Localization at IoU ≥ 0.5"],
        strict=True,
    ):
        for i, family in enumerate(["qwen", "medgemma"]):
            values = [models[f"{family}_{s}pct"]["overall"][metric] for s in [0, 100]]
            for j, value in enumerate(values):
                x = i + (j - 0.5) * 0.32
                ax.bar(
                    x,
                    value,
                    width=0.28,
                    color=COLORS[family],
                    alpha=0.30 if j == 0 else 1,
                    edgecolor=COLORS[family],
                    linewidth=0.8,
                )
                ax.text(
                    x,
                    value + 0.025,
                    f"{value:.3f}" if metric == "iou" else f"{value:.1%}",
                    ha="center",
                    fontsize=12,
                    weight="bold" if j else "normal",
                )
                ax.text(
                    x,
                    -0.075,
                    "Original" if j == 0 else "Adapted",
                    ha="center",
                    fontsize=9,
                    color=MUTED,
                )
        ax.set_ylim(0, 1.03)
        ax.set_xticks([0, 1], [NAMES["qwen"], NAMES["medgemma"]])
        ax.tick_params(axis="x", pad=26, length=0)
        ax.set_title(title, loc="left")
        if metric != "iou":
            ax.yaxis.set_major_formatter(PercentFormatter(1))
    save(fig, "grounding_comparison")

    fig, axes = frame(
        "Qwen reaches stronger grounding with less data",
        "Independent adapters at each fraction  |  Same nested training subsets and evaluation set",
        "Training fraction uses a logarithmic axis. Lines connect measured runs; no fitted extrapolation.\n"
        "All six fractions completed for both models. One seed; data efficiency is not compute equivalence.",
    )
    for ax, metric, title in zip(
        axes,
        ["iou", "localization_at_0_5"],
        ["Mean intersection-over-union", "Localization at IoU ≥ 0.5"],
        strict=True,
    ):
        for family in ["qwen", "medgemma"]:
            records = sorted(
                [
                    m
                    for m in data["models"]
                    if m["family"] == family and m["kind"] == "corrected_adapter"
                ],
                key=lambda m: m["scale_pct"],
            )
            x = [m["scale_pct"] for m in records]
            y = [m["overall"][metric] for m in records]
            ax.plot(
                x,
                y,
                color=COLORS[family],
                marker="o" if family == "qwen" else "s",
                linewidth=2.8,
                markersize=6,
                label=NAMES[family],
            )
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 25, 50, 100])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.set_xlim(0.8, 125)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Training-data fraction (%)", labelpad=9)
        ax.set_title(title, loc="left")
        if metric != "iou":
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(loc="lower right", frameon=False, fontsize=10)
    q5 = models["qwen_5pct"]["overall"]["iou"]
    med100 = models["medgemma_100pct"]["overall"]["iou"]
    axes[0].annotate(
        f"Qwen 5%: {q5:.3f}\nMedGemma 100%: {med100:.3f}",
        xy=(5, q5),
        xytext=(1.25, 0.83),
        fontsize=10,
        color=INK,
        bbox={"boxstyle": "round,pad=.6", "fc": "#EFF5FB", "ec": "none"},
        arrowprops={"arrowstyle": "-", "color": "#8A9CAF", "connectionstyle": "angle3"},
    )
    save(fig, "grounding_scaling")

    fig, axes = frame(
        "Language metrics show a much smaller separation",
        "Ordinary QA/open: 4,461 examples  |  Converted open-QA: 4,938 examples",
        "Corrected answer-text targets; no candidate options shown. Open-QA exact match is not MCQ accuracy.\n"
        "Token F1 measures lexical overlap, not clinical correctness. One seed per scale; no uncertainty intervals.",
    )
    for ax, metric, title in zip(
        axes,
        ["token_f1", "answer_exact_match"],
        ["QA + open-response token F1", "Converted open-QA exact match"],
        strict=True,
    ):
        for family in ["qwen", "medgemma"]:
            records = sorted(
                [
                    m
                    for m in data["models"]
                    if m["family"] == family and m["kind"] == "corrected_adapter"
                ],
                key=lambda m: m["scale_pct"],
            )
            ax.plot(
                [m["scale_pct"] for m in records],
                [m["overall"][metric] for m in records],
                color=COLORS[family],
                marker="o" if family == "qwen" else "s",
                linewidth=2.5,
                markersize=6,
                label=NAMES[family],
            )
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 25, 50, 100])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.set_xlim(0.8, 125)
        ax.set_ylim(0, 0.42 if metric == "token_f1" else 1)
        ax.set_xlabel("Training-data fraction (%)", labelpad=9)
        ax.set_title(title, loc="left")
        ax.legend(loc="lower right", frameon=False, fontsize=10)
        if metric == "answer_exact_match":
            ax.yaxis.set_major_formatter(PercentFormatter(1))
    save(fig, "language_diagnostics")
    fig, axes = frame(
        "Retraining changes more than the evaluation prompt",
        "Full-data adapters  |  Historical and corrected training, both evaluated with open-QA v2",
        "Bridge controls keep the evaluation protocol fixed. Training targets, filtering and sample counts change together.\n"
        "The corrected MedGemma adapter has lower observed grounding; correction does not guarantee score improvements.",
    )
    for ax, metric, title in zip(
        axes,
        ["answer_exact_match", "iou"],
        ["Converted open-QA exact match", "Mean intersection-over-union"],
        strict=True,
    ):
        for i, family in enumerate(["qwen", "medgemma"]):
            legacy = next(
                m for m in data["models"] if m["family"] == family and m["kind"] == "legacy100"
            )
            for j, record in enumerate([legacy, models[f"{family}_100pct"]]):
                value = record["overall"][metric]
                x = i + (j - 0.5) * 0.34
                ax.bar(x, value, width=0.28, color=COLORS[family], alpha=0.35 if j == 0 else 1)
                ax.text(
                    x,
                    value + 0.025,
                    f"{value:.2%}" if metric == "answer_exact_match" else f"{value:.4f}",
                    ha="center",
                    fontsize=11,
                )
                ax.text(
                    x,
                    -0.075,
                    "Legacy" if j == 0 else "Corrected",
                    ha="center",
                    fontsize=9,
                    color=MUTED,
                )
        ax.set_xticks([0, 1], [NAMES["qwen"], NAMES["medgemma"]])
        ax.tick_params(axis="x", pad=26, length=0)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, loc="left")
        if metric == "answer_exact_match":
            ax.yaxis.set_major_formatter(PercentFormatter(1))
    save(fig, "correction_bridge")
    print(f"Wrote 4 figures as PNG, SVG, and PDF to {args.output_dir}")


if __name__ == "__main__":
    main()
