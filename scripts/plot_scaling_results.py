#!/usr/bin/env python3
"""Generate the four release figures from the curated scaling summary."""

from __future__ import annotations

import csv
from pathlib import Path


def main() -> None:
    import matplotlib.pyplot as plt

    project_root = Path(__file__).resolve().parents[1]
    source = project_root / "results" / "scaling_results.csv"
    output_dir = project_root / "docs" / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    with source.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["scale_pct"] != "0"]

    scales = [int(row["scale_pct"]) for row in rows]
    series = {
        "validation_loss": ("Validation loss", [float(row["val_loss"]) for row in rows]),
        "mean_iou": ("Mean IoU", [float(row["mean_iou"]) for row in rows]),
        "localization_at_0_5": (
            "Localization@0.5",
            [float(row["localization_at_0_5"]) for row in rows],
        ),
        "open_response_f1": (
            "QA + open-response token F1",
            [float(row["open_token_f1"]) for row in rows],
        ),
    }
    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold"})
    for filename, (label, values) in series.items():
        figure, axis = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
        axis.plot(scales, values, color="#2563EB", marker="o", linewidth=2.3, markersize=6)
        axis.set_title(f"{label} vs. training-data scale")
        axis.set_xlabel("Training-data scale (%)")
        axis.set_ylabel(label)
        axis.set_xticks(scales)
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        for x_value, y_value in zip(scales, values, strict=True):
            axis.annotate(
                f"{y_value:.3f}",
                (x_value, y_value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        figure.savefig(output_dir / f"{filename}.png", dpi=220)
        plt.close(figure)


if __name__ == "__main__":
    main()
