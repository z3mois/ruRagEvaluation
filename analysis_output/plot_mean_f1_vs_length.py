from __future__ import annotations

import csv
import math
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "analysis_output" / "tables" / "full_grid.csv"
PLOT_DIR = ROOT / "analysis_output" / "plots"
FIGURE_DIR = ROOT / "text" / "figures"
TARGETS = ("relevance", "utilization", "adherence")
TARGETS_RU = {
    "relevance": "relevance",
    "utilization": "utilization",
    "adherence": "adherence",
}
EXPERIMENT = "RU Simple 3:3:1"
OUT_NAME = "f1_vs_length_RU_Simple_331.png"


def load_f1_by_target_and_model() -> dict[str, dict[str, list[tuple[int, float]]]]:
    points: dict[str, dict[str, list[tuple[int, float]]]] = {
        target: defaultdict(list) for target in TARGETS
    }

    with TABLE.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["experiment"] != EXPERIMENT:
                continue

            for target in TARGETS:
                points[target][row["model_short"]].append(
                    (int(row["max_length"]), float(row[f"test_{target}_f1"]))
                )

    return {
        target: {model: sorted(values) for model, values in by_model.items()}
        for target, by_model in points.items()
    }


def rounded_axis_limits(values: list[float]) -> tuple[float, float]:
    ymin = min(values)
    ymax = max(values)
    pad = max((ymax - ymin) * 0.18, 0.01)
    lower = max(0.0, math.floor((ymin - pad) * 100) / 100)
    upper = min(1.0, math.ceil((ymax + pad) * 100) / 100)
    return lower, upper


def main() -> None:
    by_target = load_f1_by_target_and_model()
    if not any(by_target.values()):
        raise RuntimeError(f"No rows found for experiment: {EXPERIMENT}")

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    markers = {"BGE-M3": "o", "DeBERTa-v3": "s", "ModernBERT": "^"}
    colors = {"BGE-M3": "#1f77b4", "DeBERTa-v3": "#ff7f0e", "ModernBERT": "#2ca02c"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)

    all_y = [
        score
        for by_model in by_target.values()
        for values in by_model.values()
        for _, score in values
    ]
    all_lengths = sorted(
        {
            length
            for by_model in by_target.values()
            for values in by_model.values()
            for length, _ in values
        }
    )

    for ax, target in zip(axes, TARGETS):
        for model in ("BGE-M3", "DeBERTa-v3", "ModernBERT"):
            values = by_target[target].get(model)
            if not values:
                continue

            lengths = [length for length, _ in values]
            scores = [score for _, score in values]

            ax.plot(
                lengths,
                scores,
                marker=markers.get(model, "o"),
                linewidth=2.2,
                markersize=7,
                color=colors.get(model),
                label=model,
            )

        ax.set_xscale("log", base=2)
        ax.set_xticks(all_lengths)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_ylim(*rounded_axis_limits(all_y))
        ax.set_xlabel("Максимальная длина контекста")
        ax.set_title(f"{TARGETS_RU[target]} — F1")
        ax.grid(True, alpha=0.28)

    axes[0].set_ylabel("Test F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / OUT_NAME
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    shutil.copy2(out, FIGURE_DIR / OUT_NAME)
    print(f"Saved: {out}")
    print(f"Copied: {FIGURE_DIR / OUT_NAME}")


if __name__ == "__main__":
    main()
