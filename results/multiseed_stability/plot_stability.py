"""Plot the six saved Validation AP observations without running a model."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
SEEDS = (42, 123, 2026)
STYLES = {
    "CLS-LoRA": {"linestyle": "-", "marker": "o", "markerfacecolor": "black"},
    "Mean-LoRA": {"linestyle": "--", "marker": "s", "markerfacecolor": "white"},
}


def main() -> None:
    values: dict[str, dict[int, float]] = {name: {} for name in STYLES}
    with (HERE / "seed_ap_values.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            name, seed = row["pooling_strategy"], int(row["seed"])
            if name not in values or seed not in SEEDS or seed in values[name]:
                raise ValueError(f"Unexpected or duplicate observation: {name}, {seed}")
            values[name][seed] = float(row["validation_ap"])
    if any(set(group) != set(SEEDS) for group in values.values()):
        raise ValueError("Expected exactly three seeds for each pooling strategy")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(6.4, 3.7), layout="constrained")
    x = range(len(SEEDS))
    for name, group in values.items():
        y = [group[seed] for seed in SEEDS]
        ax.plot(x, y, color="black", linewidth=1.5, markersize=6,
                markeredgecolor="black", label=name, **STYLES[name])
        for xpos, ap in zip(x, y):
            offset = 8 if name == "CLS-LoRA" else -15
            ax.annotate(f"{ap:.4f}", (xpos, ap), xytext=(0, offset),
                        textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(list(x), [str(seed) for seed in SEEDS])
    ax.set_xlim(-0.18, 2.18)
    ax.set_ylim(0.04, 0.215)
    ax.set_xlabel("Random Seed")
    ax.set_ylabel("Validation Average Precision (AP)")
    ax.grid(axis="y", linestyle=":", color="0.75", linewidth=0.7)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(HERE / "stability_comparison.svg", format="svg", metadata={"Creator": "Matplotlib"})
    plt.close(fig)


if __name__ == "__main__":
    main()
