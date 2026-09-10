"""Render the observed PSYCON counts from the public statistics file."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    stats = json.loads((root / "data/psycon/stats.json").read_text())
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    fig.patch.set_facecolor("#f8fafc")
    fig.suptitle("PSYCON · Primary dataset", fontsize=20, weight="bold", color="#152d45")
    issues = dict(stats["dialogue_labels"]["issue"])
    missing_issues = stats["conversations"] - sum(issues.values())
    if missing_issues:
        issues["unspecified"] = missing_issues
    patient_count = sum(stats["turn_labels"]["sentiment"].values())
    politeness_count = sum(stats["turn_labels"]["politeness"].values())
    groups = [
        ("Conversations by issue", issues, "#246a85"),
        (
            f"Patient sentiment · {patient_count:,} turns",
            stats["turn_labels"]["sentiment"],
            "#367d6b",
        ),
        (
            f"Therapist politeness · {politeness_count:,} turns",
            stats["turn_labels"]["politeness"],
            "#4f65a4",
        ),
        ("Therapist interpersonal behavior", stats["turn_labels"]["ipc"], "#a46646"),
    ]
    for ax, (title, values, color) in zip(axes.flat, groups):
        items = sorted(values.items(), key=lambda item: item[1])
        names = [name.replace("_", " ").capitalize() for name, _ in items]
        names = [
            name.replace(
                "Disruptive behaviour and dissocial disorders", "Disruptive / dissocial"
            ).replace("Ptsd", "PTSD")
            for name in names
        ]
        counts = [count for _, count in items]
        bars = ax.barh(names, counts, color=color, height=0.62)
        ax.bar_label(bars, labels=[f"{count:,}" for count in counts], padding=5, fontsize=9)
        ax.set_title(title, loc="left", pad=12, color="#152d45")
        ax.set_xlim(0, max(counts) * 1.2)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.18)
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    path = root / "docs/assets/psycon-overview.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
