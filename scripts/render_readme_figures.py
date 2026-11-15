"""Redraw README figures from the reported summary metrics.

Run from anywhere: python scripts/render_readme_figures.py
The data are transcribed from the project's summary slides, not recomputed here.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["svg.fonttype"] = "none"  # Keep labels selectable and SVGs easy to edit.


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/data/summary_metrics.csv"
OUT = ROOT / "docs/figures"
BLUE = "#2462b9"
NAVY = "#17324d"
GRAY = "#738497"
ORANGE = "#d77943"
PAPER = "#ffffff"


def load_rows() -> list[dict]:
    with DATA.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for field in ("ter_pct", "delta_ppl", "masked_pct"):
            row[field] = float(row[field])
    assert len(rows) == 11
    assert next(r for r in rows if r["method"] == "Combined-p90")["ter_pct"] == 1.4
    return rows


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    svg = OUT / f"{name}.svg"
    fig.savefig(svg, bbox_inches="tight", facecolor=PAPER)
    # Matplotlib puts trailing spaces on multiline SVG paths; keep Git diffs clean.
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def tradeoff(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 6.1), layout="constrained")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b6c3cf")
    ax.grid(color="#dce5ed", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    for group, color, label in (
        ("core", BLUE, "Core experiments"),
        ("ablation", ORANGE, "Ablations"),
    ):
        subset = [r for r in rows if r["group"] == group]
        ax.scatter(
            [r["delta_ppl"] for r in subset],
            [r["ter_pct"] for r in subset],
            s=[105 if r["method"] in {"Baseline", "TMFT-NER", "Combined-p90"} else 66 for r in subset],
            facecolor=color,
            edgecolor="white",
            linewidth=1.4,
            zorder=3,
            label=label,
        )

    # Keep the labels at the data points; small, deliberate offsets avoid overlap.
    offsets = {
        "Baseline": (8, 10), "RMFT-15": (8, -13), "RMFT-21.6": (8, 11),
        "NER-strict": (8, 8), "NER-moderate": (8, -11),
        "TMFT-NER": (8, -13), "MIA-p90": (8, 10),
        "Combined-p90": (8, -14), "Comb-p90 cap65": (8, 9),
        "TMFT-MIA": (-9, 10), "Combined": (-9, -13),
    }
    for row in rows:
        dx, dy = offsets[row["method"]]
        align = "right" if dx < 0 else "left"
        ax.annotate(
            row["method"], (row["delta_ppl"], row["ter_pct"]),
            xytext=(dx, dy), textcoords="offset points", ha=align, va="center",
            fontsize=8.3, color=NAVY,
        )

    ax.set_xlim(-0.08, 1.83)
    ax.set_ylim(1.0, 5.25)
    ax.set_xlabel("Increase in held-out perplexity vs. baseline (ΔPPL)", fontsize=10.5, color=NAVY, labelpad=12)
    ax.set_ylabel("Target extraction rate (TER, %)", fontsize=10.5, color=NAVY, labelpad=12)
    ax.tick_params(colors="#526476", labelsize=9)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.set_title("Privacy–utility operating points", loc="left", fontsize=16, weight="bold", color=NAVY, pad=17)
    ax.text(0.01, -0.19, "Lower TER and lower ΔPPL are preferable. Each dot is one reported run.",
            transform=ax.transAxes, fontsize=9, color="#526476")
    save(fig, "privacy_utility_tradeoff")


def overview() -> None:
    fig, ax = plt.subplots(figsize=(11.3, 3.45), layout="constrained")
    fig.patch.set_facecolor(PAPER)
    ax.set_xlim(0, 11.3)
    ax.set_ylim(0, 3.45)
    ax.axis("off")
    ax.text(0.15, 3.08, "How targeted loss masking works", color=NAVY, fontsize=16, weight="bold")
    boxes = [
        (0.15, "01", "Prepare text", "Tokenize Enron emails\nand identify PII targets"),
        (2.95, "02", "Find sensitive tokens", "NER + regex spans; optional\nonline MIA scores"),
        (5.75, "03", "Mask their loss", "Set selected labels to −100\nsubject to a mask-ratio cap"),
        (8.55, "04", "Evaluate", "Exact extraction, PPL,\nand MIA AUC"),
    ]
    for x, number, title, body in boxes:
        rect = FancyBboxPatch((x, 0.62), 2.5, 1.88, boxstyle="round,pad=0.04,rounding_size=0.13",
                              linewidth=1.2, edgecolor="#cfdae6", facecolor="#f7faff")
        ax.add_patch(rect)
        ax.text(x + 0.2, 2.13, number, fontsize=10, weight="bold", color=BLUE)
        ax.text(x + 0.2, 1.73, title, fontsize=11.1, weight="bold", color=NAVY)
        ax.text(x + 0.2, 1.28, body, fontsize=9.2, color="#526476", va="top", linespacing=1.45)
    for x in (2.69, 5.49, 8.29):
        ax.add_patch(FancyArrowPatch((x, 1.55), (x + 0.20, 1.55), arrowstyle="-|>",
                                     mutation_scale=13, linewidth=1.3, color=BLUE))
    ax.text(0.18, 0.18, "Masking removes selected tokens from the training loss; it does not remove them from the input.",
            fontsize=9, color="#526476")
    save(fig, "method_overview")


if __name__ == "__main__":
    tradeoff(load_rows())
    overview()
    print(f"Wrote README figures to {OUT}")
