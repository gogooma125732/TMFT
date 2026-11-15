"""Render paper-style README figures from the reported summary metrics.

Run: python scripts/render_readme_figures.py
The CSV values were transcribed from summary slides, not recomputed here.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from render_method_animatic import render_method_animatic

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.unicode_minus": False,
    "svg.fonttype": "none",  # Keep labels selectable and easy to edit.
})

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/data/summary_metrics.csv"
OUT = ROOT / "docs/figures"
INK = "#202020"
MUTED = "#505050"
RULE = "#767676"
GRID = "#e4e4e4"
BLUE = "#315e92"
RUST = "#af5a43"
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
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.04, facecolor=PAPER)
    # Matplotlib adds trailing spaces to multiline SVG paths.
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.04, facecolor=PAPER)
    plt.close(fig)


def tradeoff(rows: list[dict]) -> None:
    """Two matched panels keep close operating points legible at README width."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharex=True, sharey=True)
    fig.patch.set_facecolor(PAPER)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.19, wspace=0.14)
    offsets = {
        "Baseline": (9, 10), "RMFT-15": (9, -15),
        "TMFT-NER": (9, -14), "TMFT-MIA": (-9, 11), "Combined": (-9, -14),
        "RMFT-21.6": (9, -15), "NER-strict": (9, -13),
        "NER-moderate": (9, 11), "MIA-p90": (9, 11),
        "Combined-p90": (9, -14), "Comb-p90 cap65": (-9, 11),
    }
    for ax, group, label, color, marker in (
        (axes[0], "core", "(a) Core experiments", BLUE, "s"),
        (axes[1], "ablation", "(b) Ablation runs", RUST, "o"),
    ):
        subset = [row for row in rows if row["group"] == group]
        ax.set_facecolor(PAPER)
        ax.axhspan(1.0, 4.8, facecolor=color, alpha=0.065, zorder=0)
        ax.axhline(4.8, color=RULE, linestyle=(0, (4, 3)), linewidth=0.85, zorder=1)
        for spine in ax.spines.values():
            spine.set_color(RULE)
            spine.set_linewidth(0.65)
        ax.yaxis.grid(True, color=GRID, linewidth=0.55)
        ax.set_axisbelow(True)
        ax.scatter([r["delta_ppl"] for r in subset], [r["ter_pct"] for r in subset],
                   s=75, marker=marker, facecolor=color, edgecolor=PAPER,
                   linewidth=0.9, zorder=3)
        for row in subset:
            dx, dy = offsets[row["method"]]
            ax.annotate(row["method"], (row["delta_ppl"], row["ter_pct"]),
                        xytext=(dx, dy), textcoords="offset points",
                        ha="right" if dx < 0 else "left", va="center",
                        fontsize=10.8, color=INK,
                        weight="bold" if row["method"] in {"Baseline", "Combined-p90"} else "normal")
        ax.set_title(label, fontsize=13.1, weight="bold", color=color, loc="left", pad=10)
        ax.set_xlim(-0.12, 1.88)
        ax.set_ylim(1.0, 5.25)
        ax.set_xticks([0, 0.5, 1.0, 1.5])
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.tick_params(length=3, width=0.65, color=RULE, labelsize=10)
    fig.supxlabel("Increase in held-out perplexity vs. baseline (ΔPPL)", y=0.07, fontsize=11.5)
    axes[0].set_ylabel("Target extraction rate (TER, %)", labelpad=7, fontsize=11.5)
    save(fig, "privacy_utility_paper_v4")


def results_table(rows: list[dict]) -> None:
    """Recreate the six-row README table without changing its displayed values."""
    names = ["Baseline", "RMFT-15", "TMFT-NER", "TMFT-MIA", "Combined", "Combined-p90"]
    selected = {r["method"]: r for r in rows}
    fig, ax = plt.subplots(figsize=(7.8, 2.25))
    fig.patch.set_facecolor(PAPER)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0.025, 0.975)
    ax.set_ylim(0.11, 0.96)
    ax.axis("off")

    ax.plot([0.03, 0.97], [0.94, 0.94], color=INK, lw=0.95)

    columns = [
        (0.04, "Method", "left"),
        (0.38, "Exact extraction", "center"),
        (0.53, "TER (%) ↓", "center"),
        (0.67, "PPL ↓", "center"),
        (0.81, "ΔPPL ↓", "center"),
        (0.96, "Mask (%)", "right"),
    ]
    for x, label, ha in columns:
        ax.text(x, 0.86, label, ha=ha, va="center", fontsize=11.4, weight="bold", color=INK)
    ax.plot([0.03, 0.97], [0.80, 0.80], color=RULE, lw=0.65)

    y_positions = [0.735, 0.625, 0.515, 0.405, 0.295, 0.185]
    ax.plot([0.03, 0.97], [0.24, 0.24], color=RULE, lw=0.65)
    ax.add_patch(Rectangle((0.03, 0.135), 0.94, 0.10,
                           facecolor="#f1f1f1", edgecolor="none", zorder=-1))
    for name, y in zip(names, y_positions):
        row = selected[name]
        label = "Combined-p90 (ablation)" if name == "Combined-p90" else name
        values = [
            label,
            f'{row["exact_extractions"]}/500',
            f'{row["ter_pct"]:.1f}',
            f'{float(row["ppl"]):.3f}',
            "—" if name == "Baseline" else f'+{row["delta_ppl"]:.3f}',
            "0" if name == "Baseline" else f'{row["masked_pct"]:.2f}',
        ]
        for (x, _, ha), value in zip(columns, values):
            ax.text(x, y, value, ha=ha, va="center", fontsize=12.4,
                    weight="bold" if name == "Combined-p90" else "normal", color=INK)
    ax.plot([0.03, 0.97], [0.13, 0.13], color=INK, lw=0.95)
    save(fig, "summary_results_table_paper_v4")


if __name__ == "__main__":
    rows = load_rows()
    tradeoff(rows)
    results_table(rows)
    render_method_animatic()
    print(f"Wrote paper-style README figures to {OUT}")
