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
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.12, facecolor=PAPER)
    # Matplotlib adds trailing spaces to multiline SVG paths.
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.12, facecolor=PAPER)
    plt.close(fig)


def tradeoff(rows: list[dict]) -> None:
    """Two matched panels keep close operating points legible at README width."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.3), sharex=True, sharey=True)
    fig.patch.set_facecolor(PAPER)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.82, bottom=0.25, wspace=0.14)
    offsets = {
        "Baseline": (7, 9), "RMFT-15": (7, -13),
        "TMFT-NER": (7, -12), "TMFT-MIA": (-7, 9), "Combined": (-7, -12),
        "RMFT-21.6": (7, 9), "NER-strict": (7, -11),
        "NER-moderate": (7, 9), "MIA-p90": (7, 9),
        "Combined-p90": (7, -12), "Comb-p90 cap65": (-7, 9),
    }
    for ax, group, label, color, marker in (
        (axes[0], "core", "(a) Core experiments", BLUE, "s"),
        (axes[1], "ablation", "(b) Ablation runs", RUST, "o"),
    ):
        subset = [row for row in rows if row["group"] == group]
        ax.set_facecolor(PAPER)
        for spine in ax.spines.values():
            spine.set_color(RULE)
            spine.set_linewidth(0.65)
        ax.yaxis.grid(True, color=GRID, linewidth=0.55)
        ax.set_axisbelow(True)
        ax.scatter([r["delta_ppl"] for r in subset], [r["ter_pct"] for r in subset],
                   s=47, marker=marker, facecolor=PAPER, edgecolor=color,
                   linewidth=1.4, zorder=3)
        for row in subset:
            dx, dy = offsets[row["method"]]
            ax.annotate(row["method"], (row["delta_ppl"], row["ter_pct"]),
                        xytext=(dx, dy), textcoords="offset points",
                        ha="right" if dx < 0 else "left", va="center",
                        fontsize=8.1, color=INK)
        ax.set_title(label, fontsize=11.2, weight="bold", loc="left", pad=10)
        ax.set_xlim(-0.12, 1.88)
        ax.set_ylim(1.0, 5.25)
        ax.set_xticks([0, 0.5, 1.0, 1.5])
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.tick_params(length=3, width=0.65, color=RULE, labelsize=9)
    fig.supxlabel("Increase in held-out perplexity vs. baseline (ΔPPL)", y=0.175, fontsize=10)
    axes[0].set_ylabel("Target extraction rate (TER, %)", labelpad=7)
    fig.text(0.075, 0.105, "Figure 2: Privacy–utility operating points.",
             fontsize=10.8, weight="bold", color=INK)
    fig.text(0.075, 0.06,
             "Each point is one reported run. Both axes are matched; lower left is preferable. Error bars are unavailable.",
             fontsize=9.1, color=INK)
    save(fig, "privacy_utility_paper_v3")


def results_table(rows: list[dict]) -> None:
    """Recreate the six-row README table without changing its displayed values."""
    names = ["Baseline", "RMFT-15", "TMFT-NER", "TMFT-MIA", "Combined", "Combined-p90"]
    selected = {r["method"]: r for r in rows}
    fig, ax = plt.subplots(figsize=(9.5, 4.55))
    fig.patch.set_facecolor(PAPER)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.94, "Table 1: Privacy and utility across masking strategies.",
            fontsize=12, weight="bold", color=INK, va="top")
    ax.text(0.03, 0.87,
            "Core conditions and the lowest-TER ablation. Values match the reported summary results.",
            fontsize=9.4, color=MUTED, va="top")
    ax.plot([0.03, 0.97], [0.80, 0.80], color=INK, lw=0.95)

    columns = [
        (0.04, "Method", "left"),
        (0.37, "Exact extraction", "center"),
        (0.52, "TER (%) ↓", "center"),
        (0.66, "PPL ↓", "center"),
        (0.80, "ΔPPL ↓", "center"),
        (0.94, "Mask (%)", "right"),
    ]
    for x, label, ha in columns:
        ax.text(x, 0.745, label, ha=ha, va="center", fontsize=9.6, weight="bold", color=INK)
    ax.plot([0.03, 0.97], [0.698, 0.698], color=RULE, lw=0.65)

    y_positions = [0.635, 0.562, 0.489, 0.416, 0.343, 0.240]
    ax.plot([0.03, 0.97], [0.290, 0.290], color=RULE, lw=0.65)
    ax.add_patch(Rectangle((0.03, 0.196), 0.94, 0.088,
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
            ax.text(x, y, value, ha=ha, va="center", fontsize=10,
                    weight="bold" if name == "Combined-p90" else "normal", color=INK)
    ax.plot([0.03, 0.97], [0.190, 0.190], color=INK, lw=0.95)
    ax.text(0.03, 0.125,
            "TER: target extraction rate. ΔPPL: perplexity increase vs. the unrounded baseline (13.2256).",
            fontsize=9, color=MUTED, va="center")
    ax.text(0.03, 0.07,
            "The shaded row is a separate ablation; figures are reported observations, not confidence bounds.",
            fontsize=9, color=MUTED, va="center")
    save(fig, "summary_results_table_paper_v3")


if __name__ == "__main__":
    rows = load_rows()
    tradeoff(rows)
    results_table(rows)
    render_method_animatic()
    print(f"Wrote paper-style README figures to {OUT}")
