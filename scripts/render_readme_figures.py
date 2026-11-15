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
from matplotlib.patches import FancyArrowPatch, Rectangle

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
    fig, ax = plt.subplots(figsize=(9.2, 5.9))
    fig.patch.set_facecolor(PAPER)
    fig.subplots_adjust(left=0.105, right=0.975, top=0.80, bottom=0.27)
    ax.set_facecolor(PAPER)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color(RULE)
    ax.yaxis.grid(True, color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)

    for group, color, marker, label in (
        ("core", BLUE, "s", "Core experiments"),
        ("ablation", RUST, "o", "Ablations"),
    ):
        subset = [r for r in rows if r["group"] == group]
        ax.scatter(
            [r["delta_ppl"] for r in subset],
            [r["ter_pct"] for r in subset],
            s=40, marker=marker, facecolor=PAPER, edgecolor=color,
            linewidth=1.35, zorder=3, label=label,
        )

    offsets = {
        "Baseline": (6, 9), "RMFT-15": (6, -13), "RMFT-21.6": (6, 9),
        "NER-strict": (6, 8), "NER-moderate": (6, -12),
        "TMFT-NER": (6, -13), "MIA-p90": (6, 8),
        "Combined-p90": (6, -12), "Comb-p90 cap65": (6, 9),
        "TMFT-MIA": (-6, 9), "Combined": (-6, -13),
    }
    for row in rows:
        dx, dy = offsets[row["method"]]
        ax.annotate(
            row["method"], (row["delta_ppl"], row["ter_pct"]),
            xytext=(dx, dy), textcoords="offset points",
            ha="right" if dx < 0 else "left", va="center",
            fontsize=8.1, color=INK,
        )

    ax.set_xlim(-0.08, 1.83)
    ax.set_ylim(1.0, 5.25)
    ax.set_xticks([0, 0.5, 1.0, 1.5])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Increase in held-out perplexity vs. baseline (ΔPPL)", labelpad=7)
    ax.set_ylabel("Target extraction rate (TER, %)", labelpad=7)
    ax.tick_params(length=3, width=0.65, color=RULE, labelsize=9)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.11), ncol=2,
              frameon=False, handletextpad=0.3, columnspacing=2.2, fontsize=9)
    fig.text(0.105, 0.155, "Figure 2: Privacy–utility trade-off.", fontsize=10.8,
             weight="bold", color=INK)
    fig.text(0.105, 0.105,
             "Each marker is one reported run; lower TER and lower ΔPPL are preferable.\n"
             "Core and ablation settings are shown separately. Error bars are unavailable.",
             fontsize=9.3, color=INK, linespacing=1.2)
    save(fig, "privacy_utility_tradeoff")


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
    save(fig, "summary_results_table")


def overview() -> None:
    fig, ax = plt.subplots(figsize=(9.7, 2.75))
    fig.patch.set_facecolor(PAPER)
    ax.set_xlim(0, 9.7)
    ax.set_ylim(0, 2.75)
    ax.axis("off")
    boxes = [
        (0.1, "1. Prepare text", "Tokenize Enron emails\nand identify PII targets"),
        (2.55, "2. Find sensitive tokens", "NER + regex spans; optional\nonline MIA scores"),
        (5.0, "3. Mask their loss", "Set selected labels to −100\nunder a mask-ratio cap"),
        (7.45, "4. Evaluate", "Exact extraction, PPL,\nand MIA AUC"),
    ]
    for x, title, body in boxes:
        ax.add_patch(Rectangle((x, 0.78), 2.15, 1.25, fill=False,
                               edgecolor=RULE, linewidth=0.7))
        ax.text(x + 0.12, 1.76, title, fontsize=10.5, weight="bold", color=INK, va="top")
        ax.text(x + 0.12, 1.38, body, fontsize=9.1, color=INK,
                va="top", linespacing=1.3)
    for x in (2.30, 4.75, 7.20):
        ax.add_patch(FancyArrowPatch((x, 1.405), (x + 0.20, 1.405),
                                     arrowstyle="->", mutation_scale=10,
                                     linewidth=0.9, color=INK))
    ax.text(0.10, 0.50, "Figure 1: Targeted loss masking pipeline.",
            fontsize=10.5, weight="bold", color=INK)
    ax.text(0.10, 0.20,
            "Selected tokens remain in the input; their labels are omitted from the training loss.",
            fontsize=9.2, color=MUTED)
    save(fig, "method_overview")


if __name__ == "__main__":
    rows = load_rows()
    tradeoff(rows)
    results_table(rows)
    overview()
    print(f"Wrote paper-style README figures to {OUT}")
