"""Result-table and figure generation for TMFT experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save_figure(fig, output_dir: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_results(csv_path: str | Path, output_dir: str | Path = "results/figures") -> list[Path]:
    """Generate privacy-utility, MIA, and masking figures from main_results.csv."""
    frame = pd.read_csv(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(frame["mdp"], frame["ter"], s=70)
    for _, row in frame.iterrows():
        axis.annotate(row["method"], (row["mdp"], row["ter"]), xytext=(5, 5), textcoords="offset points")
    axis.set(xlabel="Delta perplexity vs baseline (lower is better)", ylabel="TER (lower is better)")
    axis.grid(alpha=0.25)
    _save_figure(fig, output_dir, "privacy_utility_tradeoff")

    fig, axis = plt.subplots(figsize=(7, 5))
    frame.plot(x="method", y=["loss_mia_auc", "min_k_mia_auc"], kind="bar", ax=axis)
    axis.axhline(0.5, linestyle="--", color="black", linewidth=1)
    axis.set(ylabel="MIA AUC", xlabel="", ylim=(0, 1))
    axis.tick_params(axis="x", rotation=25)
    _save_figure(fig, output_dir, "mia_auc")

    fig, axis = plt.subplots(figsize=(7, 5))
    frame.plot(x="method", y="masked_token_ratio", kind="bar", legend=False, ax=axis, color="#3b7a57")
    axis.axhline(0.15, linestyle="--", color="black", linewidth=1, label="RMFT target 0.15")
    axis.set(ylabel="Masked token ratio", xlabel="")
    axis.tick_params(axis="x", rotation=25)
    axis.legend()
    _save_figure(fig, output_dir, "masked_token_ratio")
    return [output_dir / name for name in ("privacy_utility_tradeoff.png", "mia_auc.png", "masked_token_ratio.png")]
