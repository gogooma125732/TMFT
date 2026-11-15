# Reported summary metrics

These values were transcribed from the project's `summary_slides.pdf` (PDF pages 6–7, core results and ablations). The slides are the source of the reported numbers; this repository does **not** contain the trained checkpoints or raw evaluation outputs needed to independently reproduce these exact rows. The figures in `docs/figures/` were redrawn from this CSV, rather than copied from the slides.

`ter_pct` is the percentage of successful exact normalized PII extractions over 500 attacks; `exact_extractions` is its integer numerator. `ser_pct` uses 454 seen targets. The attacks use a 50-token prefix and 50 greedy-generated tokens. `delta_ppl` is the reported increase in held-out perplexity relative to the unrounded baseline of 13.2256; `ppl` is rounded to three decimals in the slides. `masked_pct` is the fraction of masked training tokens, and `skips` counts skipped sample occurrences under the mask-ratio cap. AUC values closer to 0.5 indicate weaker membership inference signal.

The rows are discrete runs from a single model/dataset setup, not a statistical confidence interval or a continuous privacy–utility frontier. The ablation rows are reported comparisons and are not all exposed as one-command modes in the current CLI.
