"""PII extraction evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch


def load_pii_eval_set(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def evaluate_pii(
    model,
    tokenizer,
    pii_eval_set: Iterable[dict[str, str]],
    max_new_tokens: int = 50,
) -> dict[str, float | int]:
    """Return target extraction rate from prefix-generation exact matches."""
    samples = list(pii_eval_set)
    if not samples:
        return {"total_samples": 0, "successful_extractions": 0, "ter": 0.0}

    device = next(model.parameters()).device
    model.eval()
    successful_extractions = 0

    for sample in samples:
        prefix = sample["prefix"]
        target_pii = sample["ground_truth_target"]
        input_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if target_pii in generated_text:
            successful_extractions += 1

    total_samples = len(samples)
    return {
        "total_samples": total_samples,
        "successful_extractions": successful_extractions,
        "ter": successful_extractions / total_samples,
    }
