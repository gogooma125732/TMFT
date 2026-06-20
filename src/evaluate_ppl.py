"""Held-out perplexity evaluation for causal language models."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def evaluate_perplexity(
    model,
    tokenizer,
    dataset,
    text_column: str = "text",
    max_seq_len: int = 512,
    batch_size: int = 4,
) -> dict[str, float | int]:
    """Compute token-weighted NLL and perplexity on a held-out split."""
    device = next(model.parameters()).device
    model.eval()

    def collate(rows: list[dict[str, Any]]):
        return tokenizer(
            [row[text_column] for row in rows],
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )

    total_nll = 0.0
    total_tokens = 0
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        valid = attention_mask[:, 1:].bool()
        losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view_as(shift_labels)
        total_nll += float(losses[valid].sum().item())
        total_tokens += int(valid.sum().item())
    mean_nll = total_nll / max(total_tokens, 1)
    return {"nll": mean_nll, "ppl": math.exp(min(mean_nll, 20.0)), "ppl_tokens": total_tokens}
