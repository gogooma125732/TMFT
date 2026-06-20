"""Membership inference attacks using loss and Min-K token likelihoods."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


def _collect_scores(model, tokenizer, dataset, text_column: str, max_seq_len: int, batch_size: int, min_k: int):
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

    loss_scores: list[float] = []
    min_k_scores: list[float] = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
        labels = input_ids[:, 1:]
        valid = attention_mask[:, 1:].bool()
        token_log_probs = -F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none"
        ).view_as(labels)
        for row, row_valid in zip(token_log_probs, valid):
            values = row[row_valid]
            if values.numel() == 0:
                continue
            loss_scores.append(float(values.mean().item()))
            k = max(1, int(np.ceil(values.numel() * min_k / 100)))
            min_k_scores.append(float(torch.topk(values, k=k, largest=False).values.mean().item()))
    return np.asarray(loss_scores), np.asarray(min_k_scores)


def evaluate_mia_auc(
    model,
    tokenizer,
    member_dataset,
    nonmember_dataset,
    text_column: str = "text",
    max_samples: int = 250,
    max_seq_len: int = 512,
    batch_size: int = 4,
    min_k: int = 20,
) -> dict[str, float | int]:
    """Return balanced member-vs-nonmember AUC for two standard attacks."""
    sample_count = min(max_samples, len(member_dataset), len(nonmember_dataset))
    if sample_count < 2:
        raise ValueError("MIA evaluation requires at least two member and two non-member samples.")
    members = member_dataset.select(range(sample_count))
    nonmembers = nonmember_dataset.select(range(sample_count))
    member_loss, member_min_k = _collect_scores(
        model, tokenizer, members, text_column, max_seq_len, batch_size, min_k
    )
    nonmember_loss, nonmember_min_k = _collect_scores(
        model, tokenizer, nonmembers, text_column, max_seq_len, batch_size, min_k
    )
    n = min(len(member_loss), len(nonmember_loss), len(member_min_k), len(nonmember_min_k))
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    loss_signal = np.concatenate([member_loss[:n], nonmember_loss[:n]])
    min_k_signal = np.concatenate([member_min_k[:n], nonmember_min_k[:n]])
    return {
        "mia_samples_per_class": int(n),
        "loss_mia_auc": float(roc_auc_score(labels, loss_signal)),
        "min_k_mia_auc": float(roc_auc_score(labels, min_k_signal)),
    }
