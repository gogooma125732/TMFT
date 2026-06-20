"""Mask builders for targeted masked fine-tuning."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_NER_LABELS = {"PERSON", "ORG", "GPE", "LOC", "EMAIL", "PHONE", "DATE"}


def char_span_to_token_indices(
    offset_mapping: Sequence[tuple[int, int]],
    char_start: int,
    char_end: int,
) -> list[int]:
    """Return token indices whose character offsets overlap a character span."""
    token_indices: list[int] = []
    for idx, (token_start, token_end) in enumerate(offset_mapping):
        if token_start == token_end:
            continue
        if token_start < char_end and token_end > char_start:
            token_indices.append(idx)
    return token_indices


def ner_mask(
    offset_mapping: Sequence[tuple[int, int]],
    text: str,
    nlp_model,
    ner_labels: Iterable[str] | None = None,
) -> torch.Tensor:
    """Build a boolean token mask from spaCy NER spans."""
    labels = set(ner_labels or DEFAULT_NER_LABELS)
    mask = torch.zeros(len(offset_mapping), dtype=torch.bool)
    doc = nlp_model(text)
    for ent in doc.ents:
        if ent.label_ not in labels:
            continue
        token_indices = char_span_to_token_indices(
            offset_mapping,
            ent.start_char,
            ent.end_char,
        )
        if token_indices:
            mask[token_indices] = True
    return mask


def random_mask(
    attention_mask: torch.Tensor,
    probability: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Mask a random subset of non-padding tokens."""
    if probability <= 0:
        return torch.zeros_like(attention_mask, dtype=torch.bool)
    probs = torch.rand(
        attention_mask.shape,
        generator=generator,
        device=attention_mask.device,
    )
    return (probs < probability) & attention_mask.bool()


def _token_nll(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(input_ids.size(0), -1)
    padded = torch.zeros_like(input_ids, dtype=losses.dtype)
    padded[:, 1:] = losses
    return padded


def mia_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_model,
    reference_model,
    threshold_percentile: float = 75,
) -> torch.Tensor:
    """Build a mask from a simple token-level membership-risk score."""
    device = next(target_model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    reference_model = reference_model.to(device)
    target_was_training = target_model.training
    target_model.eval()
    reference_model.eval()

    with torch.no_grad():
        target_loss = _token_nll(target_model, input_ids, attention_mask)
        ref_loss = _token_nll(reference_model, input_ids, attention_mask)
        score = ref_loss / (target_loss + 1e-8)
        valid_scores = score[attention_mask.bool()]
        if valid_scores.numel() == 0:
            if target_was_training:
                target_model.train()
            return torch.zeros_like(attention_mask, dtype=torch.bool)
        threshold = np.percentile(valid_scores.detach().float().cpu().numpy(), threshold_percentile)
        mask = score > threshold
    if target_was_training:
        target_model.train()
    return (mask & attention_mask.bool()).cpu()
