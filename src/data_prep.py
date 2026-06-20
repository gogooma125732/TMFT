"""Reproducible Enron preprocessing and PII evaluation-set construction."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re
import shutil
from typing import Any, Iterable

import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm.auto import tqdm

from .train import load_spacy_model, load_text_dataset, load_tokenizer


EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]\d{4}(?!\d)")
SPACY_PII_LABELS = {"PERSON", "ORG", "GPE", "LOC", "DATE"}


def clean_email_text(text: Any) -> str:
    """Normalize an email and remove a conventional RFC-like header block."""
    text = str(text or "").replace("\x00", " ").replace("\r\n", "\n").strip()
    head, separator, body = text.partition("\n\n")
    header_names = ("from:", "to:", "subject:", "date:", "message-id:", "mime-version:")
    header_lines = [line.strip().lower() for line in head.splitlines()[:30]]
    if separator and sum(any(line.startswith(name) for name in header_names) for line in header_lines) >= 2:
        text = body.strip()
    return re.sub(r"[ \t]+", " ", text)


def truncate_to_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """Return the raw-text prefix represented by at most ``max_tokens`` tokens."""
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
        return_offsets_mapping=True,
    )
    offsets = encoded.get("offset_mapping", [])
    return text[: offsets[-1][1]] if offsets else ""


def detect_pii_spans(text: str, nlp_model=None, doc=None) -> list[dict[str, Any]]:
    """Detect deduplicated PII spans with spaCy plus email/phone regexes."""
    if doc is None:
        if nlp_model is None:
            raise ValueError("nlp_model or precomputed doc is required")
        doc = nlp_model(text)
    spans: list[dict[str, Any]] = []
    for entity in doc.ents:
        if entity.label_ in SPACY_PII_LABELS:
            spans.append(
                {"start": entity.start_char, "end": entity.end_char, "type": entity.label_, "value": entity.text}
            )
    for pattern, label in ((EMAIL_RE, "EMAIL"), (PHONE_RE, "PHONE")):
        for match in pattern.finditer(text):
            spans.append({"start": match.start(), "end": match.end(), "type": label, "value": match.group(0)})

    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for span in sorted(spans, key=lambda item: (item["start"], -(item["end"] - item["start"]))):
        key = (int(span["start"]), int(span["end"]), str(span["type"]))
        if key not in seen and str(span["value"]).strip():
            seen.add(key)
            deduplicated.append(span)
    return deduplicated


def _split_indices(rows: list[dict[str, Any]], seed: int) -> tuple[list[int], list[int], list[int]]:
    """Create deterministic 80/10/10 splits, approximately stratified by email PII."""
    rng = np.random.default_rng(seed)
    partitions = {True: [], False: []}
    for index, row in enumerate(rows):
        partitions[bool(row["has_email"])].append(index)

    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for indices in partitions.values():
        rng.shuffle(indices)
        n_items = len(indices)
        n_train = int(n_items * 0.8)
        n_validation = int(n_items * 0.1)
        train.extend(indices[:n_train])
        validation.extend(indices[n_train : n_train + n_validation])
        test.extend(indices[n_train + n_validation :])
    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    return train, validation, test


def _records(dataset: Dataset, indices: Iterable[int]) -> list[dict[str, Any]]:
    return [dataset[int(index)] for index in indices]


def build_pii_eval_records(
    test_dataset: Dataset,
    tokenizer,
    seen_pii: set[str],
    max_samples: int = 500,
    prefix_tokens: int = 50,
    source_split: str = "test",
) -> list[dict[str, Any]]:
    """Build prefix-completion attacks from real PII spans in held-out emails."""
    records: list[dict[str, Any]] = []
    for row in test_dataset:
        text = row["text"]
        spans = row["pii_spans"]
        if not spans:
            continue
        selected = None
        for span in spans:
            target = str(span["value"]).strip()
            before = text[: int(span["start"])]
            token_ids = tokenizer(before, add_special_tokens=False).input_ids[-prefix_tokens:]
            prefix = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            if prefix and target:
                selected = (span, target, prefix)
                break
        if selected is None:
            continue
        span, target, prefix = selected
        normalized = target.casefold().strip()
        records.append(
            {
                "prefix": prefix,
                "ground_truth_target": target,
                "pii_type": str(span["type"]),
                "seen_in_train": normalized in seen_pii,
                "source_split": source_split,
            }
        )
        if len(records) >= max_samples:
            break
    return records


def prepare_experiment_data(config: dict[str, Any], force: bool = False) -> tuple[DatasetDict, Path]:
    """Prepare datasets and write a non-placeholder PII extraction evaluation file."""
    prepared_dir = Path(config.get("prepared_data_dir", "data/processed"))
    eval_path = Path(config.get("pii_eval_path", "data/pii_eval.json"))
    if prepared_dir.exists() and eval_path.exists() and not force:
        config["text_column"] = "text"
        return load_from_disk(str(prepared_dir)), eval_path
    if force and prepared_dir.exists():
        shutil.rmtree(prepared_dir)

    tokenizer = load_tokenizer(config.get("model_name", "EleutherAI/pythia-160m"))
    nlp_model = load_spacy_model(config.get("spacy_model", "en_core_web_sm"))
    source = load_text_dataset(config)
    text_column = config["text_column"]
    max_tokens = int(config.get("max_seq_len", 512))
    max_prepared = int(config.get("max_prepared_samples", 6000))

    candidates: list[tuple[int, str]] = []
    seen_text_hashes: set[str] = set()
    print(f"Preprocessing {len(source):,} source emails on CPU...")
    for source_index, row in enumerate(tqdm(source, total=len(source), desc="Clean/tokenize")):
        text = truncate_to_tokens(clean_email_text(row.get(text_column)), tokenizer, max_tokens)
        if not text:
            continue
        text_hash = hashlib.sha1(" ".join(text.casefold().split()).encode("utf-8")).hexdigest()
        if text_hash in seen_text_hashes:
            continue
        seen_text_hashes.add(text_hash)
        candidates.append((source_index, text))

    disabled_pipes = [name for name in nlp_model.pipe_names if name not in {"tok2vec", "ner"}]
    docs = nlp_model.pipe(
        (text for _, text in candidates),
        batch_size=int(config.get("spacy_batch_size", 32)),
        disable=disabled_pipes,
    )
    rows: list[dict[str, Any]] = []
    iterator = zip(candidates, docs)
    for (source_index, text), doc in tqdm(iterator, total=len(candidates), desc="PII detection"):
        spans = detect_pii_spans(text, doc=doc)
        if not spans:
            continue
        rows.append(
            {
                "sample_id": int(source_index),
                "text": text,
                "pii_spans": spans,
                "pii_values": [str(span["value"]) for span in spans],
                "has_email": any(span["type"] == "EMAIL" for span in spans),
            }
        )
        if len(rows) >= max_prepared:
            break
    print(f"Prepared {len(rows):,} unique PII-containing emails.")
    if len(rows) < 10:
        raise ValueError(f"Only {len(rows)} PII-containing samples were prepared; at least 10 are required.")

    all_dataset = Dataset.from_list(rows)
    train_idx, validation_idx, test_idx = _split_indices(rows, int(config.get("seed", 42)))
    splits = DatasetDict(
        {
            "train": Dataset.from_list(_records(all_dataset, train_idx)),
            "validation": Dataset.from_list(_records(all_dataset, validation_idx)),
            "test": Dataset.from_list(_records(all_dataset, test_idx)),
        }
    )
    prepared_dir.parent.mkdir(parents=True, exist_ok=True)
    splits.save_to_disk(str(prepared_dir))

    seen_pii = {
        value.casefold().strip()
        for row in splits["train"]
        for value in row["pii_values"]
        if value and value.strip()
    }
    max_eval_samples = int(config.get("max_eval_samples", 500))
    seen_eval_records = build_pii_eval_records(
        splits["train"],
        tokenizer,
        seen_pii,
        max_samples=max_eval_samples // 2,
        prefix_tokens=int(config.get("eval_prefix_tokens", 50)),
        source_split="train",
    )
    unseen_eval_records = build_pii_eval_records(
        splits["test"],
        tokenizer,
        seen_pii,
        max_samples=max_eval_samples - len(seen_eval_records),
        prefix_tokens=int(config.get("eval_prefix_tokens", 50)),
        source_split="test",
    )
    eval_records = seen_eval_records + unseen_eval_records
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps(eval_records, indent=2, ensure_ascii=False), encoding="utf-8")
    if not eval_records:
        raise ValueError("No PII extraction records could be created from the test split.")
    config["text_column"] = "text"
    return splits, eval_path
