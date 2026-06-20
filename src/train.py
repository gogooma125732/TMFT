"""Training utilities for TMFT experiments."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
import json

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch
import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import AutoPeftModelForCausalLM

from .masking import DEFAULT_NER_LABELS, mia_mask, ner_mask, random_mask


METHODS = ("baseline", "rmft", "tmft_ner", "tmft_mia", "tmft_combined")


def load_config(path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_causal_lm(model_name: str, use_lora: bool, config: dict[str, Any]):
    # Keep trainable weights in FP32 and let Trainer/Accelerate handle AMP when
    # fp16=True. Loading the model directly in FP16 can make LoRA gradients FP16,
    # which triggers "Attempting to unscale FP16 gradients" in torch GradScaler.
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if not use_lora:
        return model

    peft_config = LoraConfig(
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        target_modules=list(config["target_modules"]),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    return model


def load_text_dataset(config: dict[str, Any]) -> Dataset:
    dataset_name = config.get("dataset_name", "cc0de/Enron_email")
    dataset_split = config.get("dataset_split", "train")
    text_column = config.get("text_column")
    fallback_text_column = text_column or "text"
    max_train_samples = config.get("max_train_samples")
    local_data_path = config.get("local_data_path")

    if local_data_path:
        dataset = load_local_text_dataset(local_data_path, text_column=fallback_text_column)
    elif dataset_name == "synthetic_enron_pii":
        dataset = build_synthetic_email_dataset(size=int(max_train_samples or 200), text_column=fallback_text_column)
    else:
        try:
            dataset = load_dataset(dataset_name, split=dataset_split)
        except Exception as exc:
            if not config.get("allow_synthetic_fallback", True):
                raise
            print(
                f"Warning: could not load dataset '{dataset_name}' ({type(exc).__name__}: {exc}). "
                "Falling back to synthetic_enron_pii. Use config['local_data_path'] for a real Enron file."
            )
            dataset = build_synthetic_email_dataset(size=int(max_train_samples or 200), text_column=fallback_text_column)
    if not text_column:
        candidates = ("text", "message", "body", "content", "email", "mail", "Message")
        text_column = next((column for column in candidates if column in dataset.column_names), None)
        if not text_column:
            text_column = dataset.column_names[0]
        config["text_column"] = text_column
        print(f"Using text column: {text_column}")
    if text_column not in dataset.column_names:
        raise ValueError(f"Missing text column '{text_column}'. Available: {dataset.column_names}")

    dataset = dataset.filter(lambda row: bool(row.get(text_column)))
    if max_train_samples:
        dataset = dataset.select(range(min(int(max_train_samples), len(dataset))))
    return dataset


def load_local_text_dataset(path: str | Path, text_column: str = "text") -> Dataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"local_data_path does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        dataset = load_dataset("csv", data_files=str(path), split="train")
    elif suffix in {".json", ".jsonl"}:
        dataset = load_dataset("json", data_files=str(path), split="train")
    elif suffix in {".txt", ".text"}:
        lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
        dataset = Dataset.from_list([{text_column: line} for line in lines if line])
    else:
        raise ValueError("local_data_path must be .csv, .json, .jsonl, or .txt")

    if text_column not in dataset.column_names:
        first_col = dataset.column_names[0]
        dataset = dataset.rename_column(first_col, text_column)
    return dataset


def build_synthetic_email_dataset(size: int = 200, text_column: str = "text") -> Dataset:
    names = ["John Smith", "Mary Johnson", "Kenneth Lay", "Jeff Skilling", "Lisa Taylor"]
    orgs = ["Enron", "Dynegy", "Portland General", "Houston Trading Desk", "Risk Management"]
    domains = ["enron.com", "example.com", "corp-mail.com"]
    cities = ["Houston", "Portland", "New York", "San Francisco", "Calgary"]
    templates = []
    for i in range(max(size, 1)):
        name = names[i % len(names)]
        org = orgs[i % len(orgs)]
        city = cities[i % len(cities)]
        email = f"{name.lower().replace(' ', '.')}{i}@{domains[i % len(domains)]}"
        phone = f"713-555-{1000 + (i % 9000)}"
        date = f"2001-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}"
        templates.append(
            {
                text_column: (
                    f"Subject: meeting follow up\n"
                    f"Hi team, please contact {name} at {email} or {phone} before {date}. "
                    f"{name} is coordinating the {org} review in {city}. "
                    f"Please keep the trading desk summary confidential until the next update."
                )
            }
        )
    return Dataset.from_list(templates)


def load_spacy_model(model_name: str = "en_core_web_sm"):
    import spacy

    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. Run: python -m spacy download {model_name}"
        ) from exc


@dataclass
class TMFTDataCollator:
    tokenizer: Any
    method: str
    max_seq_len: int
    text_column: str = "text"
    nlp_model: Any | None = None
    reference_model: Any | None = None
    target_model: Any | None = None
    ner_labels: set[str] | None = None
    rmft_probability: float = 0.15
    threshold_percentile: float = 75
    mia_warmup_batches: int = 50
    max_mask_ratio: float = 0.5
    seed: int = 42

    def __post_init__(self):
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)
        self.batches_seen = 0
        self.masked_tokens = 0
        self.valid_tokens = 0
        self.skipped_samples = 0

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [row[self.text_column] for row in features]
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offset_mapping = encoded.pop("offset_mapping")
        labels = encoded["input_ids"].clone()
        mask = torch.zeros_like(labels, dtype=torch.bool)

        if self.method == "rmft":
            mask |= random_mask(
                encoded["attention_mask"],
                probability=self.rmft_probability,
                generator=self.generator,
            )

        if self.method in {"tmft_ner", "tmft_combined"}:
            if self.nlp_model is None:
                raise ValueError("nlp_model is required for NER masking")
            ner_labels = self.ner_labels or DEFAULT_NER_LABELS
            for row_idx, text in enumerate(texts):
                offsets = [tuple(pair.tolist()) for pair in offset_mapping[row_idx]]
                max_seen_char = max((end for _, end in offsets), default=0)
                text_for_ner = text[:max_seen_char]
                mask[row_idx] |= ner_mask(offsets, text_for_ner, self.nlp_model, ner_labels)

        use_mia = self.method in {"tmft_mia", "tmft_combined"} and self.batches_seen >= self.mia_warmup_batches
        if use_mia:
            if self.target_model is None or self.reference_model is None:
                raise ValueError("target_model and reference_model are required for MIA masking")
            mask |= mia_mask(
                encoded["input_ids"],
                encoded["attention_mask"],
                self.target_model,
                self.reference_model,
                threshold_percentile=self.threshold_percentile,
            )

        valid_counts = encoded["attention_mask"].sum(dim=1).clamp_min(1)
        mask_ratios = mask.sum(dim=1) / valid_counts
        keep = mask_ratios <= self.max_mask_ratio
        if not keep.any():
            keep[torch.argmin(mask_ratios)] = True
        self.skipped_samples += int((~keep).sum().item())
        if not keep.all():
            encoded = {key: value[keep] for key, value in encoded.items()}
            labels = labels[keep]
            mask = mask[keep]

        self.masked_tokens += int(mask.sum().item())
        self.valid_tokens += int(encoded["attention_mask"].sum().item())
        self.batches_seen += 1

        labels[encoded["attention_mask"] == 0] = -100
        labels[mask] = -100
        encoded["labels"] = labels
        return encoded

    def summary(self) -> dict[str, float | int]:
        return {
            "batches_seen": self.batches_seen,
            "masked_token_ratio": self.masked_tokens / max(self.valid_tokens, 1),
            "skipped_samples": self.skipped_samples,
        }


def train_model(
    config: dict[str, Any],
    method: str = "tmft_combined",
    train_dataset: Dataset | None = None,
    eval_dataset: Dataset | None = None,
):
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose one of: {METHODS}")

    set_seed(int(config.get("seed", 42)))
    model_name = config.get("model_name", "EleutherAI/pythia-160m")
    tokenizer = load_tokenizer(model_name)
    model = load_causal_lm(model_name, use_lora=bool(config.get("use_lora", True)), config=config)
    train_dataset = train_dataset or load_text_dataset(config)

    nlp_model = None
    if method in {"tmft_ner", "tmft_combined"}:
        nlp_model = load_spacy_model(config.get("spacy_model", "en_core_web_sm"))

    reference_model = None
    if method in {"tmft_mia", "tmft_combined"}:
        reference_model = AutoModelForCausalLM.from_pretrained(model_name)
        reference_model.requires_grad_(False)

    collator = TMFTDataCollator(
        tokenizer=tokenizer,
        method=method,
        max_seq_len=int(config.get("max_seq_len", 512)),
        text_column=config.get("text_column", "text"),
        nlp_model=nlp_model,
        reference_model=reference_model,
        target_model=model,
        ner_labels=set(config.get("ner_labels", DEFAULT_NER_LABELS)),
        rmft_probability=float(config.get("rmft_probability", 0.15)),
        threshold_percentile=float(config.get("threshold_percentile", 75)),
        mia_warmup_batches=int(config.get("mia_warmup_batches", 50)),
        max_mask_ratio=float(config.get("max_mask_ratio", 0.5)),
        seed=int(config.get("seed", 42)),
    )

    output_dir = Path(config.get("output_dir", "./results")) / method
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(config.get("num_epochs", 1)),
        per_device_train_batch_size=int(config.get("batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        learning_rate=float(config.get("learning_rate", 2e-4)),
        warmup_steps=int(config.get("warmup_steps", 0)),
        weight_decay=float(config.get("weight_decay", 0.0)),
        fp16=bool(config.get("fp16", False)) and torch.cuda.is_available(),
        logging_steps=int(config.get("logging_steps", 10)),
        save_steps=int(config.get("save_steps", 500)),
        save_total_limit=int(config.get("save_total_limit", 2)),
        evaluation_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy=config.get("save_strategy", "epoch"),
        report_to=config.get("report_to", "none"),
        remove_unused_columns=False,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metadata = {
        "method": method,
        "model_name": model_name,
        "train_samples": len(train_dataset),
        "validation_samples": len(eval_dataset) if eval_dataset is not None else 0,
        **collator.summary(),
        "log_history": trainer.state.log_history,
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return trainer, tokenizer, output_dir


def load_trained_model(model_dir: str | Path):
    """Load either a saved LoRA adapter or a complete causal language model."""
    model_dir = Path(model_dir)
    if (model_dir / "adapter_config.json").exists():
        return AutoPeftModelForCausalLM.from_pretrained(str(model_dir))
    return AutoModelForCausalLM.from_pretrained(str(model_dir))


def upload_to_huggingface(output_dir: str | Path, repo_id: str, private: bool = True):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(output_dir), repo_id=repo_id)
    return repo_id
