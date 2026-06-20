"""Training utilities for TMFT experiments."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

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
    model_kwargs = {}
    if config.get("fp16") and torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
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
    return get_peft_model(model, peft_config)


def load_text_dataset(config: dict[str, Any]) -> Dataset:
    dataset_name = config.get("dataset_name", "argilla/enron-email-dataset")
    dataset_split = config.get("dataset_split", "train")
    text_column = config.get("text_column", "text")
    max_train_samples = config.get("max_train_samples")

    dataset = load_dataset(dataset_name, split=dataset_split)
    if text_column not in dataset.column_names:
        raise ValueError(f"Missing text column '{text_column}'. Available: {dataset.column_names}")

    dataset = dataset.filter(lambda row: bool(row.get(text_column)))
    if max_train_samples:
        dataset = dataset.select(range(min(int(max_train_samples), len(dataset))))
    return dataset


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
    seed: int = 42

    def __post_init__(self):
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

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

        if self.method in {"rmft", "tmft_combined"}:
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
                mask[row_idx] |= ner_mask(offsets, text, self.nlp_model, ner_labels)

        if self.method in {"tmft_mia", "tmft_combined"}:
            if self.target_model is None or self.reference_model is None:
                raise ValueError("target_model and reference_model are required for MIA masking")
            mask |= mia_mask(
                encoded["input_ids"],
                encoded["attention_mask"],
                self.target_model,
                self.reference_model,
                threshold_percentile=self.threshold_percentile,
            )

        labels[encoded["attention_mask"] == 0] = -100
        labels[mask] = -100
        encoded["labels"] = labels
        return encoded


def train_model(
    config: dict[str, Any],
    method: str = "tmft_combined",
    train_dataset: Dataset | None = None,
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
        report_to=config.get("report_to", "none"),
        remove_unused_columns=False,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return trainer, tokenizer, output_dir


def upload_to_huggingface(output_dir: str | Path, repo_id: str, private: bool = True):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(output_dir), repo_id=repo_id)
    return repo_id
