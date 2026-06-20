"""End-to-end CLI for reproducible TMFT experiments."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import pandas as pd
import torch
from datasets import load_from_disk

from src.data_prep import prepare_experiment_data
from src.evaluate_mia import evaluate_mia_auc
from src.evaluate_pii import evaluate_pii, load_pii_eval_set
from src.evaluate_ppl import evaluate_perplexity
from src.plot_results import plot_results
from src.train import (
    METHODS,
    load_config,
    load_tokenizer,
    load_trained_model,
    train_model,
    upload_to_huggingface,
)


def parse_args():
    parser = argparse.ArgumentParser(description="TMFT experiment orchestration")
    parser.add_argument("--mode", choices=["prepare", "train", "eval", "plot", "upload", "all"], required=True)
    parser.add_argument("--method", choices=[*METHODS, "all"], default="all")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--force_prepare", action="store_true")
    parser.add_argument("--model_dir", default=None, help="Override model directory for single-method eval/upload")
    parser.add_argument("--hf_repo_id", default=None)
    parser.add_argument("--public", action="store_true")
    return parser.parse_args()


def selected_methods(method: str) -> list[str]:
    return list(METHODS) if method == "all" else [method]


def ensure_prepared(config: dict, force: bool = False):
    splits, eval_path = prepare_experiment_data(config, force=force)
    config["text_column"] = "text"
    print(
        json.dumps(
            {"train": len(splits["train"]), "validation": len(splits["validation"]), "test": len(splits["test"]),
             "pii_eval_path": str(eval_path)},
            indent=2,
        )
    )
    return splits, eval_path


def run_train(config: dict, method: str, splits) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for current_method in selected_methods(method):
        print(f"\n===== TRAIN: {current_method} =====")
        _, _, output_dir = train_model(
            config,
            method=current_method,
            train_dataset=splits["train"],
            eval_dataset=splits["validation"],
        )
        outputs[current_method] = str(output_dir)
    return outputs


def _model_directory(config: dict, method: str, override: str | None) -> Path:
    return Path(override) if override else Path(config.get("output_dir", "results")) / method


def run_eval(config: dict, method: str, splits, eval_path: Path, model_dir: str | None = None) -> pd.DataFrame:
    eval_set = load_pii_eval_set(eval_path)
    rows: list[dict[str, object]] = []
    for current_method in selected_methods(method):
        current_dir = _model_directory(config, current_method, model_dir if method != "all" else None)
        if not current_dir.exists():
            raise FileNotFoundError(f"Missing trained model for {current_method}: {current_dir}")
        print(f"\n===== EVAL: {current_method} =====")
        tokenizer = load_tokenizer(str(current_dir))
        model = load_trained_model(current_dir)
        if torch.cuda.is_available():
            model = model.cuda()

        pii = evaluate_pii(
            model,
            tokenizer,
            eval_set,
            max_new_tokens=int(config.get("eval_max_new_tokens", 50)),
        )
        ppl = evaluate_perplexity(
            model,
            tokenizer,
            splits["validation"],
            max_seq_len=int(config.get("max_seq_len", 512)),
            batch_size=int(config.get("eval_batch_size", 4)),
        )
        mia = evaluate_mia_auc(
            model,
            tokenizer,
            splits["train"],
            splits["test"],
            max_samples=int(config.get("mia_eval_samples", 250)),
            max_seq_len=int(config.get("max_seq_len", 512)),
            batch_size=int(config.get("eval_batch_size", 4)),
            min_k=int(config.get("min_k_percent", 20)),
        )
        metadata_path = current_dir / "training_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        rows.append(
            {
                "method": current_method,
                "ter": pii["ter"],
                "ser": pii["ser"],
                "ppl": ppl["ppl"],
                "loss_mia_auc": mia["loss_mia_auc"],
                "min_k_mia_auc": mia["min_k_mia_auc"],
                "masked_token_ratio": metadata.get("masked_token_ratio", 0.0),
                "skipped_samples": metadata.get("skipped_samples", 0),
                "pii_eval_samples": pii["total_samples"],
                "mia_samples_per_class": mia["mia_samples_per_class"],
            }
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    if "baseline" in set(frame["method"]):
        baseline_ppl = float(frame.loc[frame["method"] == "baseline", "ppl"].iloc[0])
        frame["mdp"] = frame["ppl"] - baseline_ppl
    else:
        frame["mdp"] = float("nan")
    tables_dir = Path(config.get("results_table_dir", "results/tables"))
    tables_dir.mkdir(parents=True, exist_ok=True)
    output_path = tables_dir / "main_results.csv"
    frame.to_csv(output_path, index=False)
    print(f"Saved results: {output_path}")
    return frame


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.mode == "prepare":
        ensure_prepared(config, force=args.force_prepare)
        return

    if args.mode == "plot":
        csv_path = Path(config.get("results_table_dir", "results/tables")) / "main_results.csv"
        print([str(path) for path in plot_results(csv_path)])
        return

    if args.mode == "upload":
        if not args.hf_repo_id or args.method == "all":
            raise ValueError("Upload requires --hf_repo_id and one specific --method")
        directory = _model_directory(config, args.method, args.model_dir)
        upload_to_huggingface(directory, args.hf_repo_id, private=not args.public)
        print(json.dumps({"uploaded": args.hf_repo_id, "model_dir": str(directory)}, indent=2))
        return

    splits, eval_path = ensure_prepared(config, force=args.force_prepare)
    if args.mode in {"train", "all"}:
        print(json.dumps({"trained": run_train(config, args.method, splits)}, indent=2))
    if args.mode in {"eval", "all"}:
        frame = run_eval(config, args.method, splits, eval_path, args.model_dir)
        print(frame.to_string(index=False))
    if args.mode == "all":
        csv_path = Path(config.get("results_table_dir", "results/tables")) / "main_results.csv"
        print([str(path) for path in plot_results(csv_path)])


if __name__ == "__main__":
    main()
