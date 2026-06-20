"""CLI entry point for TMFT experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from transformers import AutoModelForCausalLM

from src.evaluate_pii import evaluate_pii, load_pii_eval_set
from src.train import METHODS, load_config, load_tokenizer, train_model, upload_to_huggingface


def parse_args():
    parser = argparse.ArgumentParser(description="TMFT experiment orchestration")
    parser.add_argument("--mode", choices=["train", "eval", "upload", "all"], required=True)
    parser.add_argument("--method", choices=[*METHODS, "all"], default="tmft_combined")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--eval_path", default=None, help="JSON/JSONL with prefix and ground_truth_target")
    parser.add_argument("--model_dir", default=None, help="Model directory for eval/upload")
    parser.add_argument("--hf_repo_id", default=None, help="Hugging Face repo id, e.g. username/tmft-pythia")
    parser.add_argument("--public", action="store_true", help="Create/upload as a public HF repo")
    return parser.parse_args()


def selected_methods(method: str) -> list[str]:
    return list(METHODS) if method == "all" else [method]


def run_train(config: dict, method: str):
    results = {}
    for current_method in selected_methods(method):
        trainer, tokenizer, output_dir = train_model(config, method=current_method)
        results[current_method] = str(output_dir)
    return results


def run_eval(config: dict, model_dir: str | None, eval_path: str):
    model_path = model_dir or config.get("output_dir")
    if not model_path:
        raise ValueError("--model_dir or output_dir in config is required for eval")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = load_tokenizer(model_path)
    eval_set = load_pii_eval_set(eval_path)
    return evaluate_pii(model, tokenizer, eval_set)


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.mode in {"train", "all"}:
        outputs = run_train(config, args.method)
        print(json.dumps({"trained": outputs}, indent=2))

    if args.mode in {"eval", "all"}:
        if not args.eval_path:
            raise ValueError("--eval_path is required for eval")
        metrics = run_eval(config, args.model_dir, args.eval_path)
        print(json.dumps({"eval": metrics}, indent=2))

    if args.mode in {"upload", "all"}:
        if not args.hf_repo_id:
            raise ValueError("--hf_repo_id is required for upload")
        model_dir = args.model_dir or str(Path(config.get("output_dir", "./results")) / args.method)
        repo_id = upload_to_huggingface(model_dir, args.hf_repo_id, private=not args.public)
        print(json.dumps({"uploaded": repo_id, "model_dir": model_dir}, indent=2))


if __name__ == "__main__":
    main()
