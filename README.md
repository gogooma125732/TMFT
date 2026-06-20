# TMFT Project

Targeted Masked Fine-Tuning experiments for reducing PII memorization in causal language models.

## Vessel / Jupyter Setup

Upload the entire `tmft_project/` directory to Vessel. The notebook is not
standalone; it imports code from `src/`, reads `configs/config.yaml`, and uses
`requirements.txt`.

```bash
cd /path/to/tmft_project
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

For Hugging Face upload:

```bash
huggingface-cli login
```

## Train

```bash
python main.py --mode train --method tmft_combined --config configs/config.yaml
```

Or open `tmft_experiment.ipynb` from inside the uploaded `tmft_project/`
directory and run the cells in order.

Available methods:

- `baseline`
- `rmft`
- `tmft_ner`
- `tmft_mia`
- `tmft_combined`
- `all`

## Evaluate PII Extraction

Prepare a JSON or JSONL file with:

```json
[
  {
    "prefix": "Email text prefix...",
    "ground_truth_target": "sensitive@example.com"
  }
]
```

Then run:

```bash
python main.py --mode eval --model_dir results/tmft_combined --eval_path data/pii_eval.json
```

## Upload To Hugging Face

```bash
python main.py --mode upload --method tmft_combined --model_dir results/tmft_combined --hf_repo_id YOUR_USERNAME/tmft-pythia-160m
```

Add `--public` only if you want the repository to be public.
