# TMFT: Targeted Masked Fine-Tuning

This project tests whether masking privacy-sensitive token losses during LoRA
fine-tuning reduces PII memorization with less utility degradation than random
masking. It is an empirical mitigation study, not differential privacy or
machine unlearning.

## Vessel Setup

Upload the entire `tmft_project/` directory and open a terminal in that
directory.

```bash
python -m pip install -U pip
python -m pip uninstall -y transformers peft accelerate tokenizers huggingface_hub datasets
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Restart the Jupyter kernel after installation. The tested compatibility stack
uses PyTorch 2.3.1, Transformers 4.41.2, PEFT 0.11.1, and Datasets 2.20.0.

## End-to-End Experiment

Prepare real PII-containing Enron splits and a real prefix-target evaluation
set:

```bash
python main.py --mode prepare --force_prepare
```

Train all conditions:

```bash
python main.py --mode train --method all
```

Evaluate TER, SER, held-out perplexity, MDP, Loss-MIA AUC, and Min-K MIA AUC:

```bash
python main.py --mode eval --method all
```

Generate result figures:

```bash
python main.py --mode plot
```

The full pipeline can be launched with:

```bash
python main.py --mode all --method all --force_prepare
```

For an interactive run, execute `tmft_experiment.ipynb` from top to bottom.

## Experimental Conditions

- `baseline`: standard LoRA fine-tuning
- `rmft`: random 15% loss masking
- `tmft_ner`: loss masking at spaCy plus regex PII spans
- `tmft_mia`: online token masking where the current model is more confident
  than the frozen base model
- `tmft_combined`: union of NER and post-warm-up MIA masks

## Outputs

- `data/processed/`: train, validation, and test DatasetDict
- `data/pii_eval.json`: automatically generated real PII prefix-target attacks
- `results/<method>/`: LoRA adapters and training metadata
- `results/tables/main_results.csv`: submission-ready numeric table
- `results/figures/`: PNG and PDF privacy/utility figures

Do not report results if preprocessing prints a synthetic fallback warning.
The final config disables fallback so an unavailable real dataset fails loudly.

## Hugging Face Upload

```bash
huggingface-cli login
python main.py --mode upload --method tmft_combined \
  --hf_repo_id YOUR_USERNAME/tmft-pythia-160m-tmft-combined
```

Use `--public` only after checking that the saved artifacts contain no raw PII.
