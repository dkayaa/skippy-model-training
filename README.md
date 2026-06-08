# Skipr Model Training

Train a classifier that detects sponsor/ad segments in YouTube podcast transcripts.

## Overview

This repo covers the data pipeline and model training for Skipr. It fetches YouTube transcripts, weak-labels sponsor segments with keyword heuristics, builds a balanced training set, and fine-tunes a DistilBERT classifier.

**Pipeline**

1. **Fetch transcripts** — download captions and weak-label segments with keyword heuristics
2. **Build training set** — sample 400 positive + 400 negative windows
3. **Train** — fine-tune `distilbert-base-uncased` for binary classification

## Requirements

- Python 3.10+
- ~2 GB disk for dependencies; ~300 MB for trained model weights

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Model weights

The trained model is not in the repo (`models/` is gitignored, ~255 MB).

Either:

- **Train locally** — run `scripts/03_train_classifier.py` (see below)
- **Download separately** — publish to Hugging Face and document the model URL here

## Usage

Run all scripts from the repo root:

```bash
python scripts/01_fetch_transcripts.py
python scripts/02_build_training_set.py
python scripts/03_train_classifier.py
```

### 1. Fetch and label transcripts

```bash
python scripts/01_fetch_transcripts.py
```

Fetches ~100 podcast YouTube videos, segments transcripts (window=20, stride=10), and weak-labels with sponsor keywords (e.g. "promo code", "brought to you by", brand names). Output goes to `transcripts/` (gitignored).

### 2. Build training set

```bash
python scripts/02_build_training_set.py
```

Creates `data/labeled_samples_800.jsonl` by default — 400 ad segments and 400 non-ad segments. Output format is chosen from the file extension (`.json` or `.jsonl`).

```bash
python scripts/02_build_training_set.py --output data/labeled_samples_800.jsonl
```

### 2c. Snip dataset (optional)

Trim leading and trailing words from each record to emulate a random mid-podcast span:

```bash
python scripts/02c_snip_dataset.py \
  --input data/labeled_samples_800.jsonl \
  --output data/labeled_samples_800_snipped.jsonl \
  --snip-start 10 \
  --snip-end 10
```

`-x` / `--snip-start` removes the first *x* words; `-y` / `--snip-end` removes the last *y* words. Records too short after snipping are skipped.

### 2b. Broaden dataset (optional)

Requires a running [Ollama](https://ollama.com) service.

```bash
python scripts/02b_broaden_dataset.py \
  --input data/labeled_samples_800.jsonl \
  --output data/labeled_samples_4800.jsonl \
  --multiplier 5 \
  --model llama3.2
```

Generates synthetic variants for each record while preserving the label. Each variant rotates through augmentation strategies (paraphrase, new scenario, style shift, fragment, vocabulary shift) with label-specific prompts and per-strategy temperature. Near-duplicates are filtered via `--dedup-threshold` (default `0.85`; set `1.0` to disable). Each successful generation is appended to a sidecar file immediately, so progress survives crashes. Re-run the same command to resume; use `--fresh` to start over. Use `--no-extend-existing` to write only the synthetic records. Defaults: `--multiplier 5`, `--extend-existing`.

Check dataset statistics before training:

```bash
python scripts/dataset_stats.py --input data/labeled_samples_800.jsonl
```

### 3. Train classifier

```bash
python scripts/03_train_classifier.py --input data/labeled_samples_800.jsonl
```

Fine-tunes DistilBERT for 3 epochs. Checkpoints go to `training-output/`; the final model and tokenizer are saved to `models/ad-classifier/`.

## Project structure

```
skipr-model-training/
├── scripts/
│   ├── 01_fetch_transcripts.py   # Fetch YouTube transcripts, weak-label segments
│   ├── 02_build_training_set.py  # Build balanced 800-sample training set
│   ├── 02b_broaden_dataset.py    # Augment dataset via Ollama
│   ├── 02c_snip_dataset.py       # Trim word spans from each record
│   ├── 03_train_classifier.py    # Fine-tune DistilBERT
│   ├── dataset_io.py             # Shared JSON/JSONL read/write helpers
│   ├── dataset_stats.py          # Print dataset statistics
│   ├── sort_data_stats.py        # Stats over raw transcripts/ folder
│   └── back_translate.py         # Experimental augmentation (not in pipeline)
├── data/
│   └── labeled_samples_800.json  # Training dataset (committed)
├── models/                       # Saved model, gitignored
├── transcripts/                  # Generated, gitignored
├── training-output/              # Training checkpoints, gitignored
└── logs/                         # Training logs, gitignored
```

## Model details

- **Base model:** `distilbert-base-uncased`
- **Task:** Binary sequence classification (ad segment vs not)
- **Segmentation:** 20 caption snippets per window, stride 10
- **Weak labels:** Keyword/brand matching in `01_fetch_transcripts.py`

## Dataset format

Labeled datasets use records with `text`, `label`, and optionally `start` fields. Pipeline scripts accept both:

- **JSON** — a single array of records
- **JSONL** — one record per line

Format is inferred from the file extension. JSONL is the default for new outputs.

## Notes

- Transcripts are fetched live from YouTube; availability depends on captions being enabled.
- Weak labels are noisy; the classifier learns from keyword-labeled examples.
- `scripts/back_translate.py` is experimental and not wired into the pipeline.
- Inference and the browser extension live in separate repos (`skipr-plugin`, `skipr-youtube-api`).

## License

TBD
