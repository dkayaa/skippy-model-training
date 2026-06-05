# Skippy Model Training

Train a classifier that detects sponsor/ad segments in YouTube podcast transcripts.

## Overview

This repo covers the data pipeline and model training for Skippy. It fetches YouTube transcripts, weak-labels sponsor segments with keyword heuristics, builds a balanced training set, and fine-tunes a DistilBERT classifier.

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

Creates `data/labeled_samples_800.json` — 400 ad segments and 400 non-ad segments.

Check label distribution before training:

```bash
python scripts/sort_data_stats.py
```

### 3. Train classifier

```bash
python scripts/03_train_classifier.py
```

Fine-tunes DistilBERT for 3 epochs. Checkpoints go to `training-output/`; the final model and tokenizer are saved to `models/ad-classifier/`.

## Project structure

```
skippy-model-training/
├── scripts/
│   ├── 01_fetch_transcripts.py   # Fetch YouTube transcripts, weak-label segments
│   ├── 02_build_training_set.py  # Build balanced 800-sample training set
│   ├── 03_train_classifier.py    # Fine-tune DistilBERT
│   ├── sort_data_stats.py        # Print positive/negative label counts
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

## Notes

- Transcripts are fetched live from YouTube; availability depends on captions being enabled.
- Weak labels are noisy; the classifier learns from keyword-labeled examples.
- `scripts/back_translate.py` is experimental and not wired into the pipeline.
- Inference and the browser extension live in a separate repo.

## License

TBD
