import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from dataset_io import load_dataset as load_labeled_records
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = ROOT / "data"


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def training_args_for_device(device: str) -> dict:
    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    elif device == "cpu":
        return {"use_cpu": True}
    return {}


def setup_logging(logs_dir: Path) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(logs_dir / "train.log")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    root.addHandler(stdout_handler)
    root.addHandler(file_handler)


def collect_dataset_files(input_dir: Path) -> list[Path]:
    return sorted(
        [*input_dir.glob("*.json"), *input_dir.glob("*.jsonl")],
        key=lambda path: path.name,
    )


def load_training_dataset(input_dir: Path) -> Dataset:
    records = []
    for path in collect_dataset_files(input_dir):
        for record in load_labeled_records(path):
            records.append({"text": record["text"], "label": int(record["label"])})

    if not records:
        raise ValueError(f"No training records found in {input_dir}")

    return Dataset.from_list(records)


def train_classifier(input_dir: Path) -> None:
    device = resolve_device()
    print(f"Training on {device}")

    training_output_dir = str(ROOT / "training-output")
    logs_dir = ROOT / "logs"
    setup_logging(logs_dir)

    model_dir = ROOT / "models" / "ad-classifier"
    model_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_training_dataset(input_dir)
    dataset = dataset.train_test_split(test_size=0.2)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
    )

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary"
        )
        acc = accuracy_score(labels, predictions)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    training_args = TrainingArguments(
        output_dir=training_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=str(logs_dir),
        logging_strategy="epoch",
        log_level="info",
        #report_to="tensorboard",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        **training_args_for_device(device),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Saved model to {model_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on a labeled dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing labeled dataset files (.json or .jsonl)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.is_dir():
        raise SystemExit(f"Input folder not found: {args.input}")

    if not collect_dataset_files(args.input):
        raise SystemExit(f"No .json or .jsonl files found in {args.input}")

    train_classifier(args.input)


if __name__ == "__main__":
    main()
