import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLES_FILE = ROOT / "data" / "labeled_samples_800.jsonl"


def train_classifier(samples_file: Path) -> None:
    training_output_dir = str(ROOT / "training-output")
    logs_dir = str(ROOT / "logs")
    model_dir = ROOT / "models" / "ad-classifier"
    model_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("json", data_files=str(samples_file), split="train")
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
        logging_dir=logs_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
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
        default=DEFAULT_SAMPLES_FILE,
        help="Path to labeled dataset (.json or .jsonl)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    if args.input.suffix.lower() not in {".json", ".jsonl"}:
        raise SystemExit("Input must be a .json or .jsonl file")

    train_classifier(args.input)


if __name__ == "__main__":
    main()
