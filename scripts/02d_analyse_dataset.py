"""Analyse labeled datasets from a folder of JSON/JSONL files."""

import argparse
import logging
import statistics
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from dataset_io import load_dataset
from transformers import DistilBertTokenizerFast

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = ROOT / "data"
DEFAULT_MODEL_DIR = ROOT / "models" / "ad-classifier"
MAX_SEQUENCE_LENGTH = 512


def collect_dataset_files(input_dir: Path) -> list[Path]:
    paths = [
        *input_dir.glob("*.json"),
        *input_dir.glob("*.jsonl"),
    ]
    return sorted(
        (path for path in paths if not path.name.endswith(".progress.json")),
        key=lambda path: path.name,
    )


def load_records(input_dir: Path) -> list[dict]:
    records = []
    for path in collect_dataset_files(input_dir):
        records.extend(load_dataset(path))
    return records


def word_count(text: str) -> int:
    return len(text.split())


def resolve_tokenizer_path(tokenizer_path: Path | None) -> str | Path:
    if tokenizer_path is not None:
        return tokenizer_path
    if DEFAULT_MODEL_DIR.is_dir() and (DEFAULT_MODEL_DIR / "tokenizer.json").exists():
        return DEFAULT_MODEL_DIR
    return "distilbert-base-uncased"


def token_count(tokenizer: DistilBertTokenizerFast, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True))


def analyse_dataset(input_dir: Path, tokenizer: DistilBertTokenizerFast) -> dict:
    records = load_records(input_dir)
    if not records:
        raise ValueError(f"No records found in {input_dir}")

    positive = 0
    negative = 0
    word_lengths: list[int] = []
    char_lengths: list[int] = []
    token_lengths: list[int] = []

    for record in records:
        label = int(record["label"])
        if label == 1:
            positive += 1
        elif label == 0:
            negative += 1

        text = record["text"]
        word_lengths.append(word_count(text))
        char_lengths.append(len(text))
        token_lengths.append(token_count(tokenizer, text))

    over_max_tokens = sum(1 for length in token_lengths if length > MAX_SEQUENCE_LENGTH)

    return {
        "files": [path.name for path in collect_dataset_files(input_dir)],
        "total_records": len(records),
        "positive_labels": positive,
        "negative_labels": negative,
        "avg_length_words": round(statistics.mean(word_lengths), 1),
        "min_length_words": min(word_lengths),
        "max_length_words": max(word_lengths),
        "avg_length_chars": round(statistics.mean(char_lengths), 1),
        "min_length_chars": min(char_lengths),
        "max_length_chars": max(char_lengths),
        "avg_length_tokens": round(statistics.mean(token_lengths), 1),
        "min_length_tokens": min(token_lengths),
        "max_length_tokens": max(token_lengths),
        "over_max_tokens": over_max_tokens,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
    }


def print_report(input_dir: Path, stats: dict) -> None:
    print(f"Dataset folder: {input_dir}")
    print(f"Files: {', '.join(stats['files'])}")
    print()
    print(f"Total records:  {stats['total_records']}")
    print(f"Positive (+1):  {stats['positive_labels']}")
    print(f"Negative (0):   {stats['negative_labels']}")
    print()
    print("Input context length (words)")
    print(f"  Average: {stats['avg_length_words']}")
    print(f"  Min:     {stats['min_length_words']}")
    print(f"  Max:     {stats['max_length_words']}")
    print()
    print("Input context length (characters)")
    print(f"  Average: {stats['avg_length_chars']}")
    print(f"  Min:     {stats['min_length_chars']}")
    print(f"  Max:     {stats['max_length_chars']}")
    print()
    print(f"Input context length (tokens, max {stats['max_sequence_length']})")
    print(f"  Average: {stats['avg_length_tokens']}")
    print(f"  Min:     {stats['min_length_tokens']}")
    print(f"  Max:     {stats['max_length_tokens']}")
    print(f"  Over max (truncated in training): {stats['over_max_tokens']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse labeled datasets from a folder of .json or .jsonl files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing labeled dataset files (.json or .jsonl)",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Tokenizer folder or model name (default: models/ad-classifier, else distilbert-base-uncased)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.is_dir():
        raise SystemExit(f"Input folder not found: {args.input}")

    if not collect_dataset_files(args.input):
        raise SystemExit(f"No .json or .jsonl files found in {args.input}")

    tokenizer_source = resolve_tokenizer_path(args.tokenizer)
    transformers_logger = logging.getLogger("transformers")
    previous_level = transformers_logger.level
    transformers_logger.setLevel(logging.ERROR)
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_source)
        stats = analyse_dataset(args.input, tokenizer)
    finally:
        transformers_logger.setLevel(previous_level)
    print_report(args.input, stats)


if __name__ == "__main__":
    main()
