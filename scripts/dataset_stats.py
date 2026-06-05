"""Print statistics for a labeled transcript dataset."""

import argparse
import json
import statistics
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from dataset_io import load_dataset


def word_count(text: str) -> int:
    return len(text.split())


def summarize_lengths(values: list[int]) -> dict:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}

    return {
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 1),
        "median": round(statistics.median(values), 1),
    }


def compute_stats(records: list[dict]) -> dict:
    label_counts: dict[int, int] = {}
    char_lengths: dict[int, list[int]] = {}
    word_lengths: dict[int, list[int]] = {}
    missing_text = 0
    missing_label = 0
    null_starts = 0
    present_starts = 0
    duplicate_texts = 0
    seen_texts: set[str] = set()

    for record in records:
        text = record.get("text")
        label = record.get("label")
        start = record.get("start")

        if text is None:
            missing_text += 1
            continue

        if label is None:
            missing_label += 1
            continue

        label_counts[label] = label_counts.get(label, 0) + 1
        char_lengths.setdefault(label, []).append(len(text))
        word_lengths.setdefault(label, []).append(word_count(text))

        if text in seen_texts:
            duplicate_texts += 1
        seen_texts.add(text)

        if start is None:
            null_starts += 1
        else:
            present_starts += 1

    total = len(records)
    positive = label_counts.get(1, 0)
    negative = label_counts.get(0, 0)
    labeled = positive + negative

    balance = None
    if positive and negative:
        balance = round(min(positive, negative) / max(positive, negative), 3)

    return {
        "total_records": total,
        "labeled_records": labeled,
        "positive_labels": positive,
        "negative_labels": negative,
        "other_labels": labeled - positive - negative,
        "positive_pct": round(100 * positive / labeled, 1) if labeled else 0.0,
        "negative_pct": round(100 * negative / labeled, 1) if labeled else 0.0,
        "class_balance_ratio": balance,
        "missing_text": missing_text,
        "missing_label": missing_label,
        "duplicate_texts": duplicate_texts,
        "null_starts": null_starts,
        "present_starts": present_starts,
        "char_lengths": {label: summarize_lengths(lengths) for label, lengths in char_lengths.items()},
        "word_lengths": {label: summarize_lengths(lengths) for label, lengths in word_lengths.items()},
        "unique_labels": sorted(label_counts.keys()),
    }


def print_report(path: Path, stats: dict) -> None:
    print(f"Dataset: {path}")
    print()
    print("Counts")
    print(f"  Total records:      {stats['total_records']}")
    print(f"  Labeled records:    {stats['labeled_records']}")
    print(f"  Positive (label 1): {stats['positive_labels']} ({stats['positive_pct']}%)")
    print(f"  Negative (label 0): {stats['negative_labels']} ({stats['negative_pct']}%)")

    if stats["other_labels"]:
        print(f"  Other labels:       {stats['other_labels']} ({stats['unique_labels']})")

    if stats["class_balance_ratio"] is not None:
        print(f"  Class balance:      {stats['class_balance_ratio']} (min/max)")

    print()
    print("Data quality")
    print(f"  Missing text:       {stats['missing_text']}")
    print(f"  Missing label:      {stats['missing_label']}")
    print(f"  Duplicate texts:    {stats['duplicate_texts']}")
    print(f"  Null start values:  {stats['null_starts']}")
    print(f"  Present start vals: {stats['present_starts']}")

    for label_name, label in [("Positive", 1), ("Negative", 0)]:
        if label not in stats["char_lengths"]:
            continue

        chars = stats["char_lengths"][label]
        words = stats["word_lengths"][label]
        print()
        print(f"Text length — {label_name} (label {label})")
        print(
            f"  Characters: min={chars['min']}, max={chars['max']}, "
            f"mean={chars['mean']}, median={chars['median']}"
        )
        print(
            f"  Words:      min={words['min']}, max={words['max']}, "
            f"mean={words['mean']}, median={words['median']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print statistics for a labeled dataset.")
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to input dataset (.json or .jsonl)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print stats as JSON instead of a text report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        raise SystemExit(1)

    records = load_dataset(args.input)
    stats = compute_stats(records)

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print_report(args.input, stats)


if __name__ == "__main__":
    main()
