import argparse
import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from dataset_io import write_dataset

random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS_DIR = ROOT / "transcripts"
DATA_DIR = ROOT / "data"


def build_training_set(output_path: Path, n_samples: int) -> None:
    records = []

    for file_path in TRANSCRIPTS_DIR.glob("*.json"):
        try:
            with open(file_path, encoding="utf-8") as f:
                records.append(json.load(f))
        except Exception as exc:
            print(exc)

    samples_positive = []
    samples_negative = []

    for record in records:
        for sample in record:
            if sample["label"] == 1:
                samples_positive.append(sample)
            else:
                samples_negative.append(sample)

    random.shuffle(samples_positive)
    random.shuffle(samples_negative)

    samples = (
        samples_positive[: n_samples // 2]
        + samples_negative[: n_samples // 2]
    )
    random.shuffle(samples)

    write_dataset(output_path, samples)

    print(
        f"""
SUMMARY STATS
Positive Labels: {len(samples_positive)}
Negative Labels: {len(samples_negative)}
Wrote {len(samples)} samples to {output_path}
"""
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a balanced labeled training set.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "labeled_samples_800.jsonl",
        help="Output dataset path (.json or .jsonl)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=800,
        help="Total number of samples to write (default: 800)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_training_set(args.output, args.n_samples)


if __name__ == "__main__":
    main()
