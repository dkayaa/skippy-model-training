"""Snip word spans from dataset text to emulate random mid-podcast windows."""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from dataset_io import load_dataset, write_dataset


def snip_text(text: str, start_words: int, end_words: int) -> str | None:
    words = text.split()
    if len(words) <= start_words + end_words:
        return None
    end_index = len(words) - end_words if end_words else len(words)
    return " ".join(words[start_words:end_index])


def snip_dataset(
    input_path: Path,
    output_path: Path,
    start_words: int,
    end_words: int,
    min_words: int,
) -> None:
    records = load_dataset(input_path)
    output_records = []
    skipped = 0

    for index, record in enumerate(records):
        snipped_text = snip_text(record["text"], start_words, end_words)
        if snipped_text is None or len(snipped_text.split()) < min_words:
            skipped += 1
            print(
                f"Skipped record {index}: too few words after snipping "
                f"({len(record['text'].split())} words)",
                file=sys.stderr,
            )
            continue

        output_record = dict(record)
        output_record["text"] = snipped_text
        output_records.append(output_record)

    write_dataset(output_path, output_records)

    print(
        f"Wrote {len(output_records)} records to {output_path} "
        f"(snipped first {start_words} and last {end_words} words; skipped {skipped})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Snip leading and trailing words from each dataset text field."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to input dataset (.json or .jsonl)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to output dataset (.json or .jsonl)",
    )
    parser.add_argument(
        "-x",
        "--snip-start",
        type=int,
        default=0,
        help="Number of words to remove from the start of each text (default: 0)",
    )
    parser.add_argument(
        "-y",
        "--snip-end",
        type=int,
        default=0,
        help="Number of words to remove from the end of each text (default: 0)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=1,
        help="Minimum words required in snipped text; shorter records are skipped (default: 1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    if args.snip_start < 0 or args.snip_end < 0:
        raise SystemExit("--snip-start and --snip-end must be non-negative")

    if args.min_words < 1:
        raise SystemExit("--min-words must be at least 1")

    if args.snip_start == 0 and args.snip_end == 0:
        raise SystemExit("At least one of --snip-start or --snip-end must be greater than 0")

    snip_dataset(
        input_path=args.input,
        output_path=args.output,
        start_words=args.snip_start,
        end_words=args.snip_end,
        min_words=args.min_words,
    )


if __name__ == "__main__":
    main()
