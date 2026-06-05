"""Augment a labeled transcript dataset with synthetic entries via Ollama."""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from dataset_io import append_record, load_dataset, write_dataset

DEFAULT_OLLAMA_URL = "http://localhost:11434"


def build_prompt(text: str, label: int) -> str:
    category = "a sponsor or advertisement segment" if label == 1 else "regular podcast conversation (not an ad)"
    return f"""You are generating synthetic training data for a podcast transcript classifier.

The following text is from a podcast transcript segment labeled as {category}.

Write a new paraphrased version that:
- Stays in the same category ({category})
- Sounds like natural spoken podcast dialogue
- Is similar length to the original
- Does not add explanations, labels, or metadata

Original text:
{text}

Respond with only the paraphrased text."""


def generate_variant(
    text: str,
    label: int,
    model: str,
    ollama_url: str,
    timeout: int,
) -> str:
    response = requests.post(
        f"{ollama_url}/api/generate",
        json={
            "model": model,
            "prompt": build_prompt(text, label),
            "stream": False,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    generated = response.json().get("response", "").strip()
    if not generated:
        raise ValueError("Ollama returned an empty response")
    return generated


def sidecar_paths(output_path: Path) -> tuple[Path, Path]:
    synthetic_path = output_path.parent / f"{output_path.stem}.synthetic.jsonl"
    progress_path = output_path.parent / f"{output_path.stem}.progress.json"
    return synthetic_path, progress_path


def progress_key(index: int, variant_index: int) -> str:
    return f"{index}:{variant_index}"


def load_progress(progress_path: Path) -> dict | None:
    if not progress_path.exists():
        return None

    with open(progress_path, encoding="utf-8") as f:
        return json.load(f)


def save_progress(
    progress_path: Path,
    *,
    input_path: Path,
    multiplier: int,
    extend_existing: bool,
    completed: set[str],
) -> None:
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input": str(input_path.resolve()),
                "multiplier": multiplier,
                "extend_existing": extend_existing,
                "completed": sorted(completed),
            },
            f,
            indent=2,
        )


def validate_progress(
    progress: dict,
    *,
    input_path: Path,
    multiplier: int,
    extend_existing: bool,
) -> None:
    if progress["input"] != str(input_path.resolve()):
        raise ValueError("Progress file was created for a different input dataset")
    if progress["multiplier"] != multiplier:
        raise ValueError("Progress file was created with a different multiplier")
    if progress["extend_existing"] != extend_existing:
        raise ValueError("Progress file was created with a different extend-existing setting")


def finalize_output(
    records: list[dict],
    synthetic_path: Path,
    output_path: Path,
    extend_existing: bool,
) -> tuple[int, int]:
    synthetics = load_dataset(synthetic_path) if synthetic_path.exists() else []
    output_dataset = (records + synthetics) if extend_existing else synthetics
    write_dataset(output_path, output_dataset)
    return len(records), len(synthetics)


def cleanup_sidecars(synthetic_path: Path, progress_path: Path) -> None:
    synthetic_path.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)


def broaden_dataset(
    input_path: Path,
    output_path: Path,
    multiplier: int,
    model: str,
    extend_existing: bool,
    ollama_url: str,
    timeout: int,
    retries: int,
    fresh: bool,
) -> None:
    records = load_dataset(input_path)
    synthetic_path, progress_path = sidecar_paths(output_path)

    if fresh:
        cleanup_sidecars(synthetic_path, progress_path)

    progress = load_progress(progress_path)
    completed: set[str] = set()

    if progress is not None:
        validate_progress(
            progress,
            input_path=input_path,
            multiplier=multiplier,
            extend_existing=extend_existing,
        )
        completed = set(progress["completed"])
        print(f"Resuming with {len(completed)} synthetic records already written")

    total = len(records) * multiplier
    generated_this_run = 0

    for index, record in enumerate(records):
        for variant_index in range(multiplier):
            key = progress_key(index, variant_index)
            if key in completed:
                continue

            for attempt in range(1, retries + 1):
                try:
                    generated_text = generate_variant(
                        text=record["text"],
                        label=record["label"],
                        model=model,
                        ollama_url=ollama_url,
                        timeout=timeout,
                    )
                    synthetic_record = {
                        "text": generated_text,
                        "start": None,
                        "label": record["label"],
                    }
                    append_record(synthetic_path, synthetic_record)
                    completed.add(key)
                    save_progress(
                        progress_path,
                        input_path=input_path,
                        multiplier=multiplier,
                        extend_existing=extend_existing,
                        completed=completed,
                    )
                    generated_this_run += 1
                    print(
                        f"[{len(completed)}/{total}] record {index + 1}, "
                        f"variant {variant_index + 1}/{multiplier}"
                    )
                    break
                except (requests.RequestException, ValueError) as exc:
                    if attempt == retries:
                        print(
                            f"Failed record {index + 1}, variant {variant_index + 1} "
                            f"after {retries} attempts: {exc}",
                            file=sys.stderr,
                        )
                    else:
                        wait = attempt * 2
                        print(
                            f"Retrying record {index + 1}, variant {variant_index + 1} "
                            f"({attempt}/{retries}) in {wait}s: {exc}",
                            file=sys.stderr,
                        )
                        time.sleep(wait)

    original_count, synthetic_count = finalize_output(
        records,
        synthetic_path,
        output_path,
        extend_existing,
    )
    cleanup_sidecars(synthetic_path, progress_path)

    print(
        f"\nWrote {original_count + synthetic_count if extend_existing else synthetic_count} "
        f"records to {output_path} "
        f"({original_count} original, {synthetic_count} synthetic; "
        f"{generated_this_run} generated this run)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Broaden a labeled dataset with synthetic entries from Ollama."
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
        "--multiplier",
        type=int,
        default=5,
        help="Number of synthetic records to generate per input record (default: 5)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Target Ollama model name (e.g. llama3.2)",
    )
    parser.add_argument(
        "--extend-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include original records in the output dataset (default: true)",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per synthetic record on failure (default: 3)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard existing progress sidecars and start from scratch",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.multiplier < 1:
        raise SystemExit("--multiplier must be at least 1")

    try:
        broaden_dataset(
            input_path=args.input,
            output_path=args.output,
            multiplier=args.multiplier,
            model=args.model,
            extend_existing=args.extend_existing,
            ollama_url=args.ollama_url.rstrip("/"),
            timeout=args.timeout,
            retries=args.retries,
            fresh=args.fresh,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Use --fresh to discard existing progress and start over.", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
