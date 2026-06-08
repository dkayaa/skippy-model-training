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

# Rotates by variant index: (strategy_name, temperature)
STRATEGIES: list[tuple[str, float]] = [
    ("paraphrase", 0.7),
    ("new_scenario", 1.0),
    ("style_shift", 0.9),
    ("fragment", 0.8),
    ("vocabulary_shift", 0.85),
]

OUTPUT_INSTRUCTION = """Respond with only the generated transcript text.
Do not include introductions, explanations, labels, quotes, markdown, or any text before or after the transcript."""


def strategy_for_variant(variant_index: int) -> tuple[str, float]:
    return STRATEGIES[variant_index % len(STRATEGIES)]


def build_prompt(text: str, label: int, strategy: str) -> str:
    if label == 1:
        return _build_ad_prompt(text, strategy)
    return _build_non_ad_prompt(text, strategy)


def _build_ad_prompt(text: str, strategy: str) -> str:
    if strategy == "paraphrase":
        instruction = """Write a paraphrased sponsor-read that:
- Keeps the same general structure (setup, pitch, call-to-action)
- Uses different wording and a different fictional or real-sounding brand name
- Sounds like natural spoken podcast dialogue"""

    elif strategy == "new_scenario":
        instruction = """Write a completely NEW sponsor-read segment (do not closely copy the original) that:
- Uses a different sponsor category than the original (e.g. VPN, supplements, meal kit, betting, SaaS, clothing, mattress)
- Includes a URL or promo-code style call-to-action
- Sounds like a mid-podcast advertisement read"""

    elif strategy == "style_shift":
        instruction = """Write a sponsor-read on a similar topic but in a noticeably different delivery style than the original (e.g. rushed, overly enthusiastic, deadpan, smooth radio announcer, casual bro-talk):
- Change tone and pacing words ("uh", "look", "honestly", "here's the deal")
- Keep it clearly an advertisement"""

    elif strategy == "fragment":
        instruction = """Write a SHORT excerpt from a sponsor-read (about half the original length) that:
- Contains either the opening hook plus CTA, OR the core product pitch
- Still clearly sounds like an advertisement segment"""

    else:  # vocabulary_shift
        instruction = """Write a paraphrased sponsor-read that:
- Preserves meaning and ad intent
- Uses substantially different vocabulary and sentence structure than the original
- Does not reuse long phrases from the original verbatim"""

    return f"""You are generating synthetic training data for a podcast ad-segment classifier.

{instruction}
- Similar length to the original (except fragment strategy)
- No explanations, labels, or metadata

Original text:
{text}

{OUTPUT_INSTRUCTION}"""


def _build_non_ad_prompt(text: str, strategy: str) -> str:
    if strategy == "paraphrase":
        instruction = """Write a paraphrased version of this regular podcast conversation that:
- Keeps the same topic and meaning
- Sounds like natural spoken dialogue
- Contains NO advertisements, sponsor mentions, promo codes, or URLs"""

    elif strategy == "new_scenario":
        instruction = """Write a completely NEW regular podcast conversation segment (do not closely copy the original) that:
- Covers a different topic (e.g. personal story, hot take, guest banter, sports, politics, comedy bit)
- Sounds like mid-episode podcast dialogue
- Contains NO advertisements, sponsor mentions, promo codes, or URLs"""

    elif strategy == "style_shift":
        instruction = """Write a conversation segment on a similar theme but in a noticeably different speaking style (e.g. more excited, more deadpan, more argumentative, storytelling mode):
- Must remain regular podcast conversation, NOT an ad
- No sponsor language, promo codes, or URLs"""

    elif strategy == "fragment":
        instruction = """Write a SHORT excerpt of regular podcast conversation (about half the original length) that:
- Sounds like a natural mid-episode clip
- Is NOT an advertisement
- Contains no promo codes or sponsor CTAs"""

    else:  # vocabulary_shift
        instruction = """Write a paraphrased conversation segment that:
- Preserves meaning but uses substantially different vocabulary and sentence structure
- Does not reuse long phrases from the original verbatim
- Remains non-ad conversation with no sponsor language"""

    return f"""You are generating synthetic training data for a podcast ad-segment classifier.

{instruction}
- Similar length to the original (except fragment strategy)
- No explanations, labels, or metadata

Original text:
{text}

{OUTPUT_INSTRUCTION}"""


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def word_shingles(text: str, n: int = 3) -> set[str]:
    words = normalize_text(text).split()
    if not words:
        return set()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def max_similarity(text: str, corpus: list[str]) -> float:
    shingles = word_shingles(text)
    return max((jaccard_similarity(shingles, word_shingles(other)) for other in corpus), default=0.0)


def deduplicate_records(
    records: list[dict],
    reference_texts: list[str],
    threshold: float,
) -> tuple[list[dict], int]:
    kept: list[dict] = []
    seen_texts = list(reference_texts)
    dropped = 0

    for record in records:
        text = record["text"]
        if max_similarity(text, seen_texts) >= threshold:
            dropped += 1
            continue
        kept.append(record)
        seen_texts.append(text)

    return kept, dropped


def generate_variant(
    text: str,
    label: int,
    variant_index: int,
    model: str,
    ollama_url: str,
    timeout: int,
    temperature_boost: float = 0.0,
) -> tuple[str, str]:
    strategy, temperature = strategy_for_variant(variant_index)
    response = requests.post(
        f"{ollama_url}/api/generate",
        json={
            "model": model,
            "prompt": build_prompt(text, label, strategy),
            "stream": False,
            "options": {
                "temperature": min(temperature + temperature_boost, 1.2),
            },
        },
        timeout=timeout,
    )
    response.raise_for_status()
    generated = response.json().get("response", "").strip()
    if not generated:
        raise ValueError("Ollama returned an empty response")
    return generated, strategy


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
    dedup_threshold: float | None,
    completed: set[str],
) -> None:
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input": str(input_path.resolve()),
                "multiplier": multiplier,
                "extend_existing": extend_existing,
                "dedup_threshold": dedup_threshold,
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
    dedup_threshold: float | None,
) -> None:
    if progress["input"] != str(input_path.resolve()):
        raise ValueError("Progress file was created for a different input dataset")
    if progress["multiplier"] != multiplier:
        raise ValueError("Progress file was created with a different multiplier")
    if progress["extend_existing"] != extend_existing:
        raise ValueError("Progress file was created with a different extend-existing setting")
    if progress.get("dedup_threshold") != dedup_threshold:
        raise ValueError("Progress file was created with a different dedup threshold")


def load_existing_texts(synthetic_path: Path, source_texts: list[str]) -> list[str]:
    texts = list(source_texts)
    if synthetic_path.exists():
        for record in load_dataset(synthetic_path):
            texts.append(record["text"])
    return texts


def finalize_output(
    records: list[dict],
    synthetic_path: Path,
    output_path: Path,
    extend_existing: bool,
    dedup_threshold: float | None,
) -> tuple[int, int, int]:
    synthetics = load_dataset(synthetic_path) if synthetic_path.exists() else []
    dropped = 0

    if dedup_threshold is not None:
        reference_texts = [record["text"] for record in records] if extend_existing else []
        synthetics, dropped = deduplicate_records(synthetics, reference_texts, dedup_threshold)

    output_dataset = (records + synthetics) if extend_existing else synthetics
    write_dataset(output_path, output_dataset)
    return len(records), len(synthetics), dropped


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
    dedup_threshold: float | None,
) -> None:
    records = load_dataset(input_path)
    synthetic_path, progress_path = sidecar_paths(output_path)
    source_texts = [record["text"] for record in records]

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
            dedup_threshold=dedup_threshold,
        )
        completed = set(progress["completed"])
        print(f"Resuming with {len(completed)} synthetic records already written")

    total = len(records) * multiplier
    generated_this_run = 0
    skipped_similar = 0

    for index, record in enumerate(records):
        for variant_index in range(multiplier):
            key = progress_key(index, variant_index)
            if key in completed:
                continue

            strategy, _ = strategy_for_variant(variant_index)

            for attempt in range(1, retries + 1):
                try:
                    temperature_boost = 0.1 * (attempt - 1)
                    generated_text, strategy = generate_variant(
                        text=record["text"],
                        label=record["label"],
                        variant_index=variant_index,
                        model=model,
                        ollama_url=ollama_url,
                        timeout=timeout,
                        temperature_boost=temperature_boost,
                    )

                    if dedup_threshold is not None:
                        corpus = load_existing_texts(synthetic_path, source_texts)
                        similarity = max_similarity(generated_text, corpus)
                        if similarity >= dedup_threshold:
                            skipped_similar += 1
                            if attempt == retries:
                                print(
                                    f"Skipped record {index + 1}, variant {variant_index + 1} "
                                    f"({strategy}): too similar ({similarity:.2f})",
                                    file=sys.stderr,
                                )
                                break
                            print(
                                f"Similarity {similarity:.2f} for record {index + 1}, "
                                f"variant {variant_index + 1} ({strategy}); retrying",
                                file=sys.stderr,
                            )
                            continue

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
                        dedup_threshold=dedup_threshold,
                        completed=completed,
                    )
                    generated_this_run += 1
                    print(
                        f"[{len(completed)}/{total}] record {index + 1}, "
                        f"variant {variant_index + 1}/{multiplier} ({strategy})"
                    )
                    break
                except (requests.RequestException, ValueError) as exc:
                    if attempt == retries:
                        print(
                            f"Failed record {index + 1}, variant {variant_index + 1} "
                            f"({strategy}) after {retries} attempts: {exc}",
                            file=sys.stderr,
                        )
                    else:
                        wait = attempt * 2
                        print(
                            f"Retrying record {index + 1}, variant {variant_index + 1} "
                            f"({strategy}) ({attempt}/{retries}) in {wait}s: {exc}",
                            file=sys.stderr,
                        )
                        time.sleep(wait)

    original_count, synthetic_count, dropped = finalize_output(
        records,
        synthetic_path,
        output_path,
        extend_existing,
        dedup_threshold,
    )
    cleanup_sidecars(synthetic_path, progress_path)

    print(
        f"\nWrote {original_count + synthetic_count if extend_existing else synthetic_count} "
        f"records to {output_path} "
        f"({original_count} original, {synthetic_count} synthetic; "
        f"{generated_this_run} generated this run)"
    )
    if dedup_threshold is not None:
        print(
            f"Dedup: dropped {dropped} near-duplicates at finalize "
            f"(threshold {dedup_threshold}); {skipped_similar} regeneration retries for similarity"
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
        "--dedup-threshold",
        type=float,
        default=0.85,
        help="Drop synthetics with word-shingle Jaccard similarity >= threshold (default: 0.85; use 1.0 to disable)",
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

    dedup_threshold = None if args.dedup_threshold >= 1.0 else args.dedup_threshold

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
            dedup_threshold=dedup_threshold,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Use --fresh to discard existing progress and start over.", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
