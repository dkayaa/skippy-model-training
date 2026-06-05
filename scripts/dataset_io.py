"""Shared helpers for reading and writing labeled datasets as JSON or JSONL."""

import json
from pathlib import Path


def is_jsonl(path: Path) -> bool:
    return path.suffix.lower() == ".jsonl"


def validate_record(record: dict, index: int) -> None:
    if "text" not in record or "label" not in record:
        raise ValueError(f"Record {index} must include 'text' and 'label' fields")


def load_dataset(path: Path) -> list[dict]:
    if is_jsonl(path):
        records = []
        with open(path, encoding="utf-8") as f:
            for line_number, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                validate_record(record, line_number)
                records.append(record)
        return records

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")

    for index, record in enumerate(data):
        validate_record(record, index)

    return data


def write_dataset(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if is_jsonl(path):
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return

    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
