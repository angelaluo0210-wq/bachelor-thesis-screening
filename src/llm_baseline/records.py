"""Stable record IDs, run fingerprints, and append-only prediction storage."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src import screening_data as data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_record_id(row: dict[str, Any], source_row_index: int) -> str:
    review = data.normalize_review_key(row.get(data.REVIEW_COLUMN, ""))
    if not review:
        raise ValueError("A stable record ID requires review_key")
    identifiers = []
    for name in data.IDENTIFIER_COLUMNS:
        value = data.safe_text(row.get(name, ""))
        if value:
            identifiers.append(f"{name}={value.lower()}")
    identity = "|".join(identifiers) if identifiers else f"source_row_index={source_row_index}"
    return f"{review}|{identity}"


def record_id_for_row(row: dict[str, Any], fallback_index: int) -> str:
    """Use a preserved source index when a deterministic Pilot was sampled."""
    source_index = int(row.get("_qwen_source_row_index", fallback_index))
    return stable_record_id(row, source_index)


def require_unique_record_ids(rows: Iterable[dict[str, Any]]) -> list[str]:
    values = [record_id_for_row(row, index) for index, row in enumerate(rows)]
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate stable record identifiers: {duplicates}")
    return values


def configuration_fingerprint(configuration: dict[str, Any]) -> str:
    canonical = json.dumps(configuration, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def create_or_resume_run(run_dir: Path, configuration: dict[str, Any], resume: bool) -> dict[str, Any]:
    """Create a new manifest or validate an exact-fingerprint resume target."""
    manifest_path = run_dir / "run_manifest.json"
    fingerprint = configuration_fingerprint(configuration)
    if resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Cannot resume without manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("configuration_fingerprint") != fingerprint:
            raise ValueError("Refusing to mix predictions from different prompt or API configurations")
        return manifest
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing Qwen baseline run: {run_dir}")
    run_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration, "configuration_fingerprint": fingerprint,
        "prediction_file": "predictions.jsonl",
        # Validation callers retain the historical default.  A separately
        # authorized final-Test runner records its explicit phase instead of
        # masquerading as a Validation artifact.
        "test_access": configuration.get("test_access", "forbidden"),
        "test_evaluated": bool(configuration.get("test_evaluated", False)),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def completed_record_ids(run_dir: Path) -> set[str]:
    path = run_dir / "predictions.jsonl"
    if not path.is_file():
        return set()
    completed: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
                record_id = value["record_id"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid prediction JSONL at line {line_number}: {exc}") from exc
            if record_id in completed:
                raise ValueError(f"Duplicate stored prediction record_id: {record_id}")
            completed.add(record_id)
    return completed


def append_prediction(run_dir: Path, value: dict[str, Any], known_ids: set[str] | None = None) -> None:
    if "record_id" not in value:
        raise ValueError("Prediction records require record_id")
    existing = known_ids if known_ids is not None else completed_record_ids(run_dir)
    if value["record_id"] in existing:
        raise ValueError(f"Duplicate stored prediction record_id: {value['record_id']}")
    path = run_dir / "predictions.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    if known_ids is not None:
        known_ids.add(value["record_id"])
