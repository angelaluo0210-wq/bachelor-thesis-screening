"""Build the complete cleaned processed dataset from immutable raw CSVs.

Only confirmed label=1 repairs from ``mojibake_label1_repairs.csv`` are
applied.  Reviewed manual cases are preserved and audited as no-change.  This
builder does not deduplicate, resolve label conflicts, split data, or train a
model.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    from .mojibake_label1_qc import (
        CANDIDATES_PATH,
        MANUAL_PATH,
        REPAIRS_PATH,
        article_identifier,
        is_valid_accented_latin,
    )
    from .mojibake_qc import RAW_DATA_DIR, repair_text
except ImportError:  # Support direct execution.
    from mojibake_label1_qc import (
        CANDIDATES_PATH,
        MANUAL_PATH,
        REPAIRS_PATH,
        article_identifier,
        is_valid_accented_latin,
    )
    from mojibake_qc import RAW_DATA_DIR, repair_text


ROOT = Path(__file__).resolve().parents[1]
CLEANED_DIR = ROOT / "data" / "processed" / "cleaned"
COMBINED_PATH = ROOT / "data" / "processed" / "all_reviews_cleaned.csv"
AUDIT_PATH = ROOT / "reports" / "processed_data_audit.csv"
SUMMARY_PATH = ROOT / "reports" / "processed_data_summary.json"

AUDIT_COLUMNS = [
    "review_id",
    "source_file",
    "original_row_index",
    "identifier_type",
    "article_identifier",
    "field_modified",
    "original_value",
    "cleaned_value",
    "original_snippet",
    "cleaned_snippet",
    "modification_type",
    "evidence_source",
    "QC_status",
]
PROTECTED_UNICODE = ("α", "β", "γ", "μ", "–", "—", "±", "≤", "≥", "°", "é", "ö", "ü", "ñ")

# This record appeared after the source CSV was externally updated following
# the label=1 preview.  PubMed/PMC and the previously approved fixed phrase
# establish sleep�Cwake -> sleep–wake without a guess.
POST_QC_CONFIRMED_REPAIRS = [
    {
        "review_id": "Leenaars_2019",
        "source_file": "Leenaars_2019_work_qc.csv",
        "row_index": "5",
        "identifier_type": "doi",
        "article_identifier": "https://doi.org/10.1016/s0006-8993(97)01308-5",
        "field": "abstract",
        "suspicious_pattern": "�C",
        "replacement": "–",
        "confidence": "high",
        "action": "auto_repaired",
        "reference_note": "exact DOI/title; PubMed PMID 9473663 and PMC full text confirm sleep–wake cycle",
    }
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_columns(columns: List[str], context: str) -> None:
    if not columns:
        raise ValueError(f"Empty CSV header in {context}")
    empty = [index for index, column in enumerate(columns) if not column.strip()]
    if empty:
        raise ValueError(f"Empty CSV header name(s) at positions {empty} in {context}")
    duplicates = sorted(column for column, count in Counter(columns).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate CSV header name(s) {duplicates} in {context}")


def _read_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read a CSV without replacement decoding, duplicate-key collapse, or bad-row skipping."""
    with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        try:
            columns = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty CSV file: {path}") from exc
        except csv.Error as exc:
            raise ValueError(f"Malformed CSV header in {path}: {exc}") from exc
        _validate_columns(columns, str(path))

        rows: List[Dict[str, str]] = []
        try:
            for row_index, values in enumerate(reader):
                if len(values) != len(columns):
                    raise ValueError(
                        f"Unexpected field count in {path} at logical row {row_index}: "
                        f"expected {len(columns)}, found {len(values)}"
                    )
                rows.append(dict(zip(columns, values)))
        except csv.Error as exc:
            raise ValueError(f"Malformed CSV record in {path} near physical line {reader.line_num}: {exc}") from exc
        return columns, rows


def _write_rows(path: Path, columns: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    _validate_columns(columns, str(path))
    materialized_rows = list(rows)
    expected = set(columns)
    for row_index, row in enumerate(materialized_rows):
        actual = set(row)
        if actual != expected:
            missing = sorted(expected - actual)
            extras = sorted(actual - expected)
            raise ValueError(
                f"Output schema mismatch for {path} at row {row_index}: "
                f"missing={missing}, extras={extras}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(materialized_rows)


def _expected_review_id(source_path: Path) -> str:
    return re.sub(r"_(?:work_qc|final)$", "", source_path.stem)


def _validate_source_rows(source_path: Path, rows: List[Dict[str, str]]) -> None:
    expected_review = _expected_review_id(source_path)
    embedded_record = re.compile(
        r"\ufffd\?,(?:PubMed|OpenAlex|Missing),(?:PubMed|OpenAlex|Missing),[01],"
        r"[^\r\n]+\r?\nhttps://doi\.org/",
        re.DOTALL,
    )
    for row_index, row in enumerate(rows):
        label = str(row.get("label_included", "")).strip()
        if label not in {"0", "1"}:
            raise ValueError(f"Unexpected label {label!r} in {source_path} at logical row {row_index}")
        review_id = str(row.get("review_key", "")).strip()
        if review_id != expected_review:
            raise ValueError(
                f"Unexpected review_key {review_id!r} in {source_path} at logical row {row_index}; "
                f"expected {expected_review!r}"
            )
        for field, value in row.items():
            if embedded_record.search(str(value or "")):
                raise ValueError(
                    f"Embedded secondary-record signature in {source_path} "
                    f"at logical row {row_index}, field {field!r}"
                )


def _snippet(value: Any, limit: int = 240) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def apply_confirmed_repairs(original: str, repair_rows: List[Dict[str, str]]) -> str:
    """Apply ordered, logged replacements; fail if an expected span is absent."""
    cleaned = original
    cursor = 0
    for repair in repair_rows:
        pattern = repair["suspicious_pattern"]
        replacement = repair["replacement"]
        position = cleaned.find(pattern, cursor)
        if position < 0:
            raise ValueError(
                f"Confirmed pattern {pattern!r} not found after offset {cursor} "
                f"for {repair['source_file']}:{repair['row_index']}:{repair['field']}"
            )
        cleaned = cleaned[:position] + replacement + cleaned[position + len(pattern) :]
        cursor = position + len(replacement)
    return cleaned


def _unicode_counts(rows: Iterable[Dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        text = str(row.get("title") or "") + str(row.get("abstract") or "")
        for character in PROTECTED_UNICODE:
            counts[character] += text.count(character)
    return counts


def build_processed_data() -> Dict[str, Any]:
    """Rebuild all 26 cleaned review files and the combined processed table."""
    source_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    if len(source_files) != 26:
        raise ValueError(f"Expected 26 raw review CSVs, found {len(source_files)}")
    hashes_before = {path: _sha256(path) for path in source_files}

    # Preflight every source before opening any output path. This prevents a
    # late malformed review from leaving a partially rebuilt processed tree.
    source_tables: Dict[Path, Tuple[List[str], List[Dict[str, str]]]] = {}
    for source_path in source_files:
        columns, raw_rows = _read_rows(source_path)
        _validate_source_rows(source_path, raw_rows)
        source_tables[source_path] = (columns, raw_rows)

    _, repair_log = _read_rows(REPAIRS_PATH)
    confirmed_repairs = [row for row in repair_log if row["action"] == "auto_repaired"]
    if any(row["action"] not in {"auto_repaired", "manual_review_required"} for row in repair_log):
        raise ValueError("Unexpected action in label=1 repair log")
    existing_patch_keys = {
        (row["source_file"], row["row_index"], row["field"], row["suspicious_pattern"], row["replacement"])
        for row in confirmed_repairs
    }
    for repair in POST_QC_CONFIRMED_REPAIRS:
        key = (repair["source_file"], repair["row_index"], repair["field"], repair["suspicious_pattern"], repair["replacement"])
        if key not in existing_patch_keys:
            confirmed_repairs.append(dict(repair))
    _, manual_log = _read_rows(MANUAL_PATH)
    _, candidate_log = _read_rows(CANDIDATES_PATH)

    repairs_by_field: Dict[Tuple[str, int, str], List[Dict[str, str]]] = defaultdict(list)
    for row in confirmed_repairs:
        repairs_by_field[(row["source_file"], int(row["row_index"]), row["field"])].append(row)
    manual_by_field: Dict[Tuple[str, int, str], List[Dict[str, str]]] = defaultdict(list)
    for row in manual_log:
        manual_by_field[(row["source_file"], int(row["row_index"]), row["field"])].append(row)

    # The previous specialist QC established that embedded â in Goiânia is
    # valid Latin text.  Preserve and audit any such candidate as no-change.
    normal_unicode_fields = set()
    for row in candidate_log:
        key = (row["source_file"], int(row["row_index"]), row["field"])
        if row["suspicious_pattern"] == "â":
            normal_unicode_fields.add(key)

    combined_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    raw_rows_for_unicode: List[Dict[str, Any]] = []
    all_columns: List[str] = []
    total_rows = label_one_rows = label_zero_rows = other_label_rows = 0
    modified_fields = set()
    modified_records = set()

    for source_path in source_files:
        columns, raw_rows = source_tables[source_path]
        raw_rows_for_unicode.extend(raw_rows)
        for column in columns:
            if column not in all_columns:
                all_columns.append(column)
        cleaned_rows: List[Dict[str, Any]] = []

        for row_index, raw_row in enumerate(raw_rows):
            row = dict(raw_row)
            review_id = row.get("review_key", source_path.stem)
            identifier_type, identifier_value = article_identifier(row)
            label = str(row.get("label_included", row.get("label", ""))).strip()
            total_rows += 1
            label_one_rows += label in {"1", "1.0"}
            label_zero_rows += label in {"0", "0.0"}
            other_label_rows += label not in {"0", "0.0", "1", "1.0"}

            for field in ("title", "abstract"):
                key = (source_path.name, row_index, field)
                field_repairs = repairs_by_field.get(key, [])
                if field_repairs:
                    if label not in {"1", "1.0"}:
                        raise ValueError(f"Non-label=1 repair attempted for {key}")
                    for repair in field_repairs:
                        if repair["review_id"] != review_id:
                            raise ValueError(f"review_id mismatch for {key}")
                        if repair["identifier_type"] != identifier_type or repair["article_identifier"] != identifier_value:
                            raise ValueError(f"Article identifier mismatch for {key}")
                    original = str(row.get(field) or "")
                    cleaned = apply_confirmed_repairs(original, field_repairs)
                    if cleaned == original:
                        raise ValueError(f"Confirmed repair made no change for {key}")
                    row[field] = cleaned
                    modified_fields.add(key)
                    modified_records.add((source_path.name, row_index))
                    audit_rows.append(
                        {
                            "review_id": review_id,
                            "source_file": source_path.name,
                            "original_row_index": row_index,
                            "identifier_type": identifier_type,
                            "article_identifier": identifier_value,
                            "field_modified": field,
                            "original_value": original,
                            "cleaned_value": cleaned,
                            "original_snippet": _snippet(original),
                            "cleaned_snippet": _snippet(cleaned),
                            "modification_type": "confirmed_mojibake_repair",
                            "evidence_source": "; ".join(sorted({r.get("reference_note", "confirmed label=1 QC") for r in field_repairs})),
                            "QC_status": "confirmed_repaired",
                        }
                    )

                if key in manual_by_field or key in normal_unicode_fields:
                    evidence = "human review: confirmed no change" if key in manual_by_field else "Unicode validation: valid accented Latin (Goiânia)"
                    audit_rows.append(
                        {
                            "review_id": review_id,
                            "source_file": source_path.name,
                            "original_row_index": row_index,
                            "identifier_type": identifier_type,
                            "article_identifier": identifier_value,
                            "field_modified": field,
                            "original_value": row.get(field, ""),
                            "cleaned_value": row.get(field, ""),
                            "original_snippet": _snippet(row.get(field, "")),
                            "cleaned_snippet": _snippet(row.get(field, "")),
                            "modification_type": "reviewed_no_change",
                            "evidence_source": evidence,
                            "QC_status": "reviewed_no_change",
                        }
                    )

            row["_source_file"] = source_path.name
            row["_original_row_index"] = row_index
            cleaned_rows.append(row)
            combined_rows.append(row)

        if len(cleaned_rows) != len(raw_rows):
            raise RuntimeError(f"Row count changed for {source_path.name}")
        _write_rows(
            CLEANED_DIR / f"{source_path.stem}_cleaned.csv",
            columns + ["_source_file", "_original_row_index"],
            cleaned_rows,
        )

    combined_columns = all_columns + ["_source_file", "_original_row_index"]
    _write_rows(COMBINED_PATH, combined_columns, combined_rows)
    _write_rows(AUDIT_PATH, AUDIT_COLUMNS, audit_rows)

    if total_rows != len(combined_rows):
        raise RuntimeError("Combined row count differs from per-review row total")
    if label_one_rows + label_zero_rows + other_label_rows != total_rows:
        raise RuntimeError("Processed label accounting does not equal total rows")

    protected_before = _unicode_counts(raw_rows_for_unicode)
    protected_after = _unicode_counts(combined_rows)
    lost_unicode = {
        char: protected_before[char] - protected_after[char]
        for char in PROTECTED_UNICODE
        if protected_after[char] < protected_before[char]
    }
    if lost_unicode:
        raise RuntimeError(f"Protected scientific Unicode was lost: {lost_unicode}")

    reviewed_fields = set(manual_by_field) | normal_unicode_fields
    unresolved_occurrences = 0
    unresolved_records = set()
    unresolved_label1_occurrences = 0
    unresolved_label1_records = set()
    for row in combined_rows:
        source_file = row["_source_file"]
        row_index = int(row["_original_row_index"])
        label = str(row.get("label_included", row.get("label", ""))).strip()
        for field in ("title", "abstract"):
            issues = repair_text(row.get(field))[1]
            if is_valid_accented_latin(str(row.get(field) or ""), issues):
                issues = []
            if (source_file, row_index, field) in reviewed_fields:
                issues = []
            if not issues:
                continue
            unresolved_occurrences += len(issues)
            unresolved_records.add((source_file, row_index))
            if label in {"1", "1.0"}:
                unresolved_label1_occurrences += len(issues)
                unresolved_label1_records.add((source_file, row_index))

    if unresolved_label1_occurrences:
        raise RuntimeError("Confirmed label=1 mojibake remains in processed data")

    hashes_after = {path: _sha256(path) for path in source_files}
    if hashes_before != hashes_after:
        raise RuntimeError("Raw CSV SHA-256 changed during processed-data build")

    summary = {
        "source_review_files": len(source_files),
        "total_rows": total_rows,
        "label_1_rows": label_one_rows,
        "label_0_rows": label_zero_rows,
        "other_or_malformed_label_rows_preserved": other_label_rows,
        "confirmed_repair_operations_applied": len(confirmed_repairs),
        "text_fields_modified": len(modified_fields),
        "records_modified": len(modified_records),
        "reviewed_no_change_fields": sum(row["modification_type"] == "reviewed_no_change" for row in audit_rows),
        "unresolved_mojibake_occurrences_all_labels": unresolved_occurrences,
        "unresolved_mojibake_records_all_labels": len(unresolved_records),
        "unresolved_mojibake_occurrences_label_1": unresolved_label1_occurrences,
        "unresolved_mojibake_records_label_1": len(unresolved_label1_records),
        "protected_unicode_counts_before": dict(protected_before),
        "protected_unicode_counts_after": dict(protected_after),
        "raw_sha256_unchanged": True,
        "per_review_cleaned_directory": str(CLEANED_DIR),
        "combined_cleaned_path": str(COMBINED_PATH),
        "audit_path": str(AUDIT_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(build_processed_data(), ensure_ascii=True, indent=2))
