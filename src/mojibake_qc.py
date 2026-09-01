"""Conservative mojibake detection and repair for review CSV files.

Raw CSV files are read-only inputs.  Every source file is copied to the
processed directory, with deterministic repairs applied only to title and
abstract fields.  Ambiguous artifacts are logged for manual review.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT / "数据" / "review整理"
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
REPORT_PATH = ROOT / "reports" / "mojibake_repairs.csv"
SUMMARY_PATH = ROOT / "reports" / "mojibake_summary.json"

LOG_COLUMNS = [
    "review_id",
    "source_file",
    "row_index",
    "label",
    "field",
    "original_text_snippet",
    "repaired_text_snippet",
    "suspicious_pattern",
    "replacement",
    "confidence",
    "action",
]


def _build_deterministic_replacements() -> Dict[str, str]:
    """Build only reversible, known UTF-8-as-Windows-1252 substitutions."""
    replacements = {
        # GBK pairs the final UTF-8 byte of the en dash with the following
        # ASCII ``w`` byte, so 搘 represents both pieces (–w), not only –.
        "鈥搘": "–w",
        "鈥檚": "’s",
        "â€™": "’",
        "â€˜": "‘",
        "â€“": "–",
        "â€”": "—",
        "â€œ": "“",
        "â€": "”",
        "â€¦": "…",
    }

    # Latin-1 accents and these punctuation characters have an unambiguous
    # UTF-8 -> Windows-1252 mojibake representation.  Generating the table
    # avoids heuristic whole-string transcoding.
    intended = [chr(codepoint) for codepoint in range(0x00A0, 0x0100)]
    intended.extend(["–", "—", "‘", "’", "“", "”", "…", "−", "™", "€"])
    for character in intended:
        try:
            artifact = character.encode("utf-8").decode("cp1252")
        except UnicodeDecodeError:
            continue
        if artifact != character:
            replacements[artifact] = character
    return replacements


AUTO_REPLACEMENTS = _build_deterministic_replacements()
AMBIGUOUS_MARKERS = ("鈥", "锟", "�", "Ã", "Â", "â", "卤")
_PATTERNS = sorted(set(AUTO_REPLACEMENTS) | set(AMBIGUOUS_MARKERS), key=len, reverse=True)
SUSPICIOUS_RE = re.compile("|".join(re.escape(pattern) for pattern in _PATTERNS))


def _numeric_plus_minus_context(text: str, start: int, end: int) -> bool:
    """Return true only when 卤 occurs between numeric values (e.g. 12 卤 3)."""
    before = text[:start]
    after = text[end:]
    return bool(re.search(r"\d\s*$", before) and re.match(r"^\s*\d", after))


def repair_text(text: Any) -> Tuple[Any, List[Dict[str, Any]]]:
    """Repair high-confidence artifacts and describe every suspicious match.

    Missing/non-string values pass through unchanged.  Each returned issue has
    an action of either ``auto_repaired`` or ``manual_review_required``.
    """
    if not isinstance(text, str) or not text:
        return text, []

    issues: List[Dict[str, Any]] = []
    pieces: List[str] = []
    cursor = 0
    matches = list(SUSPICIOUS_RE.finditer(text))
    # Adjacent matches are fragments of a larger damaged byte run.  Repairing
    # the fragments independently can manufacture a new artifact (for example
    # ``ââ‚¬â„¢`` -> ``â€™``), so the entire run must remain for manual review.
    forced_manual: set[int] = set()
    for index in range(len(matches) - 1):
        if matches[index].end() == matches[index + 1].start():
            forced_manual.update((index, index + 1))

    for index, match in enumerate(matches):
        pattern = match.group(0)
        replacement = AUTO_REPLACEMENTS.get(pattern)
        if pattern == "卤":
            replacement = "±" if _numeric_plus_minus_context(text, match.start(), match.end()) else None
        if index in forced_manual:
            replacement = None

        pieces.append(text[cursor : match.start()])
        if replacement is None:
            pieces.append(pattern)
            action = "manual_review_required"
            confidence = "low"
            replacement_for_log = ""
        else:
            pieces.append(replacement)
            action = "auto_repaired"
            confidence = "high"
            replacement_for_log = replacement

        issues.append(
            {
                "start": match.start(),
                "end": match.end(),
                "suspicious_pattern": pattern,
                "replacement": replacement_for_log,
                "confidence": confidence,
                "action": action,
            }
        )
        cursor = match.end()
    pieces.append(text[cursor:])
    return "".join(pieces), issues


def _snippet(text: str, start: int, end: int, radius: int = 60) -> str:
    return text[max(0, start - radius) : min(len(text), end + radius)].replace("\r", " ").replace("\n", " ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pandas() -> Any:
    """Import pandas only for dataframe/file operations, not pure text repair."""
    import pandas as pd

    return pd


def _read_csv(path: Path) -> Any:
    pd = _load_pandas()
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, engine="python", on_bad_lines="skip")
        except Exception:
            continue
    return pd.read_csv(
        path,
        encoding="utf-8",
        encoding_errors="replace",
        engine="python",
        on_bad_lines="skip",
    )


def _find_column(columns: Iterable[Any], aliases: Tuple[str, ...]) -> Any | None:
    lookup = {str(column).strip().lower(): column for column in columns}
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    return None


def clean_review_file(path: Path, output_dir: Path) -> Tuple[Any, List[Dict[str, Any]]]:
    """Write one cleaned copy and return its dataframe plus repair log rows."""
    pd = _load_pandas()
    df = _read_csv(path)
    title_col = _find_column(df.columns, ("title", "paper_title", "article_title", "study_title"))
    abstract_col = _find_column(df.columns, ("abstract", "article_abstract", "summary", "description"))
    label_col = _find_column(df.columns, ("label_included", "label", "included", "decision", "screening_label"))
    review_col = _find_column(df.columns, ("review_id", "review_key", "review", "study_id"))

    logs: List[Dict[str, Any]] = []
    numeric_labels = pd.to_numeric(df[label_col], errors="coerce") if label_col else pd.Series(index=df.index, dtype=float)
    priority_indices = list(df.index[numeric_labels.eq(1)]) + list(df.index[~numeric_labels.eq(1)])

    for row_index in priority_indices:
        for field, column in (("title", title_col), ("abstract", abstract_col)):
            if column is None:
                continue
            original = df.at[row_index, column]
            repaired, issues = repair_text(original)
            if not issues:
                continue
            df.at[row_index, column] = repaired
            original_string = str(original)
            repaired_string = str(repaired)
            offset_delta = 0
            for issue in issues:
                replacement = issue["replacement"] or issue["suspicious_pattern"]
                repaired_start = issue["start"] + offset_delta
                repaired_end = repaired_start + len(replacement)
                offset_delta += len(replacement) - (issue["end"] - issue["start"])
                logs.append(
                    {
                        "review_id": df.at[row_index, review_col] if review_col else path.stem,
                        "source_file": path.name,
                        "row_index": int(row_index),
                        "label": df.at[row_index, label_col] if label_col else pd.NA,
                        "field": field,
                        "original_text_snippet": _snippet(original_string, issue["start"], issue["end"]),
                        "repaired_text_snippet": _snippet(repaired_string, repaired_start, repaired_end),
                        "suspicious_pattern": issue["suspicious_pattern"],
                        "replacement": issue["replacement"],
                        "confidence": issue["confidence"],
                        "action": issue["action"],
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f"{path.stem}_cleaned.csv", index=False, encoding="utf-8-sig")
    return df, logs


def scan_suspicious(df: Any) -> Dict[str, int]:
    """Count suspicious patterns remaining in title/abstract after cleaning."""
    counts: Dict[str, int] = {}
    for aliases in (("title", "paper_title", "article_title", "study_title"), ("abstract", "article_abstract", "summary", "description")):
        column = _find_column(df.columns, aliases)
        if column is None:
            continue
        for value in df[column].dropna():
            if not isinstance(value, str):
                continue
            for match in SUSPICIOUS_RE.finditer(value):
                pattern = match.group(0)
                counts[pattern] = counts.get(pattern, 0) + 1
    return counts


def run_mojibake_qc(
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    report_path: Path = REPORT_PATH,
) -> Dict[str, Any]:
    """Clean all review CSVs, write logs, rescan outputs, and verify raw hashes."""
    pd = _load_pandas()
    source_files = sorted(raw_dir.glob("*.csv"))
    if not source_files:
        raise FileNotFoundError(f"No raw CSV files found in {raw_dir}")

    hashes_before = {path: _sha256(path) for path in source_files}
    all_logs: List[Dict[str, Any]] = []
    cleaned_frames: List[pd.DataFrame] = []
    total_rows = 0
    label_one_rows = 0

    for path in source_files:
        cleaned, logs = clean_review_file(path, processed_dir)
        cleaned_frames.append(cleaned)
        all_logs.extend(logs)
        total_rows += len(cleaned)
        label_col = _find_column(cleaned.columns, ("label_included", "label", "included", "decision", "screening_label"))
        if label_col is not None:
            label_one_rows += int(pd.to_numeric(cleaned[label_col], errors="coerce").eq(1).sum())

    hashes_after = {path: _sha256(path) for path in source_files}
    if hashes_before != hashes_after:
        changed = [path.name for path in source_files if hashes_before[path] != hashes_after[path]]
        raise RuntimeError(f"Raw CSV hash changed unexpectedly: {changed}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    log_df = pd.DataFrame(all_logs, columns=LOG_COLUMNS)
    if not log_df.empty:
        label_priority = pd.to_numeric(log_df["label"], errors="coerce").eq(1)
        log_df = log_df.assign(_priority=~label_priority).sort_values(
            ["_priority", "source_file", "row_index", "field"], kind="stable"
        ).drop(columns="_priority")
    log_df.to_csv(report_path, index=False, encoding="utf-8-sig")

    remaining: Dict[str, int] = {}
    for frame in cleaned_frames:
        for pattern, count in scan_suspicious(frame).items():
            remaining[pattern] = remaining.get(pattern, 0) + count

    auto_mask = log_df["action"].eq("auto_repaired") if not log_df.empty else pd.Series(dtype=bool)
    manual_mask = log_df["action"].eq("manual_review_required") if not log_df.empty else pd.Series(dtype=bool)
    affected_keys = ["source_file", "row_index"]
    summary = {
        "total_rows_scanned": total_rows,
        "label_1_rows": label_one_rows,
        "mojibake_occurrences_detected": int(len(log_df)),
        "records_with_mojibake": int(log_df[affected_keys].drop_duplicates().shape[0]) if not log_df.empty else 0,
        "auto_repaired_occurrences": int(auto_mask.sum()),
        "auto_repaired_records": int(log_df.loc[auto_mask, affected_keys].drop_duplicates().shape[0]) if auto_mask.any() else 0,
        "auto_repaired_label_1_records": int(
            log_df.loc[auto_mask & pd.to_numeric(log_df["label"], errors="coerce").eq(1), affected_keys].drop_duplicates().shape[0]
        ) if auto_mask.any() else 0,
        "manual_review_occurrences": int(manual_mask.sum()),
        "manual_review_records": int(log_df.loc[manual_mask, affected_keys].drop_duplicates().shape[0]) if manual_mask.any() else 0,
        "manual_review_label_1_records": int(
            log_df.loc[manual_mask & pd.to_numeric(log_df["label"], errors="coerce").eq(1), affected_keys].drop_duplicates().shape[0]
        ) if manual_mask.any() else 0,
        "remaining_suspicious_patterns": dict(sorted(remaining.items())),
        "raw_files_unchanged": True,
        "cleaned_directory": str(processed_dir),
        "repair_log": str(report_path),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def print_mojibake_summary(summary: Dict[str, Any]) -> None:
    print("\n=== Mojibake / encoding artifact QC ===")
    print(f"Rows scanned: {summary['total_rows_scanned']}")
    print(f"label=1 rows: {summary['label_1_rows']}")
    print(f"Detected occurrences: {summary['mojibake_occurrences_detected']}")
    print(f"Auto-repaired records: {summary['auto_repaired_records']}")
    print(f"Manual-review records: {summary['manual_review_records']}")
    print(f"Remaining suspicious patterns: {summary['remaining_suspicious_patterns']}")


if __name__ == "__main__":
    print_mojibake_summary(run_mojibake_qc())
