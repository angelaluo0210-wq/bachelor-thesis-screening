"""High-priority mojibake review and repair for label=1 records only.

The candidate set comes from ``reports/mojibake_repairs.csv``.  Raw CSVs are
never written.  Irrecoverable replacement characters are repaired only when
an exact PMID/title match to the cached PubMed abstract, or an exact-title
clean duplicate in another review, provides a high-confidence reference.
"""

from __future__ import annotations

import csv
import difflib
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    from .mojibake_qc import RAW_DATA_DIR, REPORT_PATH, repair_text
except ImportError:  # Support direct execution.
    from mojibake_qc import RAW_DATA_DIR, REPORT_PATH, repair_text


ROOT = Path(__file__).resolve().parents[1]
PUBMED_CACHE = ROOT / "data" / "reference" / "pubmed_label1_abstracts.xml"
CANDIDATES_PATH = ROOT / "reports" / "mojibake_label1_candidates.csv"
REPAIRS_PATH = ROOT / "reports" / "mojibake_label1_repairs.csv"
MANUAL_PATH = ROOT / "reports" / "mojibake_label1_manual_review.csv"
PREVIEW_PATH = ROOT / "data" / "processed" / "label1_mojibake_cleaned_preview.csv"
SUMMARY_PATH = ROOT / "reports" / "mojibake_label1_summary.json"

CANDIDATE_COLUMNS = [
    "review_id", "source_file", "row_index", "label", "title", "abstract",
    "suspicious_pattern", "field", "original_text_snippet", "issue_count",
]
REPAIR_COLUMNS = [
    "review_id", "source_file", "row_index", "identifier_type", "article_identifier",
    "field", "original_text_snippet",
    "repaired_text_snippet", "suspicious_pattern", "replacement", "confidence", "action",
    "reference_note",
]
MANUAL_COLUMNS = [
    "review_id", "source_file", "row_index", "field", "title", "abstract",
    "suspicious_pattern", "original_text_snippet", "suggested_repair_if_any", "notes",
]
PREVIEW_COLUMNS = [
    "review_id", "source_file", "row_index", "label", "title_original", "abstract_original",
    "title_cleaned", "abstract_cleaned", "repair_status", "remaining_issue_count",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, columns: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _normalize_title(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _pmid(value: Any) -> str:
    return "".join(re.findall(r"\d", str(value or "")))


def article_identifier(row: Dict[str, Any]) -> Tuple[str, str]:
    """Return a stable article identifier, preferring DOI then PMID/OpenAlex."""
    for column in ("doi", "pmid", "openalex_id"):
        value = str(row.get(column) or "").strip()
        if value:
            return column, value
    return "title", _normalize_title(row.get("title"))


def _node_text(node: ET.Element | None) -> str:
    return "" if node is None else "".join(node.itertext()).strip()


def load_pubmed_cache(path: Path = PUBMED_CACHE) -> Dict[str, Dict[str, str]]:
    """Load cached PubMed titles and abstracts keyed by exact PMID."""
    if not path.exists():
        raise FileNotFoundError(f"Cached PubMed XML not found: {path}")
    references: Dict[str, Dict[str, str]] = {}
    root = ET.parse(path).getroot()
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//MedlineCitation/PMID") or ""
        title = _node_text(article.find(".//Article/ArticleTitle"))
        parts = []
        for node in article.findall(".//Article/Abstract/AbstractText"):
            text = _node_text(node)
            label = node.get("Label")
            if label and not text.upper().startswith(label.upper() + ":"):
                text = f"{label}: {text}"
            if text:
                parts.append(text)
        references[pmid] = {"title": title, "abstract": " ".join(parts)}
    return references


def is_valid_accented_latin(text: str, issues: List[Dict[str, Any]]) -> bool:
    """Recognize a prior false positive such as the valid name ``Goiânia``."""
    if not issues or any(issue["suspicious_pattern"] != "â" for issue in issues):
        return False
    for issue in issues:
        start, end = issue["start"], issue["end"]
        if start == 0 or end == len(text):
            return False
        if not (text[start - 1].isalpha() and text[end].isalpha()):
            return False
    return True


def recover_from_reference(
    local_text: str,
    reference_text: str,
    minimum_ratio: float = 0.98,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Replace only aligned spans containing ``�`` from a trusted reference."""
    if not local_text or not reference_text:
        return local_text, []
    matcher = difflib.SequenceMatcher(None, local_text, reference_text, autojunk=False)
    if matcher.ratio() < minimum_ratio:
        return local_text, []

    replacements = []
    covered_replacement_characters = 0
    for tag, local_start, local_end, ref_start, ref_end in matcher.get_opcodes():
        local_span = local_text[local_start:local_end]
        reference_span = reference_text[ref_start:ref_end]
        if tag == "equal" or "�" not in local_span:
            continue
        if "�" in reference_span or not reference_span:
            return local_text, []
        covered_replacement_characters += local_span.count("�")
        replacements.append(
            {
                "start": local_start,
                "end": local_end,
                "suspicious_pattern": local_span,
                "replacement": reference_span,
                "confidence": "high",
                "action": "auto_repaired",
            }
        )

    if covered_replacement_characters != local_text.count("�"):
        return local_text, []

    repaired = local_text
    for item in reversed(replacements):
        repaired = repaired[: item["start"]] + item["replacement"] + repaired[item["end"] :]
    if "�" in repaired:
        return local_text, []
    return repaired, replacements


def _snippet(text: str, start: int, end: int, radius: int = 60) -> str:
    return text[max(0, start - radius) : min(len(text), end + radius)].replace("\r", " ").replace("\n", " ")


def _candidate_keys() -> Tuple[List[Dict[str, str]], List[Tuple[str, int]]]:
    base_logs = _read_rows(REPORT_PATH)
    positive_logs = [row for row in base_logs if str(row.get("label")) in {"1", "1.0"}]
    keys = sorted({(row["source_file"], int(row["row_index"])) for row in positive_logs})
    return positive_logs, keys


def _clean_title_references(candidate_titles: set[str]) -> Dict[str, List[Tuple[str, int, Dict[str, str]]]]:
    references: Dict[str, List[Tuple[str, int, Dict[str, str]]]] = defaultdict(list)
    for path in sorted(RAW_DATA_DIR.glob("*.csv")):
        for index, row in enumerate(_read_rows(path)):
            title_key = _normalize_title(row.get("title"))
            if title_key not in candidate_titles:
                continue
            title_issues = repair_text(row.get("title"))[1]
            abstract_issues = repair_text(row.get("abstract"))[1]
            if not title_issues and not abstract_issues:
                references[title_key].append((path.name, index, row))
    return references


def run_label1_mojibake_qc() -> Dict[str, Any]:
    """Generate label=1 candidates, repairs, manual review, and cleaned preview."""
    source_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    hashes_before = {path: _sha256(path) for path in source_files}
    base_logs, candidate_keys = _candidate_keys()
    if len(candidate_keys) != 64:
        raise ValueError(f"Expected 64 label=1 candidate records, found {len(candidate_keys)}")

    rows_by_file = {path.name: _read_rows(path) for path in source_files if any(key[0] == path.name for key in candidate_keys)}
    candidates = {(file, index): rows_by_file[file][index] for file, index in candidate_keys}
    pubmed = load_pubmed_cache()
    clean_duplicates = _clean_title_references({_normalize_title(row.get("title")) for row in candidates.values()})

    grouped_logs: Dict[Tuple[str, int, str], List[Dict[str, str]]] = defaultdict(list)
    for row in base_logs:
        grouped_logs[(row["source_file"], int(row["row_index"]), row["field"])].append(row)

    candidate_output = []
    for (file, index), row in candidates.items():
        for field in ("title", "abstract"):
            issues = grouped_logs.get((file, index, field), [])
            if not issues:
                continue
            candidate_output.append(
                {
                    "review_id": row.get("review_key", Path(file).stem),
                    "source_file": file,
                    "row_index": index,
                    "label": row.get("label_included", row.get("label", "")),
                    "title": row.get("title", ""),
                    "abstract": row.get("abstract", ""),
                    "suspicious_pattern": ";".join(sorted({item["suspicious_pattern"] for item in issues})),
                    "field": field,
                    "original_text_snippet": " || ".join(item["original_text_snippet"] for item in issues),
                    "issue_count": len(issues),
                }
            )
    _write_rows(CANDIDATES_PATH, CANDIDATE_COLUMNS, candidate_output)

    repair_logs: List[Dict[str, Any]] = []
    manual_rows: List[Dict[str, Any]] = []
    previews: List[Dict[str, Any]] = []
    normal_unicode_records = set()

    for (file, index), raw_row in sorted(candidates.items()):
        review_id = raw_row.get("review_key", Path(file).stem)
        identifier_type, identifier_value = article_identifier(raw_row)
        title_original = raw_row.get("title", "")
        abstract_original = raw_row.get("abstract", "")
        cleaned = {"title": title_original, "abstract": abstract_original}
        changed = False

        for field in ("title", "abstract"):
            original = cleaned[field]
            initial_issues = repair_text(original)[1]
            if not initial_issues:
                continue
            if is_valid_accented_latin(original, initial_issues):
                normal_unicode_records.add((file, index))
                continue

            repaired = original
            reference = ""
            reference_note = ""
            pmid = _pmid(raw_row.get("pmid"))
            pubmed_entry = pubmed.get(pmid)
            if field == "abstract" and pubmed_entry:
                title_ratio = difflib.SequenceMatcher(
                    None, _normalize_title(title_original), _normalize_title(pubmed_entry["title"]), autojunk=False
                ).ratio()
                if title_ratio >= 0.98:
                    reference = pubmed_entry["abstract"]
                    reference_note = f"exact PMID {pmid} and PubMed title match"

            if not reference:
                for other_file, other_index, other_row in clean_duplicates.get(_normalize_title(title_original), []):
                    if (other_file, other_index) == (file, index):
                        continue
                    candidate_reference = other_row.get(field, "")
                    if candidate_reference:
                        reference = candidate_reference
                        reference_note = f"clean exact-title duplicate {other_file}:{other_index}"
                        break

            repaired, aligned_repairs = recover_from_reference(original, reference)
            if aligned_repairs:
                changed = True
                cleaned[field] = repaired
                delta = 0
                for item in aligned_repairs:
                    repaired_start = item["start"] + delta
                    repaired_end = repaired_start + len(item["replacement"])
                    delta += len(item["replacement"]) - (item["end"] - item["start"])
                    repair_logs.append(
                        {
                            "review_id": review_id,
                            "source_file": file,
                            "row_index": index,
                            "identifier_type": identifier_type,
                            "article_identifier": identifier_value,
                            "field": field,
                            "original_text_snippet": _snippet(original, item["start"], item["end"]),
                            "repaired_text_snippet": _snippet(repaired, repaired_start, repaired_end),
                            "suspicious_pattern": item["suspicious_pattern"],
                            "replacement": item["replacement"],
                            "confidence": "high",
                            "action": "auto_repaired",
                            "reference_note": reference_note,
                        }
                    )

        remaining = []
        for field in ("title", "abstract"):
            issues = repair_text(cleaned[field])[1]
            if is_valid_accented_latin(cleaned[field], issues):
                issues = []
            for issue in issues:
                remaining.append((field, issue))
                repair_logs.append(
                    {
                        "review_id": review_id,
                        "source_file": file,
                        "row_index": index,
                        "identifier_type": identifier_type,
                        "article_identifier": identifier_value,
                        "field": field,
                        "original_text_snippet": _snippet(cleaned[field], issue["start"], issue["end"]),
                        "repaired_text_snippet": _snippet(cleaned[field], issue["start"], issue["end"]),
                        "suspicious_pattern": issue["suspicious_pattern"],
                        "replacement": "",
                        "confidence": "low",
                        "action": "manual_review_required",
                    }
                )
                manual_rows.append(
                    {
                        "review_id": review_id,
                        "source_file": file,
                        "row_index": index,
                        "field": field,
                        "title": cleaned["title"],
                        "abstract": cleaned["abstract"],
                        "suspicious_pattern": issue["suspicious_pattern"],
                        "original_text_snippet": _snippet(cleaned[field], issue["start"], issue["end"]),
                        "suggested_repair_if_any": "",
                        "notes": "Replacement character is irreversible and no qualifying PubMed/project reference was available; do not guess.",
                    }
                )

        status = "manual_review_required" if remaining else ("auto_repaired" if changed else "valid_unicode_preserved")
        previews.append(
            {
                "review_id": review_id,
                "source_file": file,
                "row_index": index,
                "label": raw_row.get("label_included", raw_row.get("label", "")),
                "title_original": title_original,
                "abstract_original": abstract_original,
                "title_cleaned": cleaned["title"],
                "abstract_cleaned": cleaned["abstract"],
                "repair_status": status,
                "remaining_issue_count": len(remaining),
            }
        )

    repair_logs.sort(key=lambda row: (row["source_file"], int(row["row_index"]), row["field"], row["action"]))
    manual_rows.sort(key=lambda row: (row["source_file"], int(row["row_index"]), row["field"]))
    _write_rows(REPAIRS_PATH, REPAIR_COLUMNS, repair_logs)
    _write_rows(MANUAL_PATH, MANUAL_COLUMNS, manual_rows)
    _write_rows(PREVIEW_PATH, PREVIEW_COLUMNS, previews)

    hashes_after = {path: _sha256(path) for path in source_files}
    if hashes_before != hashes_after:
        raise RuntimeError("A raw CSV changed during label=1 mojibake processing")

    auto_records = {(row["source_file"], int(row["row_index"])) for row in repair_logs if row["action"] == "auto_repaired"}
    manual_records = {(row["source_file"], int(row["row_index"])) for row in repair_logs if row["action"] == "manual_review_required"}
    field_counts = Counter(row["field"] for row in candidate_output)
    pattern_counts = Counter(row["suspicious_pattern"] for row in base_logs)
    review_counts = Counter(row.get("review_key", Path(file).stem) for (file, _), row in candidates.items())
    summary = {
        "candidate_records": len(candidate_keys),
        "candidate_field_rows": len(candidate_output),
        "auto_repaired_records": len(auto_records),
        "manual_review_records": len(manual_records),
        "valid_unicode_preserved_records": len(normal_unicode_records - auto_records - manual_records),
        "auto_repaired_occurrences": sum(row["action"] == "auto_repaired" for row in repair_logs),
        "manual_review_occurrences": sum(row["action"] == "manual_review_required" for row in repair_logs),
        "suspicious_patterns": dict(sorted(pattern_counts.items())),
        "affected_fields": dict(sorted(field_counts.items())),
        "records_by_review": dict(review_counts.most_common()),
        "raw_files_unchanged": True,
        "pubmed_cache_sha256": _sha256(PUBMED_CACHE),
        "candidates_path": str(CANDIDATES_PATH),
        "repairs_path": str(REPAIRS_PATH),
        "manual_review_path": str(MANUAL_PATH),
        "cleaned_preview_path": str(PREVIEW_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_label1_mojibake_qc(), ensure_ascii=True, indent=2))
