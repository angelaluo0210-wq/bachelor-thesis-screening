"""Create the fixed, deterministic review-level train/validation/test split.

The source CSV is treated as immutable.  Every input row and column is copied
verbatim to exactly one output CSV; this module does not clean, deduplicate, or
repair records.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "all_reviews_cleaned.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "splits"

TRAIN_REVIEWS = {
    "Appenzeller-Herzog_2019",
    "Bos_2018",
    "Brouwer_2019",
    "Chou_2003",
    "Chou_2004",
    "Donners_2021",
    "Jeyaraman_2020",
    "Leenaars_2020",
    "Meijboom_2021",
    "Menon_2022",
    "Nelson_2002",
    "Oud_2018",
    "Sep_2021",
    "Smid_2020",
    "van_de_Schoot_2018",
    "van_der_Waal_2022",
    "Walker_2018",
    "Wolters_2018",
}

VALIDATION_REVIEWS = {
    "Hall_2012",
    "Moran_2021",
    "van_der_Valk_2021",
    "Wassenaar_2017",
}

TEST_REVIEWS = {
    "Leenaars_2019",
    "Muthu_2021",
    "Radjenovic_2013",
    "van_Dis_2020",
}

SPLIT_REVIEWS = {
    "train": TRAIN_REVIEWS,
    "validation": VALIDATION_REVIEWS,
    "test": TEST_REVIEWS,
}

REVIEW_COLUMN_ALIASES = ("review_id", "review", "dataset", "review_key")
LABEL_COLUMN_ALIASES = ("label", "label_included", "included")
TITLE_COLUMN_ALIASES = ("title", "paper_title", "article_title", "study_title")
ABSTRACT_COLUMN_ALIASES = ("abstract", "article_abstract")

README_RECORDS = 169_288
README_POSITIVES = 2_834


class DatasetSplitError(ValueError):
    """Raised when a strict split invariant is not satisfied."""


def identify_column(
    fieldnames: Sequence[str], aliases: Sequence[str], purpose: str, *, required: bool = True
) -> str | None:
    """Return one unambiguous schema column, or fail with a useful message."""
    normalized: dict[str, list[str]] = {}
    for column in fieldnames:
        normalized.setdefault(str(column).strip().lower(), []).append(column)
    matches = [actual for alias in aliases for actual in normalized.get(alias, [])]
    if len(matches) > 1:
        raise DatasetSplitError(
            f"Cannot uniquely identify {purpose} column; candidates present: {matches}"
        )
    if not matches:
        if required:
            raise DatasetSplitError(
                f"Cannot identify {purpose} column. Expected one of {list(aliases)}; "
                f"observed columns: {list(fieldnames)}"
            )
        return None
    return matches[0]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a UTF-8 CSV without normalizing any cell values."""
    if not path.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise DatasetSplitError(f"Input CSV has no header: {path}")
        rows = list(reader)
    malformed = [index + 2 for index, row in enumerate(rows) if None in row]
    if malformed:
        preview = malformed[:20]
        raise DatasetSplitError(
            "CSV rows contain more fields than the header at physical record rows "
            f"{preview}{' ...' if len(malformed) > len(preview) else ''}"
        )
    return fieldnames, rows


def _label_kind(value: object) -> str:
    """Classify labels while preserving all original values in output rows."""
    text = "" if value is None else str(value).strip()
    if text in {"1", "1.0"}:
        return "positive"
    if text in {"0", "0.0"}:
        return "negative"
    return "other"


def label_counts(rows: Iterable[Mapping[str, str]], label_column: str) -> Counter[str]:
    return Counter(_label_kind(row.get(label_column)) for row in rows)


def validate_split_definition() -> None:
    """Validate the hard-coded review lists before inspecting or writing data."""
    expected_sizes = {"train": 18, "validation": 4, "test": 4}
    for split_name, expected_size in expected_sizes.items():
        actual_size = len(SPLIT_REVIEWS[split_name])
        if actual_size != expected_size:
            raise DatasetSplitError(
                f"{split_name} definition has {actual_size} reviews; expected {expected_size}"
            )
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = SPLIT_REVIEWS[left] & SPLIT_REVIEWS[right]
        if overlap:
            raise DatasetSplitError(
                f"Review overlap between {left} and {right}: {sorted(overlap)}"
            )


def validate_review_coverage(rows: Sequence[Mapping[str, str]], review_column: str) -> None:
    """Require exact equality between input review values and fixed definitions."""
    input_reviews = {row.get(review_column, "") for row in rows}
    defined_reviews = set().union(*SPLIT_REVIEWS.values())
    unassigned = sorted(input_reviews - defined_reviews)
    missing = sorted(defined_reviews - input_reviews)
    if unassigned or missing:
        raise DatasetSplitError(
            "Review coverage check failed. "
            f"Unassigned input review values: {unassigned or 'none'}; "
            f"defined reviews absent from input: {missing or 'none'}"
        )


def create_review_level_split(
    rows: Sequence[dict[str, str]], review_column: str
) -> dict[str, list[dict[str, str]]]:
    """Assign every row by exact review identifier (never by row/random split)."""
    review_to_split = {
        review: split_name
        for split_name, reviews in SPLIT_REVIEWS.items()
        for review in reviews
    }
    result: dict[str, list[dict[str, str]]] = {name: [] for name in SPLIT_REVIEWS}
    for row in rows:
        review = row.get(review_column, "")
        try:
            result[review_to_split[review]].append(row)
        except KeyError as exc:
            raise DatasetSplitError(f"No split assignment for review value {review!r}") from exc
    return result


def _row_identity(row: Mapping[str, str], fieldnames: Sequence[str]) -> tuple[str, ...]:
    return tuple(row.get(column, "") for column in fieldnames)


def validate_split(
    original_rows: Sequence[Mapping[str, str]],
    splits: Mapping[str, Sequence[Mapping[str, str]]],
    fieldnames: Sequence[str],
    review_column: str,
    label_column: str,
) -> None:
    """Check leakage, row/identity preservation, and label preservation."""
    observed_reviews = {
        name: {row.get(review_column, "") for row in split_rows}
        for name, split_rows in splits.items()
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = observed_reviews[left] & observed_reviews[right]
        if overlap:
            raise DatasetSplitError(
                f"Review leakage between {left} and {right}: {sorted(overlap)}"
            )

    combined = [row for name in ("train", "validation", "test") for row in splits[name]]
    if len(combined) != len(original_rows):
        raise DatasetSplitError(
            f"Row count changed: input={len(original_rows)}, outputs={len(combined)}"
        )
    original_identities = Counter(_row_identity(row, fieldnames) for row in original_rows)
    split_identities = Counter(_row_identity(row, fieldnames) for row in combined)
    if original_identities != split_identities:
        raise DatasetSplitError("Row identity or duplicate multiplicity changed during split")

    original_labels = label_counts(original_rows, label_column)
    output_labels = label_counts(combined, label_column)
    if original_labels != output_labels:
        raise DatasetSplitError(
            f"Label counts changed: input={dict(original_labels)}, outputs={dict(output_labels)}"
        )


def _is_missing(value: object) -> bool:
    return value is None or not str(value).strip()


def build_split_summary(
    splits: Mapping[str, Sequence[Mapping[str, str]]],
    review_column: str,
    label_column: str,
    title_column: str | None,
    abstract_column: str | None,
) -> tuple[list[str], list[dict[str, object]]]:
    """Build one audit row per review without changing source records."""
    columns = [
        "review_id",
        "split",
        "n_records",
        "n_positive",
        "n_negative",
        "n_other_labels",
        "positive_rate",
        "has_missing_title",
        "n_missing_title",
        "has_missing_abstract",
        "n_missing_abstract",
        "positive_with_missing_abstract",
        "negative_with_missing_abstract",
    ]
    summary: list[dict[str, object]] = []
    for split_name in ("train", "validation", "test"):
        grouped: dict[str, list[Mapping[str, str]]] = {}
        for row in splits[split_name]:
            grouped.setdefault(row[review_column], []).append(row)
        for review in sorted(grouped):
            review_rows = grouped[review]
            counts = label_counts(review_rows, label_column)
            missing_titles = (
                sum(_is_missing(row.get(title_column)) for row in review_rows)
                if title_column
                else 0
            )
            missing_abstracts = (
                sum(_is_missing(row.get(abstract_column)) for row in review_rows)
                if abstract_column
                else 0
            )
            positive_missing_abstract = (
                sum(
                    _label_kind(row.get(label_column)) == "positive"
                    and _is_missing(row.get(abstract_column))
                    for row in review_rows
                )
                if abstract_column
                else 0
            )
            negative_missing_abstract = (
                sum(
                    _label_kind(row.get(label_column)) == "negative"
                    and _is_missing(row.get(abstract_column))
                    for row in review_rows
                )
                if abstract_column
                else 0
            )
            summary.append(
                {
                    "review_id": review,
                    "split": split_name,
                    "n_records": len(review_rows),
                    "n_positive": counts["positive"],
                    "n_negative": counts["negative"],
                    "n_other_labels": counts["other"],
                    "positive_rate": counts["positive"] / len(review_rows),
                    "has_missing_title": bool(missing_titles),
                    "n_missing_title": missing_titles,
                    "has_missing_abstract": bool(missing_abstracts),
                    "n_missing_abstract": missing_abstracts,
                    "positive_with_missing_abstract": positive_missing_abstract,
                    "negative_with_missing_abstract": negative_missing_abstract,
                }
            )
    return columns, summary


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs_atomically(
    output_dir: Path,
    fieldnames: Sequence[str],
    splits: Mapping[str, Sequence[Mapping[str, str]]],
    summary_columns: Sequence[str],
    summary_rows: Sequence[Mapping[str, object]],
    review_column: str,
    label_column: str,
) -> list[Path]:
    """Write and post-validate temporary files before replacing final outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "train": "train.csv",
        "validation": "validation.csv",
        "test": "test.csv",
    }
    with tempfile.TemporaryDirectory(prefix=".split_tmp_", dir=output_dir) as temp_name:
        temp_dir = Path(temp_name)
        for split_name, filename in filenames.items():
            _write_csv(temp_dir / filename, fieldnames, splits[split_name])
        _write_csv(temp_dir / "split_summary.csv", summary_columns, summary_rows)

        reread: dict[str, list[dict[str, str]]] = {}
        for split_name, filename in filenames.items():
            reread_fields, reread[split_name] = read_csv(temp_dir / filename)
            if reread_fields != list(fieldnames):
                raise DatasetSplitError(f"Column order changed in temporary {filename}")
        original = [row for name in ("train", "validation", "test") for row in splits[name]]
        validate_split(original, reread, fieldnames, review_column, label_column)

        created: list[Path] = []
        for filename in (*filenames.values(), "split_summary.csv"):
            destination = output_dir / filename
            os.replace(temp_dir / filename, destination)
            created.append(destination)
    return created


def _print_label_warning(rows: Sequence[Mapping[str, str]], label_column: str) -> None:
    raw_counts = Counter(str(row.get(label_column, "")) for row in rows)
    abnormal = {value: count for value, count in raw_counts.items() if _label_kind(value) == "other"}
    print(f"Detected label column: {label_column}")
    print(f"Label unique values and counts: {dict(sorted(raw_counts.items()))}")
    if abnormal:
        print(
            f"WARNING: non-binary label values are preserved unchanged: {abnormal}",
            file=sys.stderr,
        )


def print_overall_summary(
    rows: Sequence[Mapping[str, str]],
    splits: Mapping[str, Sequence[Mapping[str, str]]],
    label_column: str,
) -> None:
    total_counts = label_counts(rows, label_column)
    print("\n=== Dataset Split Summary ===")
    print("\nInput:")
    print(f"reviews: {sum(len(reviews) for reviews in SPLIT_REVIEWS.values())}")
    print(f"records: {len(rows)}")
    print(f"positive: {total_counts['positive']}")
    print(f"negative: {total_counts['negative']}")
    print(f"other_labels: {total_counts['other']}")

    for split_name in ("train", "validation", "test"):
        split_rows = splits[split_name]
        counts = label_counts(split_rows, label_column)
        positive_rate = counts["positive"] / len(split_rows) if split_rows else 0.0
        print(f"\n{split_name.capitalize()}:")
        print(f"reviews: {len(SPLIT_REVIEWS[split_name])}")
        print(f"records: {len(split_rows)}")
        print(f"positive: {counts['positive']}")
        print(f"negative: {counts['negative']}")
        print(f"other_labels: {counts['other']}")
        print(f"positive_rate: {positive_rate:.8f}")
        print(f"{split_name}_record_ratio: {len(split_rows) / len(rows):.8f}")
        positive_ratio = (
            counts["positive"] / total_counts["positive"] if total_counts["positive"] else 0.0
        )
        print(f"{split_name}_positive_ratio: {positive_ratio:.8f}")

    if len(rows) != README_RECORDS or total_counts["positive"] != README_POSITIVES:
        print(
            "\nNote: processed cleaned dataset differs from original SYNERGY README statistics.\n"
            "This is expected after QC / cleaning."
        )


def run(input_path: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    """Validate, split, write, post-validate, and report the processed dataset."""
    fieldnames, rows = read_csv(input_path)
    review_column = identify_column(fieldnames, REVIEW_COLUMN_ALIASES, "review")
    label_column = identify_column(fieldnames, LABEL_COLUMN_ALIASES, "label")
    title_column = identify_column(
        fieldnames, TITLE_COLUMN_ALIASES, "title", required=False
    )
    abstract_column = identify_column(
        fieldnames, ABSTRACT_COLUMN_ALIASES, "abstract", required=False
    )
    assert review_column is not None and label_column is not None

    print(f"Input CSV: {input_path}")
    print(f"Rows: {len(rows)}")
    print(f"Columns ({len(fieldnames)}): {fieldnames}")
    print(f"Detected review column: {review_column}")
    _print_label_warning(rows, label_column)

    validate_split_definition()
    validate_review_coverage(rows, review_column)
    splits = create_review_level_split(rows, review_column)
    validate_split(rows, splits, fieldnames, review_column, label_column)
    summary_columns, summary_rows = build_split_summary(
        splits, review_column, label_column, title_column, abstract_column
    )
    if len(summary_rows) != 26:
        raise DatasetSplitError(
            f"Split summary has {len(summary_rows)} review rows; expected 26"
        )
    created = write_outputs_atomically(
        output_dir,
        fieldnames,
        splits,
        summary_columns,
        summary_rows,
        review_column,
        label_column,
    )
    print_overall_summary(rows, splits, label_column)
    print("\nChecks:")
    print("- 26 reviews fully assigned: PASS")
    print("- no review overlap: PASS")
    print("- row count preserved: PASS")
    print("- row identity/duplicate multiplicity preserved: PASS")
    print("- positive count preserved: PASS")
    print("- negative count preserved: PASS")
    return created


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        created = run(args.input, args.output_dir)
    except (DatasetSplitError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("\nFiles created:")
    for path in created:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
