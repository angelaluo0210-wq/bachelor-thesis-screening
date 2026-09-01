"""Dataset validation, review statistics, and proposed review-level split.

This module reads the raw review CSV files in the project without modifying them.
It performs schema validation, missing-value checks, duplicate detection, and
review-level summaries. It does not train a model or use the test split for
selection.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

try:
    from .mojibake_qc import print_mojibake_summary, run_mojibake_qc
except ImportError:  # Support direct execution: python src/prepare_dataset.py
    from mojibake_qc import print_mojibake_summary, run_mojibake_qc

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "数据" / "review整理"
REPORT_DIR = ROOT / "reports"
CONFIG_PATH = ROOT / "config" / "review_split.json"

VALID_LABELS = {0, 1}


def canonicalize_column_name(column_name: Any) -> str:
    """Normalize column names for cross-review schema checks."""
    if column_name is None:
        return ""
    value = str(column_name).strip().lower()
    value = value.replace("-", "_")
    value = value.replace(" ", "_")
    value = re.sub(r"_+", "_", value)
    return value


def normalize_text(value: Any) -> str:
    """Normalize text for duplicate detection and title/abstract comparisons."""
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def normalize_review_key(value: Any) -> str:
    """Normalize review identifiers conservatively for exact/near-exact matching."""
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = text.replace(" ", "_")
    text = text.replace("-", "_")
    text = re.sub(r"_+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "", text)
    return text.strip("_")


def load_eligibility_criteria(path: Path | None = None) -> pd.DataFrame:
    """Load the eligibility-criteria workbook without modifying the source Excel file."""
    workbook_path = path or ROOT / "数据" / "eligibility_criteria(排纳标准).xlsx"
    if not workbook_path.exists():
        raise FileNotFoundError(f"Eligibility criteria workbook not found: {workbook_path}")

    df = pd.read_excel(workbook_path)
    df = df.copy()
    df.columns = [canonicalize_column_name(col) for col in df.columns]
    rename_map = {
        "review_key": "review_key",
        "review_doi": "review_doi",
        "dataset": "dataset",
        "collection_doi": "collection_doi",
        "eligibility_criteria": "eligibility_criteria",
        "inclusion_criteria": "inclusion_criteria",
        "exclusion_criteria": "exclusion_criteria",
        "needs_manual_review": "needs_manual_review",
    }
    for old_name, new_name in rename_map.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
    if "review_key" not in df.columns:
        raise ValueError(f"Workbook {workbook_path} does not contain a review_key column.")

    df["review_key"] = df["review_key"].astype(str)
    df["review_key_normalized"] = df["review_key"].map(normalize_review_key)
    df["eligibility_criteria_available"] = df["eligibility_criteria"].notna() & df["eligibility_criteria"].astype(str).str.strip().ne("")
    df["inclusion_criteria_available"] = df["inclusion_criteria"].notna() & df["inclusion_criteria"].astype(str).str.strip().ne("")
    df["exclusion_criteria_available"] = df["exclusion_criteria"].notna() & df["exclusion_criteria"].astype(str).str.strip().ne("")
    return df


def match_criteria_to_reviews(article_df: pd.DataFrame, criteria_df: pd.DataFrame) -> pd.DataFrame:
    """Match raw review IDs to the eligibility workbook conservatively and report status."""
    review_ids = pd.Series(sorted(article_df["review_id"].dropna().astype(str).unique().tolist()))
    review_ids = pd.DataFrame({"review_id_article_data": review_ids})
    review_ids["review_id_article_data_normalized"] = review_ids["review_id_article_data"].map(normalize_review_key)

    criteria_lookup = criteria_df[["review_key", "review_key_normalized", "eligibility_criteria_available", "inclusion_criteria_available", "exclusion_criteria_available"]].copy()
    criteria_lookup["review_key"] = criteria_lookup["review_key"].astype(str)

    match_rows = []
    for _, row in review_ids.iterrows():
        review_id = row["review_id_article_data"]
        normalized = row["review_id_article_data_normalized"]
        exact = criteria_lookup[criteria_lookup["review_key"].astype(str) == review_id]
        if not exact.empty:
            match_entry = exact.iloc[0]
            status = "exact"
            excel_id = match_entry["review_key"]
            notes = "exact review key match"
        else:
            normalized_matches = criteria_lookup[criteria_lookup["review_key_normalized"] == normalized]
            if not normalized_matches.empty:
                match_entry = normalized_matches.iloc[0]
                status = "normalized"
                excel_id = match_entry["review_key"]
                notes = "normalized review key match"
            else:
                match_entry = None
                status = "unmatched"
                excel_id = None
                notes = "no exact or normalized review key match in eligibility workbook"

        inclusion_available = bool(match_entry["inclusion_criteria_available"]) if match_entry is not None else False
        exclusion_available = bool(match_entry["exclusion_criteria_available"]) if match_entry is not None else False
        eligibility_available = bool(match_entry["eligibility_criteria_available"]) if match_entry is not None else False

        match_rows.append({
            "review_id_article_data": review_id,
            "review_id_excel": excel_id,
            "match_status": status,
            "eligibility_criteria_available": eligibility_available,
            "inclusion_criteria_available": inclusion_available,
            "exclusion_criteria_available": exclusion_available,
            "notes": notes,
        })

    return pd.DataFrame(match_rows)


def read_review_csv(path: Path) -> pd.DataFrame:
    """Read each review CSV with a robust encoding fallback without altering source files."""
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, engine="python", on_bad_lines="skip")
        except Exception:
            continue
    return pd.read_csv(path, encoding="utf-8", engine="python", on_bad_lines="skip", encoding_errors="replace")


def infer_review_id(df: pd.DataFrame, path: Path) -> pd.Series:
    """Choose the review id field if present, otherwise fallback to the filename."""
    for candidate in ["review_id", "review_key", "review", "study_id", "id"]:
        if candidate in df.columns:
            return df[candidate].astype(str)
    return pd.Series([path.stem] * len(df), index=df.index, dtype="string")


def infer_label_column(df: pd.DataFrame) -> str | None:
    """Infer the screening label column."""
    for alias in ["label_included", "label", "included", "decision", "screening_label"]:
        if alias in df.columns:
            return alias
    for col in df.columns:
        if "label" in str(col).lower():
            return col
    return None


def infer_title_column(df: pd.DataFrame) -> str | None:
    """Infer the article title column."""
    for alias in ["title", "paper_title", "article_title", "study_title"]:
        if alias in df.columns:
            return alias
    for col in df.columns:
        if "title" in str(col).lower():
            return col
    return None


def infer_abstract_column(df: pd.DataFrame) -> str | None:
    """Infer the abstract column."""
    for alias in ["abstract", "article_abstract", "summary", "description"]:
        if alias in df.columns:
            return alias
    for col in df.columns:
        if "abstract" in str(col).lower():
            return col
    return None


def infer_identifier_columns(df: pd.DataFrame) -> Dict[str, str | None]:
    """Map article identifier columns to canonical names."""
    mapping = {"doi": None, "pmid": None, "openalex_id": None}
    alias_map = {
        "doi": ["doi", "article_doi"],
        "pmid": ["pmid", "pubmed_id", "pubmed"],
        "openalex_id": ["openalex_id", "openalex", "openalexid"],
    }
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            if alias in df.columns:
                mapping[canonical] = alias
                break
    return mapping


def infer_criteria_columns(df: pd.DataFrame) -> Dict[str, str | None]:
    """Look for inclusion/exclusion/eligibility criteria columns if present."""
    fields = {"inclusion_criteria": None, "exclusion_criteria": None, "eligibility_criteria": None}
    alias_map = {
        "inclusion_criteria": ["inclusion_criteria", "inclusion_criterion", "inclusion", "criteria_inclusion"],
        "exclusion_criteria": ["exclusion_criteria", "exclusion_criterion", "exclusion", "criteria_exclusion"],
        "eligibility_criteria": ["eligibility_criteria", "eligibility_criterion", "criteria", "selection_criteria"],
    }
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            if alias in df.columns:
                fields[canonical] = alias
                break
    return fields


def load_data() -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Load the raw review CSVs and normalize them into a single combined dataframe."""
    frames: List[pd.DataFrame] = []
    metadata: List[Dict[str, Any]] = []
    review_files = sorted(DATA_DIR.glob("*.csv"))

    if not review_files:
        raise FileNotFoundError(f"No review CSV files found in {DATA_DIR}")

    for path in review_files:
        df = read_review_csv(path)
        df.columns = [canonicalize_column_name(col) for col in df.columns]
        df["review_id"] = infer_review_id(df, path)

        label_col = infer_label_column(df)
        title_col = infer_title_column(df)
        abstract_col = infer_abstract_column(df)
        id_cols = infer_identifier_columns(df)
        criteria_cols = infer_criteria_columns(df)

        df["label"] = df[label_col] if label_col else pd.NA
        df["title"] = df[title_col] if title_col else pd.NA
        df["abstract"] = df[abstract_col] if abstract_col else pd.NA

        for canonical, source in id_cols.items():
            df[canonical] = df[source] if source else pd.NA

        for canonical, source in criteria_cols.items():
            df[canonical] = df[source] if source else pd.NA

        keep = {"review_id", "title", "abstract", "label", "doi", "pmid", "openalex_id", "inclusion_criteria", "exclusion_criteria", "eligibility_criteria"}
        for col in list(df.columns):
            if col not in keep:
                df = df.drop(columns=[col])

        metadata.append({
            "file": path.name,
            "rows": len(df),
            "columns": list(df.columns),
            "label_column": label_col,
            "title_column": title_col,
            "abstract_column": abstract_col,
            "identifier_columns": id_cols,
            "criteria_columns": criteria_cols,
        })
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined, metadata


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Return row-level schema violations for required fields."""
    required = ["review_id", "title", "abstract", "label"]
    rows = []
    for idx, row in df.iterrows():
        missing = [field for field in required if pd.isna(row.get(field)) or (isinstance(row.get(field), str) and not row.get(field).strip())]
        if missing:
            rows.append({"row_index": int(idx), "review_id": row.get("review_id"), "missing_fields": ";".join(missing)})
    return pd.DataFrame(rows)


def validate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Check label type and legal values (0/1)."""
    rows = []
    for idx, value in df["label"].items():
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(value) or (pd.notna(numeric) and numeric not in VALID_LABELS):
            rows.append({
                "row_index": int(idx),
                "review_id": df.loc[idx, "review_id"],
                "raw_label": value,
                "converted_label": numeric,
            })
    return pd.DataFrame(rows)


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Compute row-level missing value flags without deleting rows."""
    title_missing = df["title"].isna() | df["title"].astype(str).str.strip().eq("")
    abstract_missing = df["abstract"].isna() | df["abstract"].astype(str).str.strip().eq("")
    doi_missing = df["doi"].isna() | df["doi"].astype(str).str.strip().eq("")
    pmid_missing = df["pmid"].isna() | df["pmid"].astype(str).str.strip().eq("")
    openalex_missing = df["openalex_id"].isna() | df["openalex_id"].astype(str).str.strip().eq("")
    id_missing = doi_missing & pmid_missing & openalex_missing
    criteria_missing = (
        (df["inclusion_criteria"].isna() | df["inclusion_criteria"].astype(str).str.strip().eq(""))
        & (df["exclusion_criteria"].isna() | df["exclusion_criteria"].astype(str).str.strip().eq(""))
        & (df["eligibility_criteria"].isna() | df["eligibility_criteria"].astype(str).str.strip().eq(""))
    )

    result = pd.DataFrame({
        "row_index": df.index,
        "review_id": df["review_id"],
        "title_missing": title_missing.astype(bool),
        "abstract_missing": abstract_missing.astype(bool),
        "identifier_missing": id_missing.astype(bool),
        "eligibility_missing": criteria_missing.astype(bool),
        "title_and_abstract_missing": (title_missing & abstract_missing).astype(bool),
    })
    return result


def find_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Find duplicates by PMID, DOI, OpenAlex ID, and normalized title."""
    duplicate_rows = []
    duplicate_groups = []

    for identifier_name in ["pmid", "doi", "openalex_id"]:
        subset = df[["review_id", "title", "abstract", "label", identifier_name]].copy()
        subset = subset[subset[identifier_name].notna() & subset[identifier_name].astype(str).str.strip().ne("")]
        if subset.empty:
            continue
        key = ["review_id", identifier_name]
        subset[identifier_name] = subset[identifier_name].astype(str).str.strip()
        groups = subset.groupby(key, dropna=False)
        for (review_id, identifier_value), group in groups:
            if len(group) > 1:
                duplicate_groups.append({
                    "review_id": review_id,
                    "identifier_type": identifier_name,
                    "identifier_value": identifier_value,
                    "duplicate_count": int(len(group)),
                    "labels": list(group["label"].astype(str).tolist()),
                })
                for _, row in group.iterrows():
                    duplicate_rows.append({
                        "review_id": row["review_id"],
                        "identifier_type": identifier_name,
                        "identifier_value": row[identifier_name],
                        "title": row["title"],
                        "abstract": row["abstract"],
                        "label": row["label"],
                        "duplicate_count": int(len(group)),
                    })

    title_subset = df[["review_id", "title", "abstract", "label"]].copy()
    title_subset = title_subset[title_subset["title"].notna() & title_subset["title"].astype(str).str.strip().ne("")]
    title_subset["normalized_title"] = title_subset["title"].map(normalize_text)
    title_subset = title_subset[title_subset["normalized_title"].ne("")]
    for (review_id, normalized_title), group in title_subset.groupby(["review_id", "normalized_title"], dropna=False):
        if len(group) > 1:
            duplicate_groups.append({
                "review_id": review_id,
                "identifier_type": "normalized_title",
                "identifier_value": normalized_title,
                "duplicate_count": int(len(group)),
                "labels": list(group["label"].astype(str).tolist()),
            })
            for _, row in group.iterrows():
                duplicate_rows.append({
                    "review_id": row["review_id"],
                    "identifier_type": "normalized_title",
                    "identifier_value": normalized_title,
                    "title": row["title"],
                    "abstract": row["abstract"],
                    "label": row["label"],
                    "duplicate_count": int(len(group)),
                })

    duplicate_rows_df = pd.DataFrame(duplicate_rows)
    duplicate_groups_df = pd.DataFrame(duplicate_groups)
    return duplicate_rows_df, duplicate_groups_df


def find_conflicting_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Report duplicate identifiers or normalized titles with mixed labels."""
    rows = []
    for identifier_name in ["pmid", "doi", "openalex_id"]:
        subset = df[["review_id", "title", "abstract", "label", identifier_name]].copy()
        subset = subset[subset[identifier_name].notna() & subset[identifier_name].astype(str).str.strip().ne("")]
        if subset.empty:
            continue
        subset[identifier_name] = subset[identifier_name].astype(str).str.strip()
        for (review_id, identifier_value), group in subset.groupby(["review_id", identifier_name], dropna=False):
            labels = pd.to_numeric(group["label"], errors="coerce").dropna().unique()
            if len(labels) > 1:
                rows.append({
                    "review_id": review_id,
                    "identifier_type": identifier_name,
                    "identifier_value": identifier_value,
                    "labels": sorted([int(v) for v in labels.tolist()]),
                    "duplicate_rows": int(len(group)),
                    "title": group["title"].iloc[0],
                    "abstract": group["abstract"].iloc[0],
                })

    title_subset = df[["review_id", "title", "abstract", "label"]].copy()
    title_subset = title_subset[title_subset["title"].notna() & title_subset["title"].astype(str).str.strip().ne("")]
    title_subset["normalized_title"] = title_subset["title"].map(normalize_text)
    for (review_id, normalized_title), group in title_subset.groupby(["review_id", "normalized_title"], dropna=False):
        labels = pd.to_numeric(group["label"], errors="coerce").dropna().unique()
        if len(labels) > 1:
            rows.append({
                "review_id": review_id,
                "identifier_type": "normalized_title",
                "identifier_value": normalized_title,
                "labels": sorted([int(v) for v in labels.tolist()]),
                "duplicate_rows": int(len(group)),
                "title": group["title"].iloc[0],
                "abstract": group["abstract"].iloc[0],
            })

    return pd.DataFrame(rows)


def check_text_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight quality checks for empty text and suspiciously short content."""
    issues = []
    for idx, row in df.iterrows():
        title = str(row["title"]) if pd.notna(row["title"]) else ""
        abstract = str(row["abstract"]) if pd.notna(row["abstract"]) else ""
        if len(title.strip()) < 5 or len(abstract.strip()) < 20:
            issues.append({
                "row_index": int(idx),
                "review_id": row["review_id"],
                "title_length": len(title.strip()),
                "abstract_length": len(abstract.strip()),
                "title": title[:200],
                "abstract": abstract[:200],
            })
        if title.strip() and abstract.strip() and normalize_text(title) == normalize_text(abstract):
            issues.append({
                "row_index": int(idx),
                "review_id": row["review_id"],
                "reason": "identical_title_and_abstract",
                "title": title[:200],
                "abstract": abstract[:200],
            })
    return pd.DataFrame(issues)


def compute_review_statistics(df: pd.DataFrame, duplicate_rows: pd.DataFrame, conflicting_rows: pd.DataFrame, missing_rows: pd.DataFrame, criteria_match_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute review-level prevalence and QC counts."""
    review_stats = (
        df.groupby("review_id", dropna=False)
        .agg(
            total_articles=("review_id", "size"),
            included_articles=("label", lambda s: int((pd.to_numeric(s, errors="coerce") == 1).sum())),
            excluded_articles=("label", lambda s: int((pd.to_numeric(s, errors="coerce") == 0).sum())),
            missing_title=("title", lambda s: int((s.isna() | s.astype(str).str.strip().eq("")).sum())),
            missing_abstract=("abstract", lambda s: int((s.isna() | s.astype(str).str.strip().eq("")).sum())),
        )
        .reset_index()
    )
    review_stats["positive_prevalence"] = review_stats["included_articles"] / review_stats["total_articles"]
    review_stats["duplicate_count"] = 0
    review_stats["conflicting_label_count"] = 0
    review_stats["eligibility_criteria_available"] = False
    review_stats["inclusion_criteria_available"] = False
    review_stats["exclusion_criteria_available"] = False
    review_stats["criteria_match_status"] = "unmatched"

    if not duplicate_rows.empty:
        review_stats["duplicate_count"] = review_stats["review_id"].map(duplicate_rows.groupby("review_id").size()).fillna(0).astype(int)
    if not conflicting_rows.empty:
        review_stats["conflicting_label_count"] = review_stats["review_id"].map(conflicting_rows.groupby("review_id").size()).fillna(0).astype(int)
    if not missing_rows.empty:
        review_stats["missing_title"] = review_stats["review_id"].map(missing_rows.groupby("review_id")["title_missing"].sum()).fillna(0).astype(int)
        review_stats["missing_abstract"] = review_stats["review_id"].map(missing_rows.groupby("review_id")["abstract_missing"].sum()).fillna(0).astype(int)

    if criteria_match_df is not None and not criteria_match_df.empty:
        match_mapping = criteria_match_df.set_index("review_id_article_data")
        review_stats["criteria_match_status"] = review_stats["review_id"].map(match_mapping["match_status"]).fillna("unmatched")
        review_stats["eligibility_criteria_available"] = review_stats["review_id"].map(match_mapping["eligibility_criteria_available"]).fillna(False).astype(bool)
        review_stats["inclusion_criteria_available"] = review_stats["review_id"].map(match_mapping["inclusion_criteria_available"]).fillna(False).astype(bool)
        review_stats["exclusion_criteria_available"] = review_stats["review_id"].map(match_mapping["exclusion_criteria_available"]).fillna(False).astype(bool)

    review_stats["review_title"] = review_stats["review_id"]
    review_stats["review_doi"] = None
    review_stats["dataset_name"] = "review整理"
    review_stats = review_stats[[
        "review_id", "total_articles", "included_articles", "excluded_articles", "positive_prevalence",
        "missing_title", "missing_abstract", "duplicate_count", "conflicting_label_count",
        "eligibility_criteria_available", "inclusion_criteria_available", "exclusion_criteria_available",
        "criteria_match_status", "review_title", "review_doi", "dataset_name"
    ]]
    return review_stats


def propose_review_split(stats: pd.DataFrame) -> pd.DataFrame:
    """Create a deterministic proposed 18/4/4 review-level split, without finalizing it."""
    if len(stats) < 26:
        raise ValueError("Expected at least 26 reviews for the requested 18/4/4 split.")

    ordered = stats.sort_values(["total_articles", "positive_prevalence"], ascending=[False, True]).reset_index(drop=True)
    test_n = 4
    val_n = 4
    test_candidates = ordered[ordered["total_articles"] >= ordered["total_articles"].median()].copy()
    if len(test_candidates) < test_n:
        test_candidates = ordered.copy()
    test_reviews = test_candidates.sort_values(["positive_prevalence", "total_articles"], ascending=[True, False]).head(test_n)["review_id"].tolist()

    remaining = ordered[~ordered["review_id"].isin(test_reviews)].copy()
    val_candidates = remaining[remaining["total_articles"] >= remaining["total_articles"].median()].copy()
    if len(val_candidates) < val_n:
        val_candidates = remaining.copy()
    validation_reviews = val_candidates.sort_values(["positive_prevalence", "total_articles"], ascending=[False, True]).head(val_n)["review_id"].tolist()

    all_assigned = set(test_reviews) | set(validation_reviews)
    train_reviews = [r for r in ordered["review_id"].tolist() if r not in all_assigned]

    split_map = {}
    for review_id in test_reviews:
        split_map[review_id] = "test"
    for review_id in validation_reviews:
        split_map[review_id] = "validation"
    for review_id in train_reviews:
        split_map[review_id] = "train"

    split_df = ordered[["review_id", "total_articles", "included_articles", "excluded_articles", "positive_prevalence"]].copy()
    split_df["split"] = split_df["review_id"].map(split_map)
    split_df = split_df[["review_id", "split", "total_articles", "included_articles", "excluded_articles", "positive_prevalence"]]
    return split_df


def save_reports(stats: pd.DataFrame, missing_rows: pd.DataFrame, duplicate_rows: pd.DataFrame, conflicting_rows: pd.DataFrame, invalid_rows: pd.DataFrame, split_df: pd.DataFrame, criteria_match_df: pd.DataFrame | None = None) -> None:
    """Persist all QC tables and the proposed split to the reports directory."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stats.to_csv(REPORT_DIR / "review_statistics.csv", index=False)

    missing_rows.to_csv(REPORT_DIR / "missing_values.csv", index=False)
    duplicate_rows.to_csv(REPORT_DIR / "duplicate_articles.csv", index=False)
    conflicting_rows.to_csv(REPORT_DIR / "conflicting_labels.csv", index=False)
    invalid_rows.to_csv(REPORT_DIR / "invalid_labels.csv", index=False)
    split_df.to_csv(REPORT_DIR / "proposed_review_split.csv", index=False)
    if criteria_match_df is not None:
        criteria_match_df.to_csv(REPORT_DIR / "criteria_matching_report.csv", index=False)


def print_summary(stats: pd.DataFrame, missing_rows: pd.DataFrame, duplicate_rows: pd.DataFrame, conflicting_rows: pd.DataFrame, invalid_rows: pd.DataFrame, criteria_match_df: pd.DataFrame | None = None) -> None:
    """Display the dataset-level summary in the terminal."""
    print("\n=== Dataset QC summary ===")
    print(f"Total reviews: {len(stats)}")
    print(f"Total articles: {int(stats['total_articles'].sum())}")
    print(f"Included articles: {int(stats['included_articles'].sum())}")
    print(f"Excluded articles: {int(stats['excluded_articles'].sum())}")
    print(f"Missing titles: {int(missing_rows['title_missing'].sum())}")
    print(f"Missing abstracts: {int(missing_rows['abstract_missing'].sum())}")
    print(f"Duplicates: {len(duplicate_rows)}")
    print(f"Conflicting labels: {len(conflicting_rows)}")
    print(f"Invalid labels: {len(invalid_rows)}")
    if criteria_match_df is not None and not criteria_match_df.empty:
        matched = int((criteria_match_df["match_status"] != "unmatched").sum())
        total = len(criteria_match_df)
        print(f"Criteria matches: {matched}/{total}")
        print(f"Criteria available: {int(criteria_match_df['eligibility_criteria_available'].sum())}/{total}")
    print("\nPer-review counts:")
    print(stats[["review_id", "total_articles", "included_articles", "excluded_articles", "positive_prevalence"]].sort_values("total_articles", ascending=False).to_string(index=False))


def ensure_split_config() -> None:
    """Create a stub config file for future fixed split export."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps({"train": [], "validation": [], "test": []}, indent=2), encoding="utf-8")


def main() -> None:
    """Run the dataset validation and review-level preprocessing pipeline."""
    ensure_split_config()
    mojibake_summary = run_mojibake_qc()
    df, metadata = load_data()
    criteria_df = load_eligibility_criteria()
    criteria_match_df = match_criteria_to_reviews(df, criteria_df)

    schema_issues = validate_schema(df)
    invalid_labels = validate_labels(df)
    missing_rows = check_missing_values(df)
    duplicate_rows, duplicate_groups = find_duplicates(df)
    conflicting_rows = find_conflicting_labels(df)
    text_quality_issues = check_text_quality(df)
    review_stats = compute_review_statistics(df, duplicate_rows, conflicting_rows, missing_rows, criteria_match_df)
    split_df = propose_review_split(review_stats)

    save_reports(review_stats, missing_rows, duplicate_rows, conflicting_rows, invalid_labels, split_df, criteria_match_df)
    print_summary(review_stats, missing_rows, duplicate_rows, conflicting_rows, invalid_labels, criteria_match_df)
    print_mojibake_summary(mojibake_summary)

    print("\n=== Criteria workbook report ===")
    print(f"Workbook sheet names: {pd.ExcelFile(ROOT / '数据' / 'eligibility_criteria(排纳标准).xlsx').sheet_names}")
    print(f"Criteria rows: {len(criteria_df)}")
    print(f"Criteria coverage: {(criteria_match_df['match_status'] != 'unmatched').sum()}/{len(criteria_match_df)}")
    print(criteria_match_df[['review_id_article_data', 'review_id_excel', 'match_status', 'eligibility_criteria_available', 'inclusion_criteria_available', 'exclusion_criteria_available']].head().to_string(index=False))

    print("\n=== Schema observations ===")
    print(f"Observed review files: {len(metadata)}")
    for item in metadata[:3]:
        print(item["file"], "=>", item["columns"])

    print("\nImportant data notes:")
    print("- review_id maps to the raw 'review_key' field.")
    print("- dataset has 'title', 'abstract', and 'label_included' fields.")
    print("- eligibility criteria are stored in the separate Excel workbook and matched conservatively by review key.")
    print("- the proposed split is still a draft and should be confirmed before finalization.")


if __name__ == "__main__":
    main()
