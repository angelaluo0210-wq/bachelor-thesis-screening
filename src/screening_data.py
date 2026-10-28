"""Lightweight shared data logic for screening experiments.

This module deliberately depends only on the CPU data-science stack.  It does
not import PyTorch, Transformers, PEFT, model weights, or GPU helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


REVIEW_COLUMN = "review_key"
TITLE_COLUMN = "title"
ABSTRACT_COLUMN = "abstract"
LABEL_COLUMN = "label_included"
IDENTIFIER_COLUMNS = ("pmid", "doi", "openalex_id")
VALID_LABELS = {0, 1}


def safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def canonicalize_column(value: Any) -> str:
    return "_".join(str(value).strip().lower().replace("-", " ").split())


def normalize_review_key(value: Any) -> str:
    text = safe_text(value).lower().replace("-", "_").replace(" ", "_")
    return "_".join(part for part in text.split("_") if part)


def read_split(path: Path, review_keys_only: bool = False) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    frame = pd.read_csv(path, usecols=[REVIEW_COLUMN] if review_keys_only else None, low_memory=False)
    if REVIEW_COLUMN not in frame.columns:
        raise ValueError(f"{path} is missing required column {REVIEW_COLUMN!r}")
    if not review_keys_only:
        missing = sorted({REVIEW_COLUMN, TITLE_COLUMN, ABSTRACT_COLUMN, LABEL_COLUMN} - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
    return frame


def load_splits(split_dir: Path, evaluate_test: bool = False) -> dict[str, pd.DataFrame]:
    splits = {"train": read_split(split_dir / "train.csv"), "validation": read_split(split_dir / "validation.csv")}
    if evaluate_test:
        splits["test"] = read_split(split_dir / "test.csv")
    return splits


@dataclass(frozen=True)
class CriteriaSchema:
    sheet_name: str
    review_key: str
    inclusion: str | None
    exclusion: str | None
    eligibility: str | None
    review_text_columns: tuple[str, ...]


def _first_present(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    available = set(columns)
    return next((name for name in candidates if name in available), None)


def load_eligibility_criteria(path: Path) -> tuple[pd.DataFrame, CriteriaSchema]:
    if not path.exists():
        raise FileNotFoundError(f"Eligibility criteria workbook not found: {path}")
    workbook = pd.ExcelFile(path)
    candidates: list[tuple[str, pd.DataFrame]] = []
    for sheet_name in workbook.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet_name)
        frame = frame.rename(columns={column: canonicalize_column(column) for column in frame.columns})
        if _first_present(frame.columns, ("review_key", "review_id")):
            candidates.append((sheet_name, frame))
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one eligibility sheet with review_key/review_id; found {[name for name, _ in candidates]} among sheets {workbook.sheet_names}")
    sheet_name, criteria = candidates[0]
    key_column = _first_present(criteria.columns, ("review_key", "review_id"))
    assert key_column is not None
    inclusion = _first_present(criteria.columns, ("inclusion_criteria", "inclusion_criterion", "inclusion"))
    exclusion = _first_present(criteria.columns, ("exclusion_criteria", "exclusion_criterion", "exclusion"))
    eligibility = _first_present(criteria.columns, ("eligibility_criteria", "eligibility_criterion", "eligibility", "criteria"))
    if not any((inclusion, exclusion, eligibility)):
        raise ValueError(f"No criteria text columns found in workbook sheet {sheet_name!r}")
    review_text_columns = tuple(column for column in ("review_title", "review_question", "review_description", "systematic_review_title") if column in criteria.columns)
    criteria = criteria.copy()
    criteria["_normalized_review_key"] = criteria[key_column].map(normalize_review_key)
    if criteria["_normalized_review_key"].eq("").any():
        raise ValueError("Eligibility workbook contains an empty review key")
    duplicated = criteria.loc[criteria["_normalized_review_key"].duplicated(keep=False), key_column].astype(str).tolist()
    if duplicated:
        raise ValueError(f"Eligibility workbook has duplicate normalized review keys: {duplicated}")
    return criteria, CriteriaSchema(sheet_name, key_column, inclusion, exclusion, eligibility, review_text_columns)


def join_criteria(frame: pd.DataFrame, criteria: pd.DataFrame, schema: CriteriaSchema, split_name: str) -> pd.DataFrame:
    result = frame.copy()
    result["_normalized_review_key"] = result[REVIEW_COLUMN].map(normalize_review_key)
    lookup = criteria.set_index("_normalized_review_key", drop=False)
    unmatched_ids = sorted(result.loc[~result["_normalized_review_key"].isin(lookup.index), REVIEW_COLUMN].dropna().astype(str).unique())
    if unmatched_ids:
        raise ValueError(f"Unmatched eligibility criteria in {split_name}: {unmatched_ids}")

    def mapped_text(column: str | None) -> pd.Series:
        if column is None:
            return pd.Series("", index=result.index, dtype="string")
        return result["_normalized_review_key"].map(lookup[column].map(safe_text)).fillna("").map(safe_text)

    result["_inclusion_text"] = mapped_text(schema.inclusion)
    result["_exclusion_text"] = mapped_text(schema.exclusion)
    unified = mapped_text(schema.eligibility)
    neither_split = result["_inclusion_text"].eq("") & result["_exclusion_text"].eq("")
    result.loc[neither_split, "_inclusion_text"] = unified.loc[neither_split]
    result["_review_text"] = ""
    for column in schema.review_text_columns:
        values = mapped_text(column)
        result["_review_text"] = ["\n\n".join(part for part in (left, right) if part) for left, right in zip(result["_review_text"], values)]
    if (result["_inclusion_text"].eq("") & result["_exclusion_text"].eq("")).any():
        raise ValueError(f"Matched reviews have no usable criteria text in {split_name}")
    return result


def validate_labels(frame: pd.DataFrame, split_name: str) -> None:
    numeric = pd.to_numeric(frame[LABEL_COLUMN], errors="coerce")
    invalid = frame.loc[numeric.isna() | ~numeric.isin(VALID_LABELS), LABEL_COLUMN].unique().tolist()
    if invalid:
        raise ValueError(f"{split_name} contains labels outside {{0, 1}}: {invalid}")
    frame[LABEL_COLUMN] = numeric.astype(np.int64)
    if set(frame[LABEL_COLUMN].unique().tolist()) != VALID_LABELS:
        raise ValueError(f"{split_name} must contain both classes")


def validate_splits(splits: dict[str, pd.DataFrame], evaluate_test: bool = False) -> None:
    review_sets = {name: set(frame[REVIEW_COLUMN].dropna().astype(str)) for name, frame in splits.items()}
    pairs = [("train", "validation")]
    if evaluate_test:
        pairs.extend((("train", "test"), ("validation", "test")))
    for left, right in pairs:
        overlap = sorted(review_sets[left] & review_sets[right])
        if overlap:
            raise ValueError(f"Review leakage between {left} and {right}: {overlap}")
    validate_labels(splits["train"], "train")
    validate_labels(splits["validation"], "validation")
    if evaluate_test:
        validate_labels(splits["test"], "test")


def summarize_split(frame: pd.DataFrame, split_name: str) -> dict[str, Any]:
    labels = frame[LABEL_COLUMN]
    missing_title = frame[TITLE_COLUMN].isna() | frame[TITLE_COLUMN].astype(str).str.strip().eq("")
    missing_abstract = frame[ABSTRACT_COLUMN].isna() | frame[ABSTRACT_COLUMN].astype(str).str.strip().eq("")
    return {"split": split_name, "reviews": int(frame[REVIEW_COLUMN].nunique()), "rows": int(len(frame)),
            "positive": int((labels == 1).sum()), "negative": int((labels == 0).sum()),
            "positive_rate": float((labels == 1).mean()), "missing_title": int(missing_title.sum()),
            "missing_abstract": int(missing_abstract.sum()),
            "missing_abstract_positive": int((missing_abstract & labels.eq(1)).sum()),
            "missing_abstract_negative": int((missing_abstract & labels.eq(0)).sum())}


def apply_threshold(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (probabilities >= threshold).astype(np.int64)


def calculate_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = labels.astype(np.int64)
    predictions = apply_threshold(probabilities, threshold)
    tp, tn = int(((labels == 1) & (predictions == 1)).sum()), int(((labels == 0) & (predictions == 0)).sum())
    fp, fn = int(((labels == 0) & (predictions == 1)).sum()), int(((labels == 1) & (predictions == 0)).sum())
    def divide(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else float("nan")
    include_rate = divide(tp + fp, len(labels))
    metrics: dict[str, Any] = {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "recall": divide(tp, tp + fn),
        "precision": divide(tp, tp + fp), "f1": divide(2 * tp, 2 * tp + fp + fn), "fnr": divide(fn, fn + tp),
        "specificity": divide(tn, tn + fp), "npv": divide(tn, tn + fn), "predicted_include_rate": include_rate,
        "workload_reduction": 1.0 - include_rate, "accuracy": divide(tp + tn, len(labels)), "threshold": float(threshold),
        "pr_auc": float("nan"), "roc_auc": float("nan")}
    if len(np.unique(labels)) == 2:
        metrics["pr_auc"] = float(average_precision_score(labels, probabilities))
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
    return metrics
