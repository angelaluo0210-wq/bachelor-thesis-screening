"""Non-destructive validation for a future frozen-prediction join."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Any

import pandas as pd

from src import screening_data as data


@dataclass(frozen=True)
class AlignmentReport:
    status: str
    matched: int
    source_only: int
    prediction_only: int
    duplicate_source_keys: tuple[str, ...]
    duplicate_prediction_keys: tuple[str, ...]
    label_mismatches: int | None


def _keys(frame: pd.DataFrame, review_column: str) -> list[str]:
    missing = {review_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing review identifier columns: {sorted(missing)}")
    keys = []
    for index, row in frame.reset_index(drop=True).iterrows():
        review = data.normalize_review_key(row[review_column])
        if not review:
            raise ValueError(f"Empty review identifier at row {index}")
        values = [f"{name}={data.safe_text(row[name]).lower()}" for name in data.IDENTIFIER_COLUMNS if name in frame.columns and data.safe_text(row[name])]
        if not values:
            raise ValueError(f"No stable article identifier at row {index}; title-based joining is forbidden")
        keys.append(review + "|" + "|".join(values))
    return keys


def validate_one_to_one_alignment(source: pd.DataFrame, predictions: pd.DataFrame) -> AlignmentReport:
    """Compare review-plus-identifier sets; never reorder or repair either input."""
    source_keys = _keys(source, data.REVIEW_COLUMN)
    prediction_review = "review_name" if "review_name" in predictions.columns else data.REVIEW_COLUMN
    prediction_keys = _keys(predictions, prediction_review)
    source_set, prediction_set = set(source_keys), set(prediction_keys)
    source_duplicates = tuple(sorted(key for key, count in Counter(source_keys).items() if count > 1))
    prediction_duplicates = tuple(sorted(key for key, count in Counter(prediction_keys).items() if count > 1))
    label_mismatches: int | None = None
    if data.LABEL_COLUMN in source.columns and "true_label" in predictions.columns:
        if not source_duplicates and not prediction_duplicates:
            source_labels = dict(zip(source_keys, source[data.LABEL_COLUMN]))
            prediction_labels = dict(zip(prediction_keys, predictions["true_label"]))
            shared = source_set & prediction_set
            label_mismatches = sum(source_labels[key] != prediction_labels[key] for key in shared)
    mismatch = source_duplicates or prediction_duplicates or source_set != prediction_set or label_mismatches not in (None, 0)
    status = "verified_synthetic" if not mismatch else "mismatch"
    return AlignmentReport(status, len(source_set & prediction_set), len(source_set - prediction_set),
                           len(prediction_set - source_set), source_duplicates, prediction_duplicates, label_mismatches)
