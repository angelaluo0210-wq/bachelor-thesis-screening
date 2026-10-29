"""Synthetic, no-network tests for the Validation positive-case selector."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts import run_qwen_positive_case_diagnostic as diagnostic


def _row(review: str, pmid: str, label: int) -> dict[str, object]:
    return {
        "review_key": review,
        "pmid": pmid,
        "doi": "",
        "openalex_id": "",
        "label_included": label,
        "title": "title",
        "abstract": "abstract",
    }


def test_positive_case_selector_is_stratified_seeded_and_label_conditioned():
    frame = pd.DataFrame([
        *[_row("A", f"a-{index}", 1) for index in range(4)],
        *[_row("B", f"b-{index}", 1) for index in range(4)],
        _row("A", "a-negative", 0), _row("B", "b-negative", 0),
    ])
    first = diagnostic._select_positive_cases(frame, per_review=3, sample_seed=202610)
    second = diagnostic._select_positive_cases(frame, per_review=3, sample_seed=202610)
    assert first["label_included"].eq(1).all()
    assert first.groupby("review_key").size().to_dict() == {"A": 3, "B": 3}
    assert first["pmid"].tolist() == second["pmid"].tolist()
    assert diagnostic.SELECTION_VERSION in "positive_case_diagnostic_v1"


def test_positive_case_selector_rejects_a_review_without_enough_positives():
    frame = pd.DataFrame([_row("A", "1", 1), _row("B", "2", 1), _row("B", "3", 1)])
    with pytest.raises(ValueError, match="need 2"):
        diagnostic._select_positive_cases(frame, per_review=2, sample_seed=202610)
