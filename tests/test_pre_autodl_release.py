"""Regression checks for the pre-AutoDL provenance and threshold utilities."""
import json
from pathlib import Path

import numpy as np

from scripts import calibrate_validation_threshold as calibration
from src.split_dataset import SPLIT_REVIEWS


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_split_matches_the_active_split_source_of_truth():
    canonical = json.loads((ROOT / "config" / "canonical_split.json").read_text(encoding="utf-8"))
    assert canonical["status"] == "canonical_active_split"
    assert canonical["review_membership"] == {
        name: sorted(reviews) for name, reviews in SPLIT_REVIEWS.items()
    }
    all_reviews = [review for reviews in canonical["review_membership"].values() for review in reviews]
    assert len(all_reviews) == 26
    assert len(set(all_reviews)) == 26


def test_threshold_metrics_and_selection_rule_can_support_a_researcher_target():
    labels = np.array([1, 1, 0, 0])
    probabilities = np.array([0.95, 0.60, 0.70, 0.20])
    candidates = [calibration.metrics(labels, probabilities, threshold) for threshold in [0.0, 0.60, 0.70, 0.95]]
    feasible = [result for result in candidates if result["recall"] >= 1.0]
    selected = max(feasible, key=lambda result: result["threshold"])
    assert selected["threshold"] == 0.60
    assert selected["recall"] == 1.0
    assert selected["fnr"] == 0.0
