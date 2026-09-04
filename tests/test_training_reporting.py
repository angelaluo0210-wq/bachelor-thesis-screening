import logging

import numpy as np
import pandas as pd

from src.train_classifier_head import (
    calculate_metrics,
    calculate_per_review_metrics,
    load_splits,
    parse_args,
    prediction_frame,
)


def test_shared_training_defaults_are_fixed():
    args = parse_args([])

    assert args.learning_rate == 1e-4
    assert args.batch_size == 32
    assert args.epochs == 5
    assert args.max_length == 512
    assert args.dropout == 0.1
    assert args.seed == 42
    assert args.threshold == 0.5
    assert args.evaluate_test is False


def test_metrics_use_average_precision_and_nan_for_single_class(caplog):
    labels = np.array([0, 1, 0, 1])
    probabilities = np.array([0.1, 0.8, 0.4, 0.7])
    metrics = calculate_metrics(labels, probabilities, 0.5)

    assert metrics["pr_auc"] == 1.0
    assert metrics["roc_auc"] == 1.0

    with caplog.at_level(logging.WARNING):
        single_class = calculate_metrics(
            np.array([0, 0]), np.array([0.2, 0.3]), 0.5, context="Review A"
        )
    assert np.isnan(single_class["pr_auc"])
    assert np.isnan(single_class["roc_auc"])
    assert np.isnan(single_class["recall"])
    assert np.isnan(single_class["fnr"])
    assert "Review A" in caplog.text
    assert "recording NaN" in caplog.text


def test_per_review_metrics_and_prediction_schema():
    source = pd.DataFrame(
        {
            "review_key": ["A", "A", "B", "B"],
            "title": ["a", "b", "c", "d"],
            "pmid": ["1", "2", "3", "4"],
        }
    )
    labels = np.array([0, 1, 0, 0])
    probabilities = np.array([0.1, 0.9, 0.7, 0.2])
    indices = np.arange(4)

    reviews = calculate_per_review_metrics(source, probabilities, labels, indices, 0.5)
    assert [row["review_name"] for row in reviews] == ["A", "B"]
    assert reviews[0]["n_positive"] == 1
    assert reviews[0]["tp"] == 1
    assert reviews[1]["n_positive"] == 0
    assert np.isnan(reviews[1]["recall"])

    predictions = prediction_frame(source, probabilities, labels, indices, 0.5)
    assert {"review_name", "true_label", "predicted_probability", "pmid"}.issubset(
        predictions.columns
    )


def test_load_splits_does_not_open_test_without_explicit_flag(tmp_path, monkeypatch):
    calls = []

    def fake_read_split(path, review_keys_only=False):
        calls.append(path.name)
        return pd.DataFrame()

    monkeypatch.setattr("src.train_classifier_head.read_split", fake_read_split)
    load_splits(tmp_path, evaluate_test=False)

    assert calls == ["train.csv", "validation.csv"]
