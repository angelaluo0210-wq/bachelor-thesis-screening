"""Release guards for the frozen, evaluation-only final Test entry point."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from scripts import evaluate_final_test as final_test
from src.train_classifier_head import calculate_metrics


def valid_evidence():
    checkpoint_path = Path("/root/bachelor-thesis-screening") / final_test.LOCKED_CHECKPOINT_RELATIVE
    manifest = {
        "status": "completed", "test_disabled": True, "pooling": "cls",
        "selection": "validation_AP", "seed": 42,
        "runs": [{
            "strategy": "lora", "best_epoch": 4,
            "best_validation_ap": final_test.LOCKED_VALIDATION_AP,
            "checkpoint_path": str(checkpoint_path),
            "learning_rates": {"parameter_groups": [
                {"name": "classifier", "lr": 1e-4},
                {"name": "adapter", "lr": 5e-5},
            ]},
        }],
    }
    run_config = {
        "training_strategy": "lora", "pooling": "cls", "seed": 42,
        "adapter_learning_rate": 5e-5, "evaluate_test": False,
        "test_evaluated": False, "comparison_mode": True,
    }
    run_summary = {
        "training_strategy": "lora", "pooling": "cls", "seed": 42,
        "best_pr_auc_epoch": 4, "best_validation_pr_auc": final_test.LOCKED_VALIDATION_AP,
        "adapter_learning_rate": 5e-5, "test_evaluated": False,
        "best_checkpoint_path": str(checkpoint_path),
    }
    run_metrics = {
        "training_strategy": "lora", "pooling": "cls", "best_epoch": 4,
        "selection_metric": "validation_pr_auc",
        "overall_validation": {"pr_auc": final_test.LOCKED_VALIDATION_AP},
    }
    report = {
        "training_strategy": "lora", "pooling": "cls", "classifier_parameters": 769,
        "adapter_parameters": 294912, "trainable_parameters": 295681,
        "lora_target_modules": [f"module-{index}" for index in range(24)],
    }
    threshold_config = {
        "source": str(checkpoint_path.with_name("validation_predictions.csv")),
        "split": "validation", "target_recall": 0.95,
        "selection_rule": "largest_threshold_with_validation_recall_at_least_target",
        "selected": {"threshold": final_test.LOCKED_THRESHOLD},
    }
    model_path = "/root/bachelor-thesis-screening/models/pubmedbert"
    checkpoint = {
        "checkpoint_version": 2, "training_strategy": "lora", "epoch": 4, "seed": 42,
        "validation_metrics": {"pr_auc": final_test.LOCKED_VALIDATION_AP},
        "model_config": {"pooling": "cls"},
        "strategy_config": {
            "training_strategy": "lora", "lora_r": 8, "lora_alpha": 16,
            "lora_dropout": 0.05, "lora_target_modules": ["query", "value"],
        },
        "adapter_config": {"bias": "none"},
        "adapter_state_dict": {"adapter": object()},
        "classifier_head_state_dict": {"weight": object()},
        "training_arguments": {
            "adapter_learning_rate": 5e-5, "pooling": "cls", "evaluate_test": False,
        },
        "model_name": model_path,
    }
    return (
        checkpoint_path, checkpoint, manifest, run_config, run_summary, run_metrics,
        report, threshold_config, model_path,
    )


def test_release_evidence_accepts_only_the_locked_validation_selected_run():
    locked = final_test.validate_release_evidence(*valid_evidence())
    assert locked == {
        "training_strategy": "lora", "adapter_learning_rate": 5e-5,
        "pooling": "cls", "best_epoch": 4,
        "checkpoint_validation_ap": final_test.LOCKED_VALIDATION_AP,
        "threshold": final_test.LOCKED_THRESHOLD, "primary_target_recall": 0.95,
        "seed": 42, "selection_data": "validation_only",
    }


@pytest.mark.parametrize("mutation", ["test_used", "threshold_changed", "epoch_changed"])
def test_release_evidence_rejects_leakage_or_changed_frozen_choices(mutation):
    evidence = list(valid_evidence())
    if mutation == "test_used":
        evidence[3]["evaluate_test"] = True
    elif mutation == "threshold_changed":
        evidence[7]["selected"]["threshold"] = 0.5
    else:
        evidence[1]["epoch"] = 5
    with pytest.raises(ValueError, match="Release invariant failed"):
        final_test.validate_release_evidence(*evidence)


def test_final_entry_point_exposes_no_threshold_or_training_option():
    with pytest.raises(SystemExit):
        final_test.parse_args(["--threshold", "0.5"])
    source = inspect.getsource(final_test.main)
    assert "train_one_epoch" not in source
    assert "build_optimizer" not in source
    assert "calibrate" not in source


def test_final_metrics_include_screening_workload_and_macro_defined_counts():
    metrics = calculate_metrics(
        np.array([1, 1, 0, 0]), np.array([0.9, 0.1, 0.8, 0.2]), 0.5
    )
    assert metrics["npv"] == 0.5
    assert metrics["predicted_include_rate"] == 0.5
    assert metrics["workload_reduction"] == 0.5
    macro = final_test.macro_review_summary([
        {name: value for name, value in {
            "recall": 1.0, "fnr": 0.0, "precision": 0.5, "f1": 2 / 3,
            "specificity": 0.5, "npv": 1.0, "predicted_include_rate": 0.5,
            "workload_reduction": 0.5,
        }.items()},
        {name: np.nan for name in (
            "recall", "fnr", "precision", "f1", "specificity", "npv",
            "predicted_include_rate", "workload_reduction",
        )},
    ])
    assert macro["n_reviews"] == 2
    assert macro["recall"] == 1.0
    assert macro["recall_reviews_defined"] == 1


def test_json_safe_serialization_replaces_undefined_metrics():
    assert final_test.json_safe({"metric": float("nan")}) == {"metric": None}
