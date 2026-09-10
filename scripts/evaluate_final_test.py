"""Run the one-time, evaluation-only final Test release.

This entry point is intentionally specific to the frozen thesis release.  It
validates the validation-derived selection evidence and checkpoint metadata
before it opens test.csv.  It contains no training, selection, calibration, or
retry path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import train_classifier_head as trainer


LOCKED_CHECKPOINT_RELATIVE = (
    "outputs/training_strategy_comparison/20260910_150708_3f3f0eb0/"
    "lora/adapter_lr_5e-5/lora/seed_42/cls/run_20260910_191547/"
    "best_pr_auc_classifier_head.pt"
)
LOCKED_THRESHOLD = 0.016082723
LOCKED_TARGET_RECALL = 0.95
LOCKED_VALIDATION_AP = 0.19005475782306838
LOCKED_BEST_EPOCH = 4
LOCKED_SEED = 42
LOCKED_ADAPTER_LR = 5e-5
LOCKED_TEST_REVIEWS = {
    "Leenaars_2019", "Muthu_2021", "Radjenovic_2013", "van_Dis_2020"
}
LOCKED_TEST_PROFILE = {
    "Leenaars_2019": {"n_samples": 5811, "n_positive": 17},
    "Muthu_2021": {"n_samples": 2718, "n_positive": 335},
    "Radjenovic_2013": {"n_samples": 5935, "n_positive": 48},
    "van_Dis_2020": {"n_samples": 9128, "n_positive": 72},
}
EXPECTED_OUTPUT_FILES = (
    "final_test_metrics.json",
    "final_test_predictions.csv",
    "per_review_test_metrics.csv",
    "final_test_config.json",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--comparison-manifest",
        type=Path,
        required=True,
        help="Manifest that selected the LoRA 5e-5 run using validation AP.",
    )
    parser.add_argument(
        "--threshold-config",
        type=Path,
        required=True,
        help="Validation-only locked-threshold JSON.",
    )
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--criteria-path", type=Path, required=True)
    parser.add_argument("--canonical-split", type=Path, required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--require-cuda", action="store_true",
        help="Fail rather than silently evaluating on CPU/MPS.",
    )
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    if args.num_workers < 0:
        parser.error("num-workers cannot be negative")
    return args


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required release artifact not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_path(value: Any) -> str:
    return str(value).replace("\\", "/").rstrip("/")


def _matches_locked_checkpoint(value: Any) -> bool:
    return _normalized_path(value).endswith(LOCKED_CHECKPOINT_RELATIVE)


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"Release invariant failed for {label}: {actual!r} != {expected!r}")


def _require_close(actual: Any, expected: float, label: str) -> None:
    try:
        valid = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-15)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"Release invariant failed for {label}: {actual!r} != {expected!r}")


def validate_release_evidence(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    manifest: dict[str, Any],
    run_config: dict[str, Any],
    run_summary: dict[str, Any],
    run_metrics: dict[str, Any],
    parameter_report: dict[str, Any],
    threshold_config: dict[str, Any],
    expected_model_name_or_path: str,
) -> dict[str, Any]:
    """Validate all frozen choices without opening the Test split."""
    if not _matches_locked_checkpoint(checkpoint_path):
        raise ValueError(f"Checkpoint is not the locked release path: {checkpoint_path}")

    _require_equal(manifest.get("status"), "completed", "manifest status")
    _require_equal(manifest.get("test_disabled"), True, "manifest test_disabled")
    _require_equal(manifest.get("pooling"), "cls", "manifest pooling")
    _require_equal(manifest.get("selection"), "validation_AP", "manifest selection")
    _require_equal(manifest.get("seed"), LOCKED_SEED, "manifest seed")
    matching_runs = [
        run for run in manifest.get("runs", [])
        if _matches_locked_checkpoint(run.get("checkpoint_path", ""))
    ]
    if len(matching_runs) != 1:
        raise ValueError("Manifest must contain exactly one locked checkpoint run")
    selected_run = matching_runs[0]
    _require_equal(selected_run.get("strategy"), "lora", "manifest selected strategy")
    _require_equal(selected_run.get("best_epoch"), LOCKED_BEST_EPOCH, "manifest best epoch")
    _require_close(selected_run.get("best_validation_ap"), LOCKED_VALIDATION_AP,
                   "manifest best validation AP")
    adapter_groups = [
        group for group in selected_run.get("learning_rates", {}).get("parameter_groups", [])
        if group.get("name") == "adapter"
    ]
    if len(adapter_groups) != 1:
        raise ValueError("Manifest selected run must contain exactly one adapter LR group")
    _require_close(adapter_groups[0].get("lr"), LOCKED_ADAPTER_LR, "manifest adapter LR")

    _require_equal(run_config.get("training_strategy"), "lora", "run config strategy")
    _require_equal(run_config.get("pooling"), "cls", "run config pooling")
    _require_equal(run_config.get("seed"), LOCKED_SEED, "run config seed")
    _require_close(run_config.get("adapter_learning_rate"), LOCKED_ADAPTER_LR,
                   "run config adapter LR")
    _require_equal(run_config.get("evaluate_test"), False, "run config evaluate_test")
    _require_equal(run_config.get("test_evaluated"), False, "run config test_evaluated")
    _require_equal(run_config.get("comparison_mode"), True, "run config comparison_mode")

    _require_equal(run_summary.get("training_strategy"), "lora", "run summary strategy")
    _require_equal(run_summary.get("pooling"), "cls", "run summary pooling")
    _require_equal(run_summary.get("seed"), LOCKED_SEED, "run summary seed")
    _require_equal(run_summary.get("best_pr_auc_epoch"), LOCKED_BEST_EPOCH,
                   "run summary best epoch")
    _require_close(run_summary.get("best_validation_pr_auc"), LOCKED_VALIDATION_AP,
                   "run summary best validation AP")
    _require_close(run_summary.get("adapter_learning_rate"), LOCKED_ADAPTER_LR,
                   "run summary adapter LR")
    _require_equal(run_summary.get("test_evaluated"), False, "run summary test_evaluated")
    if not _matches_locked_checkpoint(run_summary.get("best_checkpoint_path", "")):
        raise ValueError("Run summary does not identify the locked checkpoint")

    _require_equal(run_metrics.get("training_strategy"), "lora", "metrics strategy")
    _require_equal(run_metrics.get("pooling"), "cls", "metrics pooling")
    _require_equal(run_metrics.get("best_epoch"), LOCKED_BEST_EPOCH, "metrics best epoch")
    _require_equal(run_metrics.get("selection_metric"), "validation_pr_auc",
                   "metrics selection")
    _require_close(run_metrics.get("overall_validation", {}).get("pr_auc"),
                   LOCKED_VALIDATION_AP, "metrics validation AP")

    _require_equal(parameter_report.get("training_strategy"), "lora",
                   "parameter report strategy")
    _require_equal(parameter_report.get("pooling"), "cls", "parameter report pooling")
    _require_equal(parameter_report.get("classifier_parameters"), 769,
                   "classifier parameter count")
    _require_equal(parameter_report.get("adapter_parameters"), 294912,
                   "adapter parameter count")
    _require_equal(parameter_report.get("trainable_parameters"), 295681,
                   "trainable parameter count")
    _require_equal(len(parameter_report.get("lora_target_modules", [])), 24,
                   "LoRA target module count")

    _require_equal(threshold_config.get("split"), "validation", "threshold split")
    _require_equal(
        threshold_config.get("selection_rule"),
        "largest_threshold_with_validation_recall_at_least_target",
        "threshold selection rule",
    )
    _require_close(threshold_config.get("target_recall"), LOCKED_TARGET_RECALL,
                   "threshold target recall")
    _require_close(threshold_config.get("selected", {}).get("threshold"), LOCKED_THRESHOLD,
                   "locked threshold")
    expected_validation_predictions = str(Path(LOCKED_CHECKPOINT_RELATIVE).with_name(
        "validation_predictions.csv"
    )).replace("\\", "/")
    if not _normalized_path(threshold_config.get("source", "")).endswith(
        expected_validation_predictions
    ):
        raise ValueError("Threshold source is not the locked run's validation predictions")

    _require_equal(checkpoint.get("checkpoint_version"), 2, "checkpoint version")
    _require_equal(checkpoint.get("training_strategy"), "lora", "checkpoint strategy")
    _require_equal(checkpoint.get("epoch"), LOCKED_BEST_EPOCH, "checkpoint epoch")
    _require_equal(checkpoint.get("seed"), LOCKED_SEED, "checkpoint seed")
    _require_close(checkpoint.get("validation_metrics", {}).get("pr_auc"),
                   LOCKED_VALIDATION_AP, "checkpoint validation AP")
    _require_equal(checkpoint.get("model_config", {}).get("pooling"), "cls",
                   "checkpoint pooling")
    strategy = checkpoint.get("strategy_config", {})
    _require_equal(strategy.get("training_strategy"), "lora", "checkpoint strategy config")
    _require_equal(strategy.get("lora_r"), 8, "checkpoint LoRA r")
    _require_equal(strategy.get("lora_alpha"), 16, "checkpoint LoRA alpha")
    _require_close(strategy.get("lora_dropout"), 0.05, "checkpoint LoRA dropout")
    _require_equal(sorted(strategy.get("lora_target_modules", [])), ["query", "value"],
                   "checkpoint LoRA target modules")
    _require_equal(checkpoint.get("adapter_config", {}).get("bias"), "none",
                   "checkpoint LoRA bias")
    if not checkpoint.get("adapter_state_dict"):
        raise ValueError("Checkpoint has no LoRA adapter state")
    if not checkpoint.get("classifier_head_state_dict"):
        raise ValueError("Checkpoint has no classifier-head state")
    training_arguments = checkpoint.get("training_arguments", {})
    _require_close(training_arguments.get("adapter_learning_rate"), LOCKED_ADAPTER_LR,
                   "checkpoint adapter LR")
    _require_equal(training_arguments.get("pooling"), "cls", "checkpoint training pooling")
    _require_equal(training_arguments.get("evaluate_test"), False,
                   "checkpoint training evaluate_test")
    _require_equal(checkpoint.get("model_name"), expected_model_name_or_path,
                   "checkpoint base-model path")

    return {
        "training_strategy": "lora",
        "adapter_learning_rate": LOCKED_ADAPTER_LR,
        "pooling": "cls",
        "best_epoch": LOCKED_BEST_EPOCH,
        "checkpoint_validation_ap": LOCKED_VALIDATION_AP,
        "threshold": LOCKED_THRESHOLD,
        "primary_target_recall": LOCKED_TARGET_RECALL,
        "seed": LOCKED_SEED,
        "selection_data": "validation_only",
    }


def macro_review_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "recall", "fnr", "precision", "f1", "specificity", "npv",
        "predicted_include_rate", "workload_reduction",
    )
    result: dict[str, Any] = {"n_reviews": len(rows)}
    for name in metric_names:
        values = [float(row[name]) for row in rows if np.isfinite(float(row[name]))]
        result[name] = float(np.mean(values)) if values else None
        result[f"{name}_reviews_defined"] = len(values)
    return result


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint_path = args.checkpoint.resolve()
    run_dir = checkpoint_path.parent
    if args.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite/reuse final-test output directory: {args.output_dir}"
        )
    required_run_files = (
        "config.json", "run_summary.json", "metrics.json", "parameter_report.json",
        "validation_predictions.csv",
    )
    missing = [name for name in required_run_files if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Locked run directory is incomplete: {missing}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Locked checkpoint not found: {checkpoint_path}")
    for path in (
        args.comparison_manifest, args.threshold_config, args.criteria_path,
        args.canonical_split, Path(args.model_name_or_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required release input not found: {path}")

    # All release/provenance checks occur before the Test split is opened.
    manifest = read_json(args.comparison_manifest)
    run_config = read_json(run_dir / "config.json")
    run_summary = read_json(run_dir / "run_summary.json")
    run_metrics = read_json(run_dir / "metrics.json")
    report = read_json(run_dir / "parameter_report.json")
    threshold_config = read_json(args.threshold_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    locked = validate_release_evidence(
        checkpoint_path, checkpoint, manifest, run_config, run_summary, run_metrics,
        report, threshold_config, args.model_name_or_path,
    )
    canonical = read_json(args.canonical_split)
    _require_equal(set(canonical.get("review_membership", {}).get("test", [])),
                   LOCKED_TEST_REVIEWS, "canonical Test review membership")

    trainer.configure_logging()
    trainer.set_seed(LOCKED_SEED)
    device = trainer.select_device()
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for final Test, but it is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model, restored_checkpoint = trainer.load_classifier_checkpoint(checkpoint_path, device)
    _require_equal(restored_checkpoint.get("epoch"), LOCKED_BEST_EPOCH,
                   "restored checkpoint epoch")

    # Crossing this line is the single authorized Test access.
    test = trainer.read_split(args.test_split)
    trainer.validate_labels(test, "test")
    actual_reviews = set(test[trainer.REVIEW_COLUMN].astype(str).unique())
    _require_equal(actual_reviews, LOCKED_TEST_REVIEWS, "actual Test review membership")
    actual_profile = {
        str(review): {
            "n_samples": int(len(group)),
            "n_positive": int((group[trainer.LABEL_COLUMN] == 1).sum()),
        }
        for review, group in test.groupby(trainer.REVIEW_COLUMN, sort=True)
    }
    _require_equal(actual_profile, LOCKED_TEST_PROFILE, "actual Test split profile")
    criteria, schema = trainer.load_eligibility_criteria(args.criteria_path)
    test = trainer.join_criteria(test, criteria, schema, "test")
    test_summary = trainer.summarize_split(test, "test")

    max_length = int(checkpoint.get("training_arguments", {}).get("max_length", 512))
    model_limit = min(
        int(getattr(tokenizer, "model_max_length", max_length)),
        int(getattr(model.encoder.config, "max_position_embeddings", max_length)),
    )
    if max_length != 512 or max_length > model_limit:
        raise ValueError(f"Checkpoint max_length={max_length} is invalid for final Test")
    loader = trainer.make_loader(
        test, tokenizer, max_length, args.batch_size, False,
        args.num_workers, LOCKED_SEED,
    )
    pos_weight = float(checkpoint["pos_weight"])
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )
    test_loss, global_metrics, probabilities, labels, indices = trainer.evaluate(
        model, loader, criterion, device, LOCKED_THRESHOLD, "Frozen final Test evaluation"
    )
    per_review = trainer.calculate_per_review_metrics(
        test, probabilities, labels, indices, LOCKED_THRESHOLD, context_name="Test"
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictions = trainer.prediction_frame(
        test, probabilities, labels, indices, LOCKED_THRESHOLD
    )
    predictions.to_csv(args.output_dir / "final_test_predictions.csv", index=False)
    pd.DataFrame(per_review).to_csv(
        args.output_dir / "per_review_test_metrics.csv", index=False
    )
    metrics_payload = {
        "threshold_independent": {
            "pr_auc": global_metrics["pr_auc"],
            "roc_auc": global_metrics["roc_auc"],
        },
        "at_locked_threshold": global_metrics,
        "micro_global": global_metrics,
        "macro_review_level": macro_review_summary(per_review),
        "test_loss_weighted_bce": test_loss,
    }
    write_json(args.output_dir / "final_test_metrics.json", metrics_payload)
    final_config = {
        **locked,
        "selected_checkpoint": _normalized_path(args.checkpoint),
        "checkpoint_sha256": sha256(checkpoint_path),
        "model_name_or_path": args.model_name_or_path,
        "max_length": max_length,
        "batch_size": args.batch_size,
        "test_split": _normalized_path(args.test_split),
        "test_split_sha256": sha256(args.test_split),
        "test_row_count": len(test),
        "test_review_count": len(actual_reviews),
        "test_reviews": sorted(actual_reviews),
        "test_data_qc": test_summary,
        "comparison_manifest": _normalized_path(args.comparison_manifest),
        "threshold_config": _normalized_path(args.threshold_config),
        "criteria_path": _normalized_path(args.criteria_path),
        "criteria_sha256": sha256(args.criteria_path),
        "canonical_split": _normalized_path(args.canonical_split),
        "device": str(device),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "test_was_evaluation_only": True,
        "training_performed": False,
        "threshold_selected_on_test": False,
        "checkpoint_selected_on_test": False,
        "hyperparameters_selected_on_test": False,
        "expected_output_files": list(EXPECTED_OUTPUT_FILES),
    }
    write_json(args.output_dir / "final_test_config.json", final_config)
    print(f"Final Test complete. Frozen outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
