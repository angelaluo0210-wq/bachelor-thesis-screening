"""Calibrate a screening threshold from validation predictions only.

The researcher supplies the recall target.  The selected threshold is the
largest threshold achieving that target, preserving the most workload reduction
among thresholds with the requested validation recall.  This utility rejects
files that do not look like validation-prediction artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = probabilities >= threshold
    tp = int(np.sum((labels == 1) & predicted)); fn = int(np.sum((labels == 1) & ~predicted))
    fp = int(np.sum((labels == 0) & predicted)); tn = int(np.sum((labels == 0) & ~predicted))
    return {"threshold": float(threshold), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "recall": tp / (tp + fn), "fnr": fn / (tp + fn),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "specificity": tn / (tn + fp), "predicted_include_rate": float(predicted.mean())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--target-recall", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.target_recall <= 1:
        parser.error("target-recall must be in (0, 1]")
    if "validation" not in args.validation_predictions.name.lower():
        parser.error("Only a validation-prediction artifact is accepted")
    frame = pd.read_csv(args.validation_predictions, usecols=["true_label", "predicted_probability"])
    labels = frame["true_label"].to_numpy(dtype=np.int64); probabilities = frame["predicted_probability"].to_numpy(dtype=float)
    if set(np.unique(labels)) != {0, 1} or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Predictions must contain finite probabilities and both binary labels")
    candidates = np.unique(np.concatenate(([0.0], probabilities)))
    feasible = [metrics(labels, probabilities, float(value)) for value in candidates]
    feasible = [result for result in feasible if result["recall"] >= args.target_recall]
    selected = max(feasible, key=lambda result: result["threshold"])
    result = {"source": str(args.validation_predictions), "split": "validation", "target_recall": args.target_recall,
              "selection_rule": "largest_threshold_with_validation_recall_at_least_target", "selected": selected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
