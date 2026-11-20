"""Hard-decision screening metrics for later authorized validation/Test use."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src import screening_data as data


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float | None]:
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Wilson interval requires 0 <= successes <= total")
    if total == 0:
        return {"estimate": None, "lower": None, "upper": None, "n": 0}
    estimate = successes / total
    denominator = 1 + z * z / total
    center = (estimate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(estimate * (1 - estimate) / total + z * z / (4 * total * total)) / denominator
    return {"estimate": estimate, "lower": max(0.0, center - margin), "upper": min(1.0, center + margin), "n": total}


def hard_decision_metrics(labels: np.ndarray, decisions: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    decisions = np.asarray(decisions, dtype=np.int64)
    if len(labels) != len(decisions) or not set(np.unique(decisions)).issubset({0, 1}):
        raise ValueError("Labels and hard decisions must have equal binary length")
    # Existing definitions are reused. A binary decision is passed at 0.5 only
    # for the confusion-matrix measures; ranking AUC metrics are unavailable.
    metrics = data.calculate_metrics(labels, decisions.astype(float), 0.5)
    metrics["pr_auc"] = None
    metrics["roc_auc"] = None
    metrics["score_type"] = "hard_binary_decision"
    metrics["confidence_intervals"] = {
        "recall": wilson_interval(metrics["tp"], metrics["tp"] + metrics["fn"]),
        "fnr": wilson_interval(metrics["fn"], metrics["tp"] + metrics["fn"]),
        "precision": wilson_interval(metrics["tp"], metrics["tp"] + metrics["fp"]),
        "workload_reduction": wilson_interval(metrics["tn"] + metrics["fn"], len(labels)),
    }
    return metrics


POLICY_OUTCOMES = {
    ("completed", "include"): (1, "explicit_include"),
    ("completed", "exclude"): (0, "explicit_exclude"),
    ("completed", "uncertain"): (1, "uncertain"),
    ("malformed", None): (1, "malformed_output"),
    ("transport_error", None): (1, "api_transport_failure"),
}


def apply_screening_policy(frame: pd.DataFrame) -> pd.DataFrame:
    """Map frozen Qwen outcomes to retained/excluded without dropping records."""
    required = {data.REVIEW_COLUMN, data.LABEL_COLUMN, "decision", "status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Evaluation requires columns: {sorted(missing)}")
    result = frame.copy()
    normalized = [(str(status), None if pd.isna(decision) else str(decision))
                  for status, decision in zip(result["status"], result["decision"])]
    mapped = [POLICY_OUTCOMES.get(value) for value in normalized]
    if any(value is None for value in mapped):
        bad = sorted({outcome for outcome, value in zip(normalized, mapped) if value is None})
        raise ValueError(f"Unknown Qwen outcome(s) under the frozen screening policy: {bad}")
    result["retained"] = [value[0] for value in mapped]
    result["outcome_category"] = [value[1] for value in mapped]
    return result


def evaluate_policy_records(frame: pd.DataFrame) -> dict[str, Any]:
    """Evaluate every input under the frozen retain/exclude policy.

    ``uncertain``, malformed JSON, and exhausted API failures remain in the
    denominator and are retained for manual screening.  They are never removed
    from the primary analysis or converted to automatic exclusions.
    """
    scored = apply_screening_policy(frame)
    labels = scored[data.LABEL_COLUMN].to_numpy(dtype=np.int64)
    overall = hard_decision_metrics(labels, scored["retained"].to_numpy(dtype=np.int64))
    categories = {name: int((scored["outcome_category"] == name).sum()) for name in (
        "explicit_include", "explicit_exclude", "uncertain", "malformed_output", "api_transport_failure",
    )}
    per_review = []
    for review, group in scored.groupby(data.REVIEW_COLUMN, sort=True):
        per_review.append({"review_name": str(review), "n_samples": len(group), **hard_decision_metrics(
            group[data.LABEL_COLUMN].to_numpy(dtype=np.int64), group["retained"].to_numpy(dtype=np.int64),
        )})
    macro_names = ("recall", "fnr", "precision", "f1", "workload_reduction")
    macro = {"n_reviews": len(per_review)}
    for name in macro_names:
        values = [row[name] for row in per_review if row[name] is not None and np.isfinite(row[name])]
        macro[name] = float(np.mean(values)) if values else None
        macro[f"{name}_reviews_defined"] = len(values)
    valid = scored[scored["status"] == "completed"]
    valid_metrics = None
    if not valid.empty:
        valid_metrics = hard_decision_metrics(
            valid[data.LABEL_COLUMN].to_numpy(dtype=np.int64), valid["retained"].to_numpy(dtype=np.int64)
        )
    total = len(scored)
    valid_count = len(valid)
    technical_count = categories["malformed_output"] + categories["api_transport_failure"]
    uncertainty_count = categories["uncertain"]
    return {
        "screening_policy": "retain_unresolved_v1",
        "outcome_counts": categories,
        "response_quality": {
            "denominator_records": total,
            "valid_model_predictions": {
                "count": valid_count,
                "coverage": valid_count / total if total else None,
                "explicit_include": categories["explicit_include"],
                "explicit_exclude": categories["explicit_exclude"],
                "uncertain": uncertainty_count,
                "uncertain_rate_among_valid": uncertainty_count / valid_count if valid_count else None,
                "uncertain_rate_among_all": uncertainty_count / total if total else None,
            },
            "technical_failures": {
                "count": technical_count,
                "rate": technical_count / total if total else None,
                "malformed_output": categories["malformed_output"],
                "api_transport_failure": categories["api_transport_failure"],
            },
        },
        "end_to_end_primary": {"denominator_records": total, "metrics": overall},
        "supplementary_valid_responses": {
            "denominator_records": valid_count,
            "excluded_technical_failure_records": technical_count,
            "metrics": valid_metrics,
            "note": "Supplementary only: records without a valid model response are excluded from this denominator.",
        },
        # Retained aliases preserve the prior programmatic interface.
        "micro_global": overall, "macro_review_level": macro, "per_review": per_review,
    }


def evaluate_resolved_records(frame: pd.DataFrame) -> dict[str, Any]:
    """Legacy strict evaluator retained for callers that require binary-only input."""
    required = {data.REVIEW_COLUMN, data.LABEL_COLUMN, "decision", "status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Evaluation requires columns: {sorted(missing)}")
    unresolved = frame[frame["status"] != "completed"]
    if not unresolved.empty:
        raise ValueError(f"Refusing evaluation with {len(unresolved)} unresolved record(s)")
    decisions = frame["decision"].map({"include": 1, "exclude": 0})
    if decisions.isna().any():
        raise ValueError("Completed records have an invalid decision")
    labels = frame[data.LABEL_COLUMN].to_numpy(dtype=np.int64)
    overall = hard_decision_metrics(labels, decisions.to_numpy(dtype=np.int64))
    per_review = []
    for review, group in frame.groupby(data.REVIEW_COLUMN, sort=True):
        per_review.append({"review_name": str(review), "n_samples": len(group), **hard_decision_metrics(
            group[data.LABEL_COLUMN].to_numpy(dtype=np.int64),
            group["decision"].map({"include": 1, "exclude": 0}).to_numpy(dtype=np.int64),
        )})
    macro_names = ("recall", "fnr", "precision", "f1", "workload_reduction")
    macro = {"n_reviews": len(per_review)}
    for name in macro_names:
        values = [row[name] for row in per_review if row[name] is not None and np.isfinite(row[name])]
        macro[name] = float(np.mean(values)) if values else None
        macro[f"{name}_reviews_defined"] = len(values)
    return {"micro_global": overall, "macro_review_level": macro, "per_review": per_review}
