"""Guarded infrastructure for the separately authorized Qwen final Test run.

No code in this module opens a Test file until ``load_authorized_test`` is
called with the exact release acknowledgement.  Its other functions are
designed to be exercised solely with synthetic frames in offline tests.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src import screening_data as data

from .alignment import validate_one_to_one_alignment
from .client import QwenClient
from .evaluation import apply_screening_policy, evaluate_policy_records
from .prompt import build_prompt
from .reconstruction import reconstruct_input
from .records import (append_prediction, completed_record_ids, record_id_for_row,
                      require_unique_record_ids)


LOCKED_TEST_REVIEWS = frozenset({
    "Leenaars_2019", "Muthu_2021", "Radjenovic_2013", "van_Dis_2020",
})
LOCKED_TEST_RECORDS = 23_592
FINAL_TEST_UNLOCK_ACKNOWLEDGEMENT = "I_AUTHORIZE_FINAL_QWEN_TEST_EVALUATION"


class FinalTestAuthorizationError(PermissionError):
    """Raised before any Test path is opened without the explicit release acknowledgement."""


class FinalEvaluationGateError(ValueError):
    """Raised when an incomplete or misaligned run attempts final metrics."""


@dataclass
class ConservativeRateLimiter:
    """Serial request limiter, injectable with a fake clock in offline tests."""

    requests_per_minute: float = 30.0
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    _last_request_started: float | None = None

    def __post_init__(self) -> None:
        if not 0 < self.requests_per_minute <= 600:
            raise ValueError("requests_per_minute must be in (0, 600]")

    def wait(self) -> None:
        interval = 60.0 / self.requests_per_minute
        now = self.clock()
        if self._last_request_started is not None:
            remaining = interval - (now - self._last_request_started)
            if remaining > 0:
                self.sleep(remaining)
        self._last_request_started = self.clock()


def require_final_test_authorization(acknowledgement: str | None) -> None:
    if acknowledgement != FINAL_TEST_UNLOCK_ACKNOWLEDGEMENT:
        raise FinalTestAuthorizationError(
            "Final Test is sealed. Supply the exact explicit release acknowledgement before opening Test data."
        )


def load_authorized_test(split_path: Path, acknowledgement: str) -> pd.DataFrame:
    """The sole Test-file read boundary for the Qwen runner."""
    require_final_test_authorization(acknowledgement)
    return data.read_split(split_path)


def validate_locked_test_population(
    frame: pd.DataFrame,
    expected_reviews: frozenset[str] = LOCKED_TEST_REVIEWS,
    expected_records: int = LOCKED_TEST_RECORDS,
) -> list[str]:
    """Require the complete, predeclared population before any request is sent."""
    if len(frame) != expected_records:
        raise FinalEvaluationGateError(
            f"Final Test requires exactly {expected_records} records, received {len(frame)}"
        )
    reviews = frozenset(frame[data.REVIEW_COLUMN].dropna().astype(str).unique())
    if reviews != expected_reviews:
        raise FinalEvaluationGateError(
            f"Final Test review membership mismatch: {sorted(reviews)} != {sorted(expected_reviews)}"
        )
    ids = require_unique_record_ids(row.to_dict() for _, row in frame.iterrows())
    return ids


def _outcome_category(status: str | None, decision: str | None) -> str:
    if decision in {"include", "exclude"}:
        return f"explicit_{decision}"
    if decision == "uncertain":
        return "uncertain"
    if status == "malformed":
        return "malformed_output"
    if status == "transport_error":
        return "api_transport_failure"
    return f"unknown_status:{status}"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _prediction_metadata(row: dict[str, Any]) -> dict[str, Any]:
    result = {name: data.safe_text(row.get(name, "")) for name in data.IDENTIFIER_COLUMNS}
    result["true_label"] = int(row[data.LABEL_COLUMN])
    return result


def execute_final_records(
    frame: pd.DataFrame,
    tokenizer: Any,
    max_length: int,
    client: QwenClient,
    run_dir: Path,
    limiter: ConservativeRateLimiter,
    expected_reviews: frozenset[str] = LOCKED_TEST_REVIEWS,
    expected_records: int = LOCKED_TEST_RECORDS,
) -> dict[str, Any]:
    """Append one durable outcome per record; resume only the identical run manifest."""
    expected_ids = validate_locked_test_population(frame, expected_reviews, expected_records)
    expected_set = set(expected_ids)
    complete = completed_record_ids(run_dir)
    extras = complete - expected_set
    if extras:
        raise FinalEvaluationGateError("Stored predictions contain IDs outside the frozen Test population")

    started = time.monotonic()
    for fallback_index, (_, source_row) in enumerate(frame.iterrows()):
        row = source_row.to_dict()
        record_id = record_id_for_row(row, fallback_index)
        if record_id in complete:
            continue
        content = reconstruct_input(tokenizer, row, max_length)
        parsed, response, retries, transport_error = client.classify_with_retries(
            build_prompt(content), before_attempt=limiter.wait,
        )
        usage = response.get("usage") if isinstance(response, dict) and isinstance(response.get("usage"), dict) else None
        returned_model = response.get("model") if isinstance(response, dict) else None
        provider_request_id = next((response.get(key) for key in ("_qwen_provider_request_id", "request_id", "id")
                                    if isinstance(response, dict) and isinstance(response.get(key), str)), None)
        append_prediction(run_dir, {
            "record_id": record_id, "source_row_index": fallback_index,
            "review_name": str(row[data.REVIEW_COLUMN]), "status": parsed.status,
            "decision": parsed.decision, "parse_error": parsed.error,
            "transport_error": transport_error, "retry_count": retries, "usage": usage,
            "returned_model": returned_model if isinstance(returned_model, str) else None,
            "provider_request_id": provider_request_id, **_prediction_metadata(row),
        }, known_ids=complete)
        _atomic_json(run_dir / "progress.json", summarize_predictions(run_dir, expected_set, started))
    return summarize_predictions(run_dir, expected_set, started)


def summarize_predictions(run_dir: Path, expected_ids: set[str], started: float | None = None) -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    prediction_path = run_dir / "predictions.jsonl"
    if prediction_path.is_file():
        for line in prediction_path.read_text(encoding="utf-8").splitlines():
            values.append(json.loads(line))
    actual_ids = [str(value["record_id"]) for value in values]
    if len(actual_ids) != len(set(actual_ids)):
        raise FinalEvaluationGateError("Prediction JSONL contains duplicate record IDs")
    categories = Counter(_outcome_category(value.get("status"), value.get("decision")) for value in values)
    usage = Counter()
    for value in values:
        if isinstance(value.get("usage"), dict):
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if isinstance(value["usage"].get(key), (int, float)):
                    usage[key] += value["usage"][key]
    actual_set = set(actual_ids)
    return {
        "expected_records": len(expected_ids), "completed_records": len(actual_set),
        "missing_records": len(expected_ids - actual_set), "extra_records": len(actual_set - expected_ids),
        "outcome_counts": dict(sorted(categories.items())), "token_usage_totals": dict(sorted(usage.items())),
        "elapsed_seconds": None if started is None else time.monotonic() - started,
    }


def require_complete_prediction_set(
    frame: pd.DataFrame, predictions: pd.DataFrame,
    expected_reviews: frozenset[str] = LOCKED_TEST_REVIEWS,
    expected_records: int = LOCKED_TEST_RECORDS,
) -> None:
    expected = set(validate_locked_test_population(frame, expected_reviews, expected_records))
    if "record_id" not in predictions.columns:
        raise FinalEvaluationGateError("Qwen predictions are missing record_id")
    actual_values = predictions["record_id"].astype(str).tolist()
    actual = set(actual_values)
    if len(actual) != len(actual_values) or actual != expected:
        raise FinalEvaluationGateError(
            f"Qwen predictions must exactly equal the frozen Test IDs; missing={len(expected - actual)}, extra={len(actual - expected)}"
        )


def evaluate_final_gate(
    source: pd.DataFrame, qwen_predictions: pd.DataFrame, pubmedbert_predictions: pd.DataFrame,
    expected_reviews: frozenset[str] = LOCKED_TEST_REVIEWS,
    expected_records: int = LOCKED_TEST_RECORDS,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Validate three-way identity/labels, then calculate only frozen-policy metrics."""
    require_complete_prediction_set(source, qwen_predictions, expected_reviews, expected_records)
    required_qwen = {"review_name", *data.IDENTIFIER_COLUMNS, "true_label", "status", "decision", "record_id"}
    required_pubmed = {"review_name", "true_label", "predicted_label", *data.IDENTIFIER_COLUMNS}
    missing_qwen, missing_pubmed = required_qwen - set(qwen_predictions), required_pubmed - set(pubmedbert_predictions)
    if missing_qwen or missing_pubmed:
        raise FinalEvaluationGateError(
            f"Missing prediction columns: qwen={sorted(missing_qwen)}, pubmedbert={sorted(missing_pubmed)}"
        )
    qwen_alignment = validate_one_to_one_alignment(source, qwen_predictions)
    pubmed_alignment = validate_one_to_one_alignment(source, pubmedbert_predictions)
    if qwen_alignment.status != "verified_synthetic" or pubmed_alignment.status != "verified_synthetic":
        raise FinalEvaluationGateError("One-to-one Qwen/PubMedBERT/gold alignment failed")
    if not set(pd.to_numeric(pubmedbert_predictions["predicted_label"], errors="coerce").dropna().unique()).issubset({0, 1}):
        raise FinalEvaluationGateError("PubMedBERT predicted_label must be binary")

    source_ids = [record_id_for_row(row.to_dict(), index) for index, (_, row) in enumerate(source.iterrows())]
    qwen_by_id = qwen_predictions.set_index("record_id")
    if not qwen_by_id.index.is_unique:
        raise FinalEvaluationGateError("Qwen prediction record IDs are not unique")
    pubmed_ids = [
        data.normalize_review_key(row["review_name"]) + "|" + "|".join(
            f"{name}={data.safe_text(row[name]).lower()}" for name in data.IDENTIFIER_COLUMNS if data.safe_text(row[name])
        ) for _, row in pubmedbert_predictions.iterrows()
    ]
    pubmed_by_id = pubmedbert_predictions.assign(record_id=pubmed_ids).set_index("record_id")
    if not pubmed_by_id.index.is_unique:
        raise FinalEvaluationGateError("PubMedBERT prediction record IDs are not unique")
    evaluation_rows = []
    paired_rows = []
    for record_id, (_, row) in zip(source_ids, source.iterrows()):
        qwen, pubmed = qwen_by_id.loc[record_id], pubmed_by_id.loc[record_id]
        evaluation_rows.append({data.REVIEW_COLUMN: row[data.REVIEW_COLUMN], data.LABEL_COLUMN: int(row[data.LABEL_COLUMN]),
                                "status": qwen["status"], "decision": qwen["decision"]})
        paired_rows.append({"record_id": record_id, "review_name": row[data.REVIEW_COLUMN],
                            "true_label": int(row[data.LABEL_COLUMN]), "qwen_status": qwen["status"],
                            "qwen_decision": qwen["decision"], "pubmedbert_predicted_label": int(pubmed["predicted_label"])})
    metrics = evaluate_policy_records(pd.DataFrame(evaluation_rows))
    paired = pd.DataFrame(paired_rows)
    # The frozen policy turns all unresolved Qwen outcomes into retained=1.
    paired["qwen_retained"] = apply_screening_policy(pd.DataFrame(evaluation_rows))["retained"].to_numpy()
    paired["decision_difference"] = paired["qwen_retained"] != paired["pubmedbert_predicted_label"]
    return metrics, paired
