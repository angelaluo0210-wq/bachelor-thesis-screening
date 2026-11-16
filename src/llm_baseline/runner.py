"""Validation-only incremental Qwen execution support for later AutoDL use."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .client import QwenClient
from .prompt import build_prompt
from .records import (append_prediction, completed_record_ids, record_id_for_row,
                      require_unique_record_ids, stable_record_id)
from .reconstruction import reconstruct_input


def _usage(response: dict[str, Any] | None) -> dict[str, Any] | None:
    if not response:
        return None
    value = response.get("usage")
    return value if isinstance(value, dict) else None


def reconstruction_audit(frame: pd.DataFrame, tokenizer: Any, max_length: int) -> dict[str, Any]:
    """Check every planned record against the production BERT token sequence."""
    roundtrip_failures = Counter()
    records = []
    for index, row in frame.iterrows():
        source_index = int(row.get("_qwen_source_row_index", index))
        content = reconstruct_input(tokenizer, row.to_dict(), max_length)
        failed = [field for field, passed in content.reconstruction_roundtrip.items() if not passed]
        roundtrip_failures.update(failed)
        records.append({"record_id": stable_record_id(row.to_dict(), source_index),
                        "source_row_index": source_index,
                        "roundtrip": content.reconstruction_roundtrip})
    return {"records_checked": len(records), "token_id_sequence_match": len(records),
            "decoded_field_roundtrip_failures": dict(sorted(roundtrip_failures.items())),
            "records": records}


def _response_metadata(response: dict[str, Any] | None) -> dict[str, Any]:
    if not response:
        return {"returned_model": None, "provider_request_id": None, "finish_reason": None}
    try:
        finish_reason = response["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError):
        finish_reason = None
    return {
        "returned_model": response.get("model") if isinstance(response.get("model"), str) else None,
        "provider_request_id": next((response.get(key) for key in ("_qwen_provider_request_id", "request_id", "id")
                                     if isinstance(response.get(key), str)), None),
        "finish_reason": finish_reason if isinstance(finish_reason, str) else None,
    }


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


def execute_validation_records(
    frame: pd.DataFrame, tokenizer: Any, max_length: int, client: QwenClient, run_dir: Any,
) -> dict[str, Any]:
    """Append one outcome per unresolved Validation record and support safe resume.

    The caller must have already created/validated a run manifest. The function
    accepts only the supplied frame and has no Test-file path or Test handling.
    """
    rows = [row.to_dict() for _, row in frame.iterrows()]
    record_ids = require_unique_record_ids(rows)
    complete = completed_record_ids(run_dir)
    if not complete.issubset(set(record_ids)):
        raise ValueError("Stored predictions contain records absent from the supplied Validation frame")
    written = 0
    outcome_counts = Counter()
    for index, row in enumerate(rows):
        source_index = int(row.get("_qwen_source_row_index", index))
        record_id = record_id_for_row(row, index)
        if record_id in complete:
            continue
        content = reconstruct_input(tokenizer, row, max_length)
        prompt = build_prompt(content)
        parsed, response, retries, transport_error = client.classify_with_retries(prompt)
        append_prediction(run_dir, {
            "record_id": record_id, "source_row_index": source_index, "review_name": str(row.get("review_key", "")),
            "status": parsed.status, "decision": parsed.decision, "parse_error": parsed.error,
            "transport_error": transport_error, "retry_count": retries, "usage": _usage(response),
            **_response_metadata(response),
            "estimated_cost_usd": None,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }, known_ids=complete)
        outcome_counts[_outcome_category(parsed.status, parsed.decision)] += 1
        written += 1
    final = completed_record_ids(run_dir)
    categories = Counter()
    usage_totals = Counter()
    returned_models = Counter()
    provider_request_ids = 0
    prediction_path = run_dir / "predictions.jsonl"
    if prediction_path.is_file():
        for line in prediction_path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            categories[_outcome_category(value.get("status"), value.get("decision"))] += 1
            usage = value.get("usage")
            if isinstance(usage, dict):
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    if isinstance(usage.get(key), (int, float)):
                        usage_totals[key] += usage[key]
            if isinstance(value.get("returned_model"), str):
                returned_models[value["returned_model"]] += 1
            provider_request_ids += int(isinstance(value.get("provider_request_id"), str))
    return {"input_records": len(record_ids), "written": written, "completed_records": len(final),
            "outcome_counts": dict(sorted(categories.items())), "token_usage_totals": dict(sorted(usage_totals.items())),
            "returned_models": dict(sorted(returned_models.items())), "provider_request_ids_recorded": provider_request_ids}
