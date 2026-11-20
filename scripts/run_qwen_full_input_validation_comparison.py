"""Authorized executor for one prepared mode of the Qwen Validation comparison.

No published tokenizer is proven identical to Model Studio's managed
``qwen3.8-flash`` service.  Context safety therefore uses a conservative UTF-8
byte upper bound plus chat-envelope allowance, never a surrogate token count.
Provider-returned usage remains the only billable token measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_qwen_full_input_validation_comparison import _raw_fields, sample_validation
from src import screening_data as data
from src.llm_baseline.client import QwenClient, QwenConfig, UrllibTransport
from src.llm_baseline.prompt import build_prompt, build_prompt_fields
from src.llm_baseline.records import (append_prediction, completed_record_ids,
                                      create_or_resume_run, record_id_for_row,
                                      require_unique_record_ids)
from src.llm_baseline.final_test import ConservativeRateLimiter
from src.llm_baseline.reconstruction import reconstruct_input
from src.llm_baseline.runner import _response_metadata, _usage

PROVIDER_MAX_INPUT_TOKENS = 991_808  # Official Model Studio qwen3.8-flash limit, verified 2026-11-12.
CONSERVATIVE_SAFE_INPUT_BYTES = 900_000
CHAT_ENVELOPE_ALLOWANCE_BYTES = 4_096
MAX_OUTPUT_TOKENS = 128


def conservative_context_audit(prompt: str) -> dict[str, int | str]:
    """Bound input without claiming UTF-8 bytes are provider tokens.

    900K bytes leaves 91,808 tokens of headroom below the documented input
    ceiling even under the deliberately pessimistic one-byte-per-token proxy;
    the 4 KiB allowance covers the OpenAI chat envelope and future small
    provider-side framing changes.
    """
    prompt_bytes = len(prompt.encode("utf-8"))
    checked_bytes = prompt_bytes + CHAT_ENVELOPE_ALLOWANCE_BYTES
    if checked_bytes > CONSERVATIVE_SAFE_INPUT_BYTES:
        raise ValueError(
            f"Conservative context bound exceeded ({checked_bytes}>{CONSERVATIVE_SAFE_INPUT_BYTES} UTF-8 bytes); "
            "refusing request without truncating input"
        )
    return {"method": "utf8_bytes_plus_chat_allowance_not_token_count", "prompt_utf8_bytes": prompt_bytes,
            "chat_envelope_allowance_bytes": CHAT_ENVELOPE_ALLOWANCE_BYTES,
            "checked_utf8_bytes": checked_bytes, "safe_limit_bytes": CONSERVATIVE_SAFE_INPUT_BYTES}


def select_technical_validation_records(sample: pd.DataFrame, audit_path: Path, count: int) -> pd.DataFrame:
    """Choose a deterministic, coverage-oriented subset from the frozen manifest.

    This does not sample new records or inspect model outcomes.  It greedily
    covers label, abstract truncation, review, and short/long abstract strata.
    """
    audit = pd.read_csv(audit_path, usecols=["record_id", "abstract_truncated", "abstract_original_tokens"])
    indexed = sample.copy()
    indexed["_technical_record_id"] = [record_id_for_row(row.to_dict(), i) for i, (_, row) in enumerate(sample.iterrows())]
    indexed = indexed.merge(audit, left_on="_technical_record_id", right_on="record_id", how="left", validate="one_to_one")
    if indexed["abstract_truncated"].isna().any():
        raise ValueError("Technical selection could not match the frozen truncation audit")
    q25, q75 = indexed["abstract_original_tokens"].quantile([.25, .75]).tolist()
    def features(row: pd.Series) -> set[str]:
        value = {f"label={int(row[data.LABEL_COLUMN])}", f"truncated={bool(row['abstract_truncated'])}",
                 f"review={row[data.REVIEW_COLUMN]}"}
        if row["abstract_original_tokens"] <= q25: value.add("abstract_length=short")
        if row["abstract_original_tokens"] >= q75: value.add("abstract_length=long")
        return value
    available = {index: features(row) for index, row in indexed.iterrows()}
    chosen: list[int] = []
    covered: set[str] = set()
    label_target = {0: count // 2, 1: count - count // 2}
    label_count = {0: 0, 1: 0}
    review_count: dict[str, int] = {}
    # First guarantee every available label × truncation state is represented.
    for label in (1, 0):
        for truncated in (True, False):
            candidates = [index for index in indexed.index if index not in chosen and int(indexed.loc[index, data.LABEL_COLUMN]) == label and bool(indexed.loc[index, "abstract_truncated"]) == truncated]
            if not candidates:
                continue
            best = min(candidates, key=lambda index: (
                review_count.get(str(indexed.loc[index, data.REVIEW_COLUMN]), 0),
                int(indexed.loc[index, "_qwen_source_row_index"]),
            ))
            chosen.append(best); covered.update(available[best]); label_count[label] += 1
            review = str(indexed.loc[best, data.REVIEW_COLUMN]); review_count[review] = review_count.get(review, 0) + 1
    while len(chosen) < min(count, len(indexed)):
        candidates = [index for index in indexed.index if index not in chosen and label_count[int(indexed.loc[index, data.LABEL_COLUMN])] < label_target[int(indexed.loc[index, data.LABEL_COLUMN])]]
        best = min(candidates, key=lambda index: (
            -len(available[index] - covered),
            review_count.get(str(indexed.loc[index, data.REVIEW_COLUMN]), 0),
            int(indexed.loc[index, "_qwen_source_row_index"]),
        ))
        chosen.append(best); covered.update(available[best])
        label = int(indexed.loc[best, data.LABEL_COLUMN]); label_count[label] += 1
        review = str(indexed.loc[best, data.REVIEW_COLUMN]); review_count[review] = review_count.get(review, 0) + 1
    selected = indexed.loc[chosen].sort_values("_qwen_source_row_index").drop(columns=["record_id", "_technical_record_id", "abstract_truncated", "abstract_original_tokens"])
    if len(selected) != count:
        raise ValueError(f"Could not select {count} technical-validation records")
    return selected.reset_index(drop=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", choices=("truncated", "full"), required=True)
    parser.add_argument("--prepared-dir", type=Path, default=ROOT / "outputs" / "qwen_full_input_validation_comparison")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "processed" / "splits")
    parser.add_argument("--criteria-path", type=Path, default=ROOT / "data" / "eligibility_criteria(排纳标准).xlsx")
    parser.add_argument("--pubmedbert-tokenizer", type=Path, default=ROOT / "model" / "pubmedbert")
    parser.add_argument("--qwen-context-window", type=int, default=PROVIDER_MAX_INPUT_TOKENS,
                        help="Must equal the provider-verified qwen3.8-flash input limit (991808).")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--requests-per-minute", type=float, default=30.0)
    parser.add_argument("--technical-validation-records", type=int,
                        help="Run only this fixed prefix per mode, then stop; requires valid non-length JSON responses.")
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("Refusing API use without --execute")
    if args.qwen_context_window != PROVIDER_MAX_INPUT_TOKENS:
        parser.error("Refusing an unverified context limit; qwen3.8-flash is currently verified at 991808 input tokens")
    if not 0 < args.requests_per_minute <= 600:
        parser.error("requests-per-minute must be in (0, 600]")
    if args.technical_validation_records is not None and args.technical_validation_records < 1:
        parser.error("technical-validation-records must be positive")
    manifest_path = args.prepared_dir / "sample_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared manifest is missing: {manifest_path}")
    validation = data.read_split(args.split_dir / "validation.csv")
    criteria, schema = data.load_eligibility_criteria(args.criteria_path)
    validation = data.join_criteria(validation, criteria, schema, "validation")
    sample, _ = sample_validation(validation)
    expected = set(pd.read_csv(manifest_path)["record_id"].astype(str))
    actual = {record_id_for_row(row.to_dict(), i) for i, (_, row) in enumerate(sample.iterrows())}
    if actual != expected or len(actual) != 5000:
        raise ValueError("Current Validation inputs no longer match the frozen 5,000-record manifest")
    if args.technical_validation_records is not None:
        sample = select_technical_validation_records(
            sample, args.prepared_dir / "truncation_audit.csv", args.technical_validation_records
        )
    bert_tokenizer = AutoTokenizer.from_pretrained(args.pubmedbert_tokenizer, local_files_only=True) if args.mode == "truncated" else None
    prompts: list[tuple[str, dict[str, int | str]]] = []
    for _, row in sample.iterrows():
        values = row.to_dict()
        prompt = build_prompt(reconstruct_input(bert_tokenizer, values)) if args.mode == "truncated" else build_prompt_fields(**_raw_fields(values))
        prompts.append((prompt, conservative_context_audit(prompt)))
    run_name = f"technical_validation_{args.mode}" if args.technical_validation_records is not None else f"predictions_{args.mode}"
    run_dir = args.prepared_dir / run_name
    config = {"phase": "technical_validation" if args.technical_validation_records is not None else "validation_comparison_execution", "mode": args.mode, "records": len(sample),
              "qwen_context_window": args.qwen_context_window,
              "context_check": {"method": "utf8_bytes_plus_chat_allowance_not_token_count", "safe_limit_bytes": CONSERVATIVE_SAFE_INPUT_BYTES},
              "api": QwenConfig(max_tokens=MAX_OUTPUT_TOKENS).public_dict(),
              "requests_per_minute": args.requests_per_minute,
              "difference_from_legacy_test": {"max_tokens": {"legacy": None, "comparison": MAX_OUTPUT_TOKENS}},
              "technical_selection_record_ids": [record_id_for_row(row.to_dict(), i) for i, (_, row) in enumerate(sample.iterrows())] if args.technical_validation_records is not None else None}
    create_or_resume_run(run_dir, config, args.resume)
    rows = [row.to_dict() for _, row in sample.iterrows()]
    ids = require_unique_record_ids(rows); completed = completed_record_ids(run_dir)
    if not completed.issubset(set(ids)): raise ValueError("Stored predictions do not belong to this manifest")
    client = QwenClient(QwenConfig(max_tokens=MAX_OUTPUT_TOKENS), UrllibTransport())
    limiter = ConservativeRateLimiter(args.requests_per_minute)
    consecutive_technical_failures = 0
    for i, (row, prompt_and_audit) in enumerate(zip(rows, prompts)):
        prompt, context_audit = prompt_and_audit
        record_id = record_id_for_row(row, i)
        if record_id in completed: continue
        parsed, response, retries, transport_error = client.classify_with_retries(prompt, before_attempt=limiter.wait)
        metadata = _response_metadata(response)
        usage = _usage(response)
        output_at_limit = bool(isinstance(usage, dict) and usage.get("completion_tokens") == MAX_OUTPUT_TOKENS)
        append_prediction(run_dir, {"record_id": record_id, "source_row_index": int(row["_qwen_source_row_index"]),
            "review_name": str(row[data.REVIEW_COLUMN]), "gold_label": int(row[data.LABEL_COLUMN]), "status": parsed.status, "decision": parsed.decision,
            "input_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(), "context_audit": context_audit,
            "parse_error": parsed.error, "transport_error": transport_error, "retry_count": retries,
            "usage": usage, "output_at_max_tokens": output_at_limit, **metadata, "estimated_cost_usd": None,
            "completed_utc": datetime.now(timezone.utc).isoformat()}, known_ids=completed)
        consecutive_technical_failures = 0 if parsed.status == "completed" else consecutive_technical_failures + 1
        if consecutive_technical_failures >= 3:
            raise RuntimeError("Three consecutive exhausted technical failures; stopping before further API requests")
        if args.technical_validation_records is not None and (parsed.status != "completed" or parsed.decision not in {"include", "exclude", "uncertain"} or metadata["finish_reason"] == "length" or output_at_limit):
            raise RuntimeError("Technical validation failed: malformed/invalid decision or output-length truncation; do not start batch run")
    if args.technical_validation_records is not None:
        stored = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line]
        if len(stored) != len(rows) or any(
            value.get("status") != "completed" or value.get("decision") not in {"include", "exclude", "uncertain"}
            or value.get("finish_reason") == "length" or value.get("output_at_max_tokens")
            for value in stored
        ):
            raise RuntimeError("Technical validation artifacts contain an invalid, length-truncated, or boundary-length response; do not start batch run")
    print(json.dumps({"mode": args.mode, "completed_records": len(completed)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
