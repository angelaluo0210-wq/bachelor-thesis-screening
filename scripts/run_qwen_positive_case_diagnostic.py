"""Run the frozen Qwen positive-case diagnostic on Validation records only.

This label-conditioned diagnostic is not an overall performance evaluation.  It
never opens a Test split, and it reuses the frozen Qwen prompt, API client,
input reconstruction, and retain/exclude policy from the Validation Pilot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd
import transformers
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_qwen_baseline import _git_metadata, _tokenizer_identity
from src import screening_data as data
from src.llm_baseline.client import QwenClient, QwenConfig, UrllibTransport
from src.llm_baseline.prompt import PROMPT_VERSION, prompt_fingerprint
from src.llm_baseline.records import configuration_fingerprint, create_or_resume_run, sha256_file, stable_record_id
from src.llm_baseline.runner import execute_validation_records, reconstruction_audit


SELECTION_VERSION = "positive_case_diagnostic_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "processed" / "splits")
    parser.add_argument("--criteria-path", type=Path, default=ROOT / "data" / "eligibility_criteria(排纳标准).xlsx")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--per-review", type=int, default=3,
                        help="Frozen number of known-positive Validation records per review.")
    parser.add_argument("--sample-seed", type=int, default=202610)
    parser.add_argument("--execute", action="store_true",
                        help="Required before the authorized Validation-only API diagnostic.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.max_length < 16 or args.per_review < 1:
        parser.error("max-length must be at least 16 and per-review must be positive")
    return args


def _select_positive_cases(frame: pd.DataFrame, per_review: int, sample_seed: int) -> pd.DataFrame:
    """Select fixed known-positive records independently within each Validation review."""
    indexed = frame.copy().reset_index(drop=True)
    indexed["_qwen_source_row_index"] = range(len(indexed))
    positives = indexed.loc[indexed[data.LABEL_COLUMN].eq(1)].copy()
    selected: list[int] = []
    for review, group in positives.groupby(data.REVIEW_COLUMN, sort=True):
        if len(group) < per_review:
            raise ValueError(f"Validation review {review!r} has only {len(group)} positive record(s), need {per_review}")
        ranked = []
        for index, row in group.iterrows():
            record_id = stable_record_id(row.to_dict(), int(row["_qwen_source_row_index"]))
            digest = hashlib.sha256(
                f"{SELECTION_VERSION}\n{sample_seed}\n{record_id}".encode("utf-8")
            ).hexdigest()
            ranked.append((digest, index))
        selected.extend(index for _, index in sorted(ranked)[:per_review])
    return indexed.loc[selected].sort_values("_qwen_source_row_index").reset_index(drop=True)


def _prepare_positive_case_diagnostic(args: argparse.Namespace) -> tuple[dict[str, object], pd.DataFrame, object]:
    """Load Validation and criteria only, then prepare the known-positive diagnostic."""
    validation = data.read_split(args.split_dir / "validation.csv")
    data.validate_labels(validation, "validation")
    criteria, schema = data.load_eligibility_criteria(args.criteria_path)
    validation = data.join_criteria(validation, criteria, schema, "validation")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    selected = _select_positive_cases(validation, args.per_review, args.sample_seed)
    audit = reconstruction_audit(selected, tokenizer, args.max_length)
    positive_by_review = {
        str(review): int(count)
        for review, count in validation.loc[validation[data.LABEL_COLUMN].eq(1)].groupby(data.REVIEW_COLUMN).size().items()
    }
    selected_by_review = {str(review): int(count) for review, count in selected.groupby(data.REVIEW_COLUMN).size().items()}
    config = {
        "experiment": "qwen3.8_flash_zero_shot",
        "phase": "validation_positive_case_diagnostic",
        "diagnostic_only": True,
        "test_access": "forbidden",
        "test_evaluated": False,
        "model_name_or_path": args.model_name_or_path,
        "max_length": args.max_length,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_fingerprint(),
        "api": QwenConfig().public_dict(),
        "input_sha256": {
            "validation.csv": sha256_file(args.split_dir / "validation.csv"),
            "criteria.xlsx": sha256_file(args.criteria_path),
        },
        "tokenizer": _tokenizer_identity(args.model_name_or_path),
        "runtime": {"python": platform.python_version(), "transformers": transformers.__version__},
        "source": _git_metadata(),
        "positive_case_selection": {
            "method": f"sha256({SELECTION_VERSION} + seed + stable_record_id), stratified by review",
            "sample_seed": args.sample_seed,
            "label_filter": f"{data.LABEL_COLUMN} == 1",
            "per_review": args.per_review,
            "validation_population_rows": len(validation),
            "validation_positive_rows": int(validation[data.LABEL_COLUMN].eq(1).sum()),
            "positive_rows_by_review": positive_by_review,
            "selected_rows_by_review": selected_by_review,
            "selected_rows": len(selected),
        },
    }
    plan = {
        "configuration": config,
        "configuration_fingerprint": configuration_fingerprint(config),
        "selected_record_ids": audit["records"],
        "reconstruction_audit": audit,
        "planned_utc": datetime.now(timezone.utc).isoformat(),
    }
    return plan, selected, tokenizer


def _diagnostic_summary(result: dict[str, object]) -> dict[str, object]:
    outcomes = result.get("outcome_counts", {})
    assert isinstance(outcomes, dict)
    return {
        "diagnostic_only": True,
        "interpretation": "Positive-case diagnostic only; not an overall performance evaluation.",
        "known_positive_records": int(result["completed_records"]),
        "outcome_counts": outcomes,
        "automatic_positive_exclusions": int(outcomes.get("explicit_exclude", 0)),
        "retained_for_screening": int(outcomes.get("explicit_include", 0)) + int(outcomes.get("uncertain", 0))
        + int(outcomes.get("malformed_output", 0)) + int(outcomes.get("api_transport_failure", 0)),
        "token_usage_totals": result.get("token_usage_totals", {}),
        "returned_models": result.get("returned_models", {}),
        "provider_request_ids_recorded": result.get("provider_request_ids_recorded", 0),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan, selected, tokenizer = _prepare_positive_case_diagnostic(args)
    if not args.execute:
        if args.output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing Qwen diagnostic plan: {args.output_dir}")
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "positive_case_diagnostic_plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    manifest = create_or_resume_run(args.output_dir, plan["configuration"], args.resume)
    started = time.monotonic()
    (args.output_dir / "reconstruction_audit.json").write_text(
        json.dumps(plan["reconstruction_audit"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result = execute_validation_records(selected, tokenizer, args.max_length,
                                        QwenClient(QwenConfig(), UrllibTransport()), args.output_dir)
    summary = _diagnostic_summary(result)
    (args.output_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "execution_summary.json").write_text(
        json.dumps({"manifest": manifest, "elapsed_seconds": time.monotonic() - started, **result},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({**result, "diagnostic_summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
