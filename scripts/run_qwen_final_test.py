"""Run the one-time Qwen final Test evaluation after an explicit release.

This is deliberately separate from the Validation-only runners.  It has no
sampling option: the only executable path requires the exact Test unlock
acknowledgement and then processes the predeclared complete population.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
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
from src.llm_baseline.final_test import (
    ConservativeRateLimiter, evaluate_final_gate, evaluate_qwen_final,
    execute_final_records, load_authorized_test, require_final_test_authorization,
    validate_locked_test_population,
)
from src.llm_baseline.prompt import PROMPT_VERSION, prompt_fingerprint
from src.llm_baseline.records import (configuration_fingerprint, create_or_resume_run,
                                      sha256_file)
from src.llm_baseline.runner import reconstruction_audit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--criteria-path", type=Path, required=True)
    parser.add_argument("--pubmedbert-predictions", type=Path,
                        help="Optional frozen PubMedBERT per-record predictions for supplementary paired analysis.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--requests-per-minute", type=float, default=30.0,
                        help="Conservative serial Qwen request cap; must be in (0, 600].")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true",
                        help="Required in addition to --unlock-test before any Test file can be opened.")
    parser.add_argument("--unlock-test", metavar="ACKNOWLEDGEMENT",
                        help="Exact release acknowledgement; absent or wrong values refuse Test access.")
    args = parser.parse_args(argv)
    if args.max_length < 16:
        parser.error("max-length must be at least 16")
    if not 0 < args.requests_per_minute <= 600:
        parser.error("requests-per-minute must be in (0, 600]")
    if not args.execute:
        parser.error("Refusing to open Test data without --execute and --unlock-test")
    return args


def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction log is missing: {path}")
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(values)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    # This guard deliberately precedes every operation which could inspect a
    # Test path, including existence checks and hashing.
    require_final_test_authorization(args.unlock_test)
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite final-Test output directory: {args.output_dir}")

    # Crossing this call is the only Test CSV read boundary in this runner.
    test = load_authorized_test(args.test_split, args.unlock_test)
    data.validate_labels(test, "test")
    validate_locked_test_population(test)
    criteria, schema = data.load_eligibility_criteria(args.criteria_path)
    test = data.join_criteria(test, criteria, schema, "test")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    audit = reconstruction_audit(test, tokenizer, args.max_length)
    if audit["token_id_sequence_match"] != len(test) or audit["decoded_field_roundtrip_failures"]:
        raise ValueError("Final Test reconstruction audit failed; refusing any API request")

    config = {
        "experiment": "qwen3.8_flash_zero_shot", "phase": "final_test",
        "test_access": "explicitly_authorized", "test_evaluated": False,
        "model_name_or_path": args.model_name_or_path, "max_length": args.max_length,
        "prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_fingerprint(),
        "api": QwenConfig().public_dict(), "requests_per_minute": args.requests_per_minute,
        "input_sha256": {"test.csv": sha256_file(args.test_split),
                           "criteria.xlsx": sha256_file(args.criteria_path)},
        "tokenizer": _tokenizer_identity(args.model_name_or_path),
        "runtime": {"python": platform.python_version(), "transformers": transformers.__version__},
        "source": _git_metadata(),
        "population": {"records": len(test), "reviews": sorted(test[data.REVIEW_COLUMN].astype(str).unique()),
                       "sampling": "forbidden"},
    }
    if args.pubmedbert_predictions is not None:
        config["input_sha256"]["pubmedbert_predictions.csv"] = sha256_file(args.pubmedbert_predictions)
    manifest = create_or_resume_run(args.output_dir, config, args.resume)
    _write_json(args.output_dir / "reconstruction_audit.json", audit)
    result = execute_final_records(test, tokenizer, args.max_length,
                                   QwenClient(QwenConfig(), UrllibTransport()), args.output_dir,
                                   ConservativeRateLimiter(args.requests_per_minute))
    predictions = _read_jsonl(args.output_dir / "predictions.jsonl")
    metrics = evaluate_qwen_final(test, predictions)
    comparison = {
        "paired_analysis": "not_requested",
        "note": "PubMedBERT comparison metrics, if reported, come from its prior independent final Test evaluation; no paired error analysis was performed.",
    }
    if args.pubmedbert_predictions is not None:
        pubmedbert = pd.read_csv(args.pubmedbert_predictions, low_memory=False)
        _, paired = evaluate_final_gate(test, predictions, pubmedbert)
        paired.to_csv(args.output_dir / "paired_prediction_differences.csv", index=False)
        comparison = {"paired_analysis": "completed", "pubmedbert_prediction_path": str(args.pubmedbert_predictions)}
    pd.DataFrame(metrics["per_review"]).to_csv(args.output_dir / "per_review_qwen_metrics.csv", index=False)
    _write_json(args.output_dir / "qwen_final_test_metrics.json", {
        **metrics,
        "comparison_note": "PubMedBERT comparison metrics are from a prior independent final Test evaluation; this Qwen run does not require or perform paired error analysis unless supplementary predictions are explicitly supplied.",
        "supplementary_paired_analysis": comparison,
    })
    _write_json(args.output_dir / "final_test_execution_summary.json", {
        "manifest": manifest, "configuration_fingerprint": configuration_fingerprint(config),
        "test_evaluated": True, "complete_population_verified": True,
        "supplementary_paired_analysis": comparison, **result,
    })
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
