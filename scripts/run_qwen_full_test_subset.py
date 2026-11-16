"""Explicitly authorized Qwen Full inference for the prepared fixed Test subset."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_qwen_full_input_validation_comparison import _raw_fields
from scripts.run_qwen_full_input_validation_comparison import conservative_context_audit
from src import screening_data as data
from src.llm_baseline.client import QwenClient, QwenConfig, UrllibTransport
from src.llm_baseline.final_test import ConservativeRateLimiter
from src.llm_baseline.prompt import build_prompt_fields
from src.llm_baseline.records import append_prediction, completed_record_ids, create_or_resume_run, record_id_for_row, require_unique_record_ids, sha256_file
from src.llm_baseline.runner import _response_metadata, _usage


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--prepared-dir", type=Path, default=ROOT / "outputs" / "pubmedbert_vs_qwen_full_test_subset")
    p.add_argument("--test-split", type=Path, default=ROOT / "data" / "processed" / "splits" / "test.csv")
    p.add_argument("--criteria-path", type=Path, default=ROOT / "data" / "eligibility_criteria(排纳标准).xlsx")
    p.add_argument("--requests-per-minute", type=float, default=30.0)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if not args.execute: p.error("Refusing any Qwen API request without --execute")
    if not 0 < args.requests_per_minute <= 600: p.error("requests-per-minute must be in (0, 600]")
    manifest_path = args.prepared_dir / "sample_manifest.csv"
    config_path = args.prepared_dir / "experiment_config.json"
    if not manifest_path.is_file() or not config_path.is_file(): raise FileNotFoundError("Prepared manifest/config is missing")
    prepared = json.loads(config_path.read_text(encoding="utf-8"))
    if sha256_file(manifest_path) != prepared["sample_manifest_sha256"]: raise ValueError("Sample manifest hash differs from prepared configuration")
    manifest = pd.read_csv(manifest_path, low_memory=False)
    if len(manifest) != 5000 or manifest.record_id.duplicated().any(): raise ValueError("Prepared manifest is incomplete or duplicate")
    test = data.read_split(args.test_split).reset_index(drop=True); test["_qwen_source_row_index"] = range(len(test))
    test_ids = [record_id_for_row(row.to_dict(), i) for i, (_, row) in enumerate(test.iterrows())]
    if len(test_ids) != len(set(test_ids)): raise ValueError("Test stable IDs are duplicate")
    test_by_id = test.assign(record_id=test_ids).set_index("record_id", drop=False)
    expected = manifest.record_id.astype(str).tolist()
    if not set(expected).issubset(set(test_by_id.index)): raise ValueError("Manifest includes records absent from current Test split")
    sample = test_by_id.loc[expected].reset_index(drop=True)
    if [record_id_for_row(row.to_dict(), i) for i, (_, row) in enumerate(sample.iterrows())] != expected: raise ValueError("Manifest/Test stable-ID reconstruction mismatch")
    if not (sample[data.LABEL_COLUMN].astype(int).to_numpy() == manifest.gold_label.astype(int).to_numpy()).all(): raise ValueError("Manifest gold labels differ from Test split")
    criteria, schema = data.load_eligibility_criteria(args.criteria_path)
    sample = data.join_criteria(sample, criteria, schema, "test_subset")
    prompts = []
    for _, row in sample.iterrows():
        prompt = build_prompt_fields(**_raw_fields(row.to_dict()))
        prompts.append((prompt, conservative_context_audit(prompt)))
    run_dir = args.prepared_dir / "predictions_qwen_full"
    run_config = {"phase": "test_subset_qwen_full", "records": len(sample), "sample_manifest_sha256": prepared["sample_manifest_sha256"],
                  "input_sha256": {"test.csv": sha256_file(args.test_split), "criteria.xlsx": sha256_file(args.criteria_path)},
                  "api": QwenConfig(max_tokens=128).public_dict(), "requests_per_minute": args.requests_per_minute,
                  "prompt_version": prepared["qwen"]["prompt_version"], "prompt_sha256": prepared["qwen"]["prompt_sha256"],
                  "screening_policy": "retain_unresolved_v1", "context_check": prepared["qwen"]["context_check"]}
    create_or_resume_run(run_dir, run_config, args.resume)
    rows = [row.to_dict() for _, row in sample.iterrows()]; ids = require_unique_record_ids(rows); complete = completed_record_ids(run_dir)
    if not complete.issubset(set(ids)): raise ValueError("Stored predictions are outside the fixed Test subset")
    client, limiter, consecutive = QwenClient(QwenConfig(max_tokens=128), UrllibTransport()), ConservativeRateLimiter(args.requests_per_minute), 0
    for i, (row, (prompt, context)) in enumerate(zip(rows, prompts)):
        record_id = record_id_for_row(row, i)
        if record_id in complete: continue
        parsed, response, retries, transport_error = client.classify_with_retries(prompt, before_attempt=limiter.wait)
        metadata, usage = _response_metadata(response), _usage(response)
        output_at_limit = bool(isinstance(usage, dict) and usage.get("completion_tokens") == 128)
        append_prediction(run_dir, {"record_id": record_id, "source_row_index": int(row["_qwen_source_row_index"]), "review_name": str(row[data.REVIEW_COLUMN]), "gold_label": int(row[data.LABEL_COLUMN]), "status": parsed.status, "decision": parsed.decision, "parse_error": parsed.error, "transport_error": transport_error, "retry_count": retries, "input_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(), "context_audit": context, "usage": usage, "output_at_max_tokens": output_at_limit, **metadata, "estimated_cost_usd": None, "completed_utc": datetime.now(timezone.utc).isoformat()}, known_ids=complete)
        consecutive = 0 if parsed.status == "completed" else consecutive + 1
        if consecutive >= 3: raise RuntimeError("Three consecutive exhausted technical failures; stopping before further API requests")
    print(json.dumps({"completed_records": len(completed), "expected_records": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())
