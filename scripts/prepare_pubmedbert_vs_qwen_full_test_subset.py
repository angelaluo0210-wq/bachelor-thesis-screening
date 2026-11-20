"""Prepare the fixed Test subset comparison without model/API inference."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_qwen_full_input_validation_comparison import _raw_fields
from scripts.run_qwen_baseline import _git_metadata
from scripts.run_qwen_full_input_validation_comparison import conservative_context_audit
from src import screening_data as data
from src.llm_baseline.client import QwenConfig
from src.llm_baseline.prompt import PROMPT_VERSION, build_prompt_fields, prompt_fingerprint
from src.llm_baseline.records import record_id_for_row, sha256_file

SEED = 42
TARGET = 5000


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def largest_remainder(weights: pd.Series, total: int) -> dict[str, int]:
    if total < 0 or total > int(weights.sum()):
        raise ValueError("Requested quota is infeasible")
    exact = {str(k): total * int(v) / int(weights.sum()) for k, v in weights.items()}
    allocation = {k: int(math.floor(v)) for k, v in exact.items()}
    for key in sorted(exact, key=lambda k: (-(exact[k] - allocation[k]), k))[:total - sum(allocation.values())]:
        allocation[key] += 1
    if any(allocation[str(k)] > int(v) for k, v in weights.items()):
        raise ValueError("Allocation exceeds stratum availability")
    return allocation


def sample_test(test: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    indexed = test.copy().reset_index(drop=True)
    indexed["_qwen_source_row_index"] = range(len(indexed))
    data.validate_labels(indexed, "test")
    review_counts = indexed.groupby(data.REVIEW_COLUMN, sort=True).size()
    review_quota = largest_remainder(review_counts, TARGET)
    rng = np.random.default_rng(SEED)
    selected: list[pd.DataFrame] = []
    label_quotas: dict[str, dict[str, int]] = {}
    for review in sorted(review_quota):
        group = indexed[indexed[data.REVIEW_COLUMN].astype(str).eq(review)]
        # A second, within-review largest-remainder allocation preserves each
        # Review's observed label composition while meeting its exact quota.
        counts = group.groupby(data.LABEL_COLUMN, sort=True).size()
        quota = largest_remainder(counts, review_quota[review])
        label_quotas[review] = quota
        for label in sorted(quota):
            candidates = group[group[data.LABEL_COLUMN].eq(int(label))]
            positions = rng.choice(candidates.index.to_numpy(), size=quota[label], replace=False)
            selected.append(indexed.loc[positions])
    sample = pd.concat(selected).sort_values("_qwen_source_row_index").reset_index(drop=True)
    if len(sample) != TARGET or sample["_qwen_source_row_index"].duplicated().any():
        raise ValueError("Test sampling did not produce 5,000 unique records")
    summary = {"method": "hierarchical largest-remainder allocation: review size, then label distribution within review",
               "random_seed": SEED, "target_records": TARGET,
               "test_by_review_label": {str(r): {str(k): int(v) for k, v in g.groupby(data.LABEL_COLUMN).size().items()} for r, g in indexed.groupby(data.REVIEW_COLUMN, sort=True)},
               "sample_by_review_label": {str(r): {str(k): int(v) for k, v in g.groupby(data.LABEL_COLUMN).size().items()} for r, g in sample.groupby(data.REVIEW_COLUMN, sort=True)},
               "review_quotas": review_quota, "within_review_label_quotas": label_quotas,
               "test_records": len(indexed), "sample_records": len(sample), "reviews": sorted(sample[data.REVIEW_COLUMN].astype(str).unique())}
    return sample, summary


def main() -> int:
    out = ROOT / "outputs" / "pubmedbert_vs_qwen_full_test_subset"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    test_path = ROOT / "data" / "processed" / "splits" / "test.csv"
    prediction_path = ROOT / "outputs" / "final_test" / "final_test_predictions.csv"
    final_config_path = ROOT / "outputs" / "final_test" / "final_test_config.json"
    criteria_path = ROOT / "data" / "eligibility_criteria(排纳标准).xlsx"
    test = data.read_split(test_path)
    sample, sampling = sample_test(test)
    manifest = sample[["_qwen_source_row_index", data.REVIEW_COLUMN, data.LABEL_COLUMN, *data.IDENTIFIER_COLUMNS]].copy()
    manifest.insert(0, "record_id", [record_id_for_row(row.to_dict(), i) for i, (_, row) in enumerate(sample.iterrows())])
    manifest = manifest.rename(columns={"_qwen_source_row_index": "test_source_row_index", data.REVIEW_COLUMN: "review_key", data.LABEL_COLUMN: "gold_label"})
    if manifest.record_id.duplicated().any(): raise ValueError("Manifest record IDs are not unique")
    predictions = pd.read_csv(prediction_path, low_memory=False).rename(columns={"review_name": "review_key", "true_label": "gold_label"})
    prediction_ids = [record_id_for_row(row.to_dict(), i) for i, (_, row) in enumerate(predictions.iterrows())]
    predictions.insert(0, "record_id", prediction_ids)
    if predictions.record_id.duplicated().any(): raise ValueError("Historical PubMedBERT prediction IDs are not unique")
    extracted = manifest.merge(predictions[["record_id", "review_key", "gold_label", "predicted_probability", "predicted_label"]], on="record_id", how="left", suffixes=("_manifest", "_prediction"), validate="one_to_one")
    if extracted.predicted_probability.isna().any(): raise ValueError("A sampled Test record has no PubMedBERT prediction")
    if not (extracted.review_key_manifest == extracted.review_key_prediction).all() or not (extracted.gold_label_manifest == extracted.gold_label_prediction).all():
        raise ValueError("Historical prediction review key or gold label disagrees with Test manifest")
    final_config = json.loads(final_config_path.read_text(encoding="utf-8"))
    threshold = float(final_config["threshold"])
    if not ((extracted.predicted_probability >= threshold).astype(int) == extracted.predicted_label.astype(int)).all():
        raise ValueError("Historical predicted_label disagrees with frozen Test threshold")
    criteria, schema = data.load_eligibility_criteria(criteria_path)
    qwen = data.join_criteria(sample, criteria, schema, "test_subset_preflight")
    prompt_audits = []
    for i, (_, row) in enumerate(qwen.iterrows()):
        fields = _raw_fields(row.to_dict())
        prompt_audits.append(conservative_context_audit(build_prompt_fields(**fields)))
    out.mkdir(parents=True)
    manifest.to_csv(out / "sample_manifest.csv", index=False)
    extracted = extracted.drop(columns=["review_key_prediction", "gold_label_prediction"]).rename(columns={"review_key_manifest": "review_key", "gold_label_manifest": "gold_label"})
    extracted.to_csv(out / "pubmedbert_predictions_subset.csv", index=False)
    manifest_hash = sha256_file(out / "sample_manifest.csv")
    sampling["sample_manifest_sha256"] = manifest_hash
    _write_json(out / "sampling_summary.json", sampling)
    config = {"experiment": "pubmedbert_vs_qwen_full_test_subset", "phase": "prepared_not_executed", "sample_manifest_sha256": manifest_hash,
              "pubmedbert": {"historical_prediction_path": str(prediction_path), "historical_prediction_sha256": sha256_file(prediction_path), "final_config_path": str(final_config_path), "selected_checkpoint": final_config["selected_checkpoint"], "checkpoint_sha256": final_config["checkpoint_sha256"], "threshold": threshold, "max_length": final_config["max_length"]},
              "qwen": {"mode": "full", "api": QwenConfig(max_tokens=128).public_dict(), "prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_fingerprint(), "screening_policy": "retain_unresolved_v1", "requests_per_minute": 30.0, "context_check": {"method": "utf8_bytes_plus_chat_allowance_not_token_count", "safe_limit_bytes": 900000}},
              "inputs": {"test.csv": sha256_file(test_path), "criteria.xlsx": sha256_file(criteria_path)}, "source": _git_metadata(), "runtime": {"python": platform.python_version()}}
    _write_json(out / "experiment_config.json", config)
    missing_title = int(qwen.title.isna().sum() + qwen.title.fillna("").astype(str).str.strip().eq("").sum() - qwen.title.isna().sum())
    missing_abstract = int(qwen.abstract.isna().sum() + qwen.abstract.fillna("").astype(str).str.strip().eq("").sum() - qwen.abstract.isna().sum())
    report = f"# Test-subset preflight\n\n- Records: {len(sample)} unique records from the frozen formal Test split.\n- Sampling: hierarchical largest-remainder allocation; seed {SEED}.\n- PubMedBERT historical predictions: all {len(extracted)} records matched by stable ID; frozen threshold `{threshold}` reproduced every stored label.\n- Qwen: Full raw fields, identical validated Full-Validation prompt/API configuration; no label is placed in the prompt.\n- Criteria: all sampled reviews matched; unusable criteria were rejected by the shared join function.\n- Missing title: {missing_title}; missing abstract: {missing_abstract}.\n- Conservative context check: max checked UTF-8 bytes = {max(a['checked_utf8_bytes'] for a in prompt_audits)}; configured safety limit = 900000.\n- Planned Qwen API requests: 5,000. No API request has been made by this preparation script.\n"
    (out / "preflight_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "records": len(sample), "manifest_sha256": manifest_hash, "review_quotas": sampling["review_quotas"]}, ensure_ascii=False))
    return 0

if __name__ == "__main__": raise SystemExit(main())
