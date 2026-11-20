"""Verify and evaluate the frozen paired Qwen Validation input comparison.

This script reads existing predictions only. It never calls the Qwen API.
All validation gates run before the first result file is written.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_qwen_full_input_validation_comparison import _raw_fields, sample_validation
from src import screening_data as data
from src.llm_baseline.client import QwenConfig
from src.llm_baseline.evaluation import POLICY_OUTCOMES
from src.llm_baseline.prompt import PROMPT_VERSION, build_prompt, build_prompt_fields, prompt_fingerprint
from src.llm_baseline.records import configuration_fingerprint, record_id_for_row, sha256_file
from src.llm_baseline.reconstruction import _field_texts, reconstruct_input


FIELDS = ("title", "abstract", "inclusion", "exclusion")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def historical_or_public_json(path: Path) -> Path:
    if path.is_file():
        return path
    relative = path.relative_to(ROOT)
    public = ROOT / "results/reproducibility/qwen_historical_public" / relative.with_name(relative.stem + ".public.json")
    require(public.is_file(), f"Historical JSON and public copy are both absent: {relative}")
    record = read_json(public)
    require(record.get("_public_copy", {}).get("original_repository_path") == relative.as_posix(),
            f"Public copy has the wrong original path: {public}")
    return public


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def metrics(rows: list[dict], mode: str) -> dict:
    counts = Counter()
    for row in rows:
        gold = row["gold_label"]
        retained = row[f"{mode}_retained"]
        counts[(gold, retained)] += 1
    tp, fp, tn, fn = counts[(1, 1)], counts[(0, 1)], counts[(0, 0)], counts[(1, 0)]
    n = len(rows)
    require(tp + fp + tn + fn == n, "Confusion matrix does not cover the group")
    positives = tp + fn
    return {
        "n": n, "positives": positives, "negatives": tn + fp,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall": tp / positives if positives else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "fnr": fn / positives if positives else None,
        "workload_reduction": (tn + fn) / n if n else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=ROOT / "outputs/qwen_full_input_validation_comparison")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/qwen_input_length_sensitivity")
    args = parser.parse_args()
    prepared = args.prepared_dir.resolve()
    output = args.output_dir.resolve()
    require(not output.exists(), f"Refusing to overwrite results: {output}")

    paths = {
        "experiment_config": historical_or_public_json(prepared / "experiment_config.json"),
        "sample_manifest": prepared / "sample_manifest.csv",
        "sampling_summary": prepared / "sampling_summary.json",
        "truncation_audit": prepared / "truncation_audit.csv",
        "truncation_summary": prepared / "truncation_summary.json",
        "full_manifest": historical_or_public_json(prepared / "predictions_full/run_manifest.json"),
        "full_predictions": prepared / "predictions_full/predictions.jsonl",
        "truncated_manifest": historical_or_public_json(prepared / "predictions_truncated/run_manifest.json"),
        "truncated_predictions": prepared / "predictions_truncated/predictions.jsonl",
        "validation_csv": ROOT / "data/processed/splits/validation.csv",
        "criteria_xlsx": next((ROOT / "data").glob("eligibility_criteria*.xlsx")),
    }
    config = read_json(paths["experiment_config"])
    runs = {mode: read_json(paths[f"{mode}_manifest"]) for mode in ("full", "truncated")}
    predictions = {mode: read_jsonl(paths[f"{mode}_predictions"]) for mode in ("full", "truncated")}
    manifest = pd.read_csv(paths["sample_manifest"], low_memory=False)
    audit = pd.read_csv(paths["truncation_audit"], low_memory=False)

    require(config["sample"] == {"records": 5000, "include": 300, "exclude": 4700, "seed": 42}, "Unexpected sample specification")
    require(config["prompt_version"] == PROMPT_VERSION == "qwen-zero-shot-screening-v2", "Prompt version differs")
    require(config["prompt_sha256"] == prompt_fingerprint(), "Prompt template fingerprint differs")
    require(config["screening_policy"] == "retain_unresolved_v1", "Screening policy differs")
    # The archived base URL records the historical routing, whereas the current
    # client may use a shared default or a private QWEN_BASE_URL. Compare the
    # experiment parameters without substituting today's endpoint for history.
    endpoint_fields = {"base_url", "endpoint_kind", "endpoint_sha256"}
    archived_api = {key: value for key, value in config["model"].items() if key not in endpoint_fields}
    current_api = {key: value for key, value in QwenConfig(max_tokens=128).public_dict().items() if key not in endpoint_fields}
    require(archived_api == current_api, "Prepared API parameters differ from the executor")
    require(config["input_sha256"]["validation.csv"] == sha256_file(paths["validation_csv"]), "Validation CSV hash differs")
    require(config["input_sha256"]["criteria.xlsx"] == sha256_file(paths["criteria_xlsx"]), "Criteria workbook hash differs")
    require(config["tokenizer"]["files_sha256"] == {
        name: sha256_file(ROOT / "model/pubmedbert" / name)
        for name in config["tokenizer"]["files_sha256"]
    }, "PubMedBERT tokenizer files differ")

    public_endpoint_hashes = [
        item.get("_public_copy", {}).get("historical_endpoint_sha256")
        for item in (config, runs["full"], runs["truncated"])
    ]
    if any(public_endpoint_hashes):
        require(all(public_endpoint_hashes) and len(set(public_endpoint_hashes)) == 1,
                "Public copies do not establish one historical endpoint")

    common = None
    for mode in ("full", "truncated"):
        run = runs[mode]
        cfg = run["configuration"]
        require(cfg["phase"] == "validation_comparison_execution" and cfg["mode"] == mode and cfg["records"] == 5000, f"{mode}: run phase/mode/size differs")
        require(cfg["api"] == config["model"], f"{mode}: API configuration differs")
        require(cfg["requests_per_minute"] == 30.0 and cfg["technical_selection_record_ids"] is None, f"{mode}: request schedule or record selection differs")
        require(cfg["context_check"] == {"method": "utf8_bytes_plus_chat_allowance_not_token_count", "safe_limit_bytes": 900000}, f"{mode}: context check differs")
        require(run["configuration_fingerprint"] == configuration_fingerprint(cfg), f"{mode}: run fingerprint differs")
        without_mode = copy.deepcopy(cfg)
        del without_mode["mode"]
        if common is None:
            common = without_mode
        else:
            require(without_mode == common, "Full and Truncated run configurations differ beyond input mode")
        values = predictions[mode]
        ids = [value["record_id"] for value in values]
        require(len(values) == len(set(ids)) == 5000, f"{mode}: incomplete or duplicate prediction IDs")
        require(all(value.get("returned_model") in (cfg["api"]["model"], None) for value in values), f"{mode}: returned model ID differs")

    require(len(manifest) == len(audit) == 5000, "Manifest or truncation audit is incomplete")
    require(manifest.record_id.is_unique and audit.record_id.is_unique, "Duplicate manifest/audit ID")
    require(manifest.source_row_index.is_unique, "Duplicate Validation source row")
    expected_ids = manifest.record_id.astype(str).tolist()
    require(predictions["full"] and predictions["truncated"], "Predictions are empty")
    for mode in ("full", "truncated"):
        require([row["record_id"] for row in predictions[mode]] == expected_ids, f"{mode}: prediction order or IDs differ from sample")
    require(audit.record_id.astype(str).tolist() == expected_ids, "Truncation audit order or IDs differ from sample")
    require(manifest.gold_label.value_counts().to_dict() == {0: 4700, 1: 300}, "Sample label totals differ")

    validation = data.read_split(paths["validation_csv"]).reset_index(drop=True)
    data.validate_labels(validation, "validation")
    require(len(validation) == 22397, "Validation population size differs")
    criteria, schema = data.load_eligibility_criteria(paths["criteria_xlsx"])
    joined = data.join_criteria(validation, criteria, schema, "validation")
    sample, _ = sample_validation(joined)
    require(sample["_qwen_source_row_index"].tolist() == manifest.source_row_index.astype(int).tolist(), "Recreated sample row indices differ")
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model/pubmedbert", local_files_only=True)

    paired: list[dict] = []
    status_counts = {mode: Counter() for mode in ("full", "truncated")}
    anomaly_counts = {mode: Counter() for mode in ("full", "truncated")}
    field_counts = Counter()
    input_hash_matches = {"full": 0, "truncated": 0}
    for i, (_, source) in enumerate(sample.iterrows()):
        row = source.to_dict()
        record_id = record_id_for_row(row, i)
        manifest_row = manifest.iloc[i]
        audit_row = audit.iloc[i]
        gold = int(row[data.LABEL_COLUMN])
        source_index = int(row["_qwen_source_row_index"])
        require(record_id == expected_ids[i] and record_id == audit_row.record_id, f"Row {i}: source ID differs")
        require(source_index == int(manifest_row.source_row_index), f"Row {i}: source index differs")
        require(str(row[data.REVIEW_COLUMN]) == str(manifest_row.review_key) == str(audit_row.review_key), f"Row {i}: review differs")
        require(gold == int(manifest_row.gold_label) == int(audit_row.gold_label), f"Row {i}: gold label differs")
        for identifier in data.IDENTIFIER_COLUMNS:
            require(data.safe_text(row.get(identifier)) == data.safe_text(manifest_row.get(identifier)), f"Row {i}: {identifier} differs")

        raw = _raw_fields(row)
        reconstructed = reconstruct_input(tokenizer, row, 512)
        prompt = {
            "full": build_prompt_fields(**raw),
            "truncated": build_prompt(reconstructed),
        }
        source_texts = _field_texts(row)
        flags = {}
        for field in FIELDS:
            original = len(tokenizer.encode(source_texts[field], add_special_tokens=False))
            retained = len(reconstructed.token_ids[field])
            cut = original > retained
            require(original == int(audit_row[f"{field}_original_tokens"]), f"Row {i}: {field} original token count differs")
            require(retained == int(audit_row[f"{field}_retained_tokens"]), f"Row {i}: {field} retained token count differs")
            require(cut == bool(audit_row[f"{field}_truncated"]), f"Row {i}: {field} truncation flag differs")
            flags[field] = cut
            field_counts[field] += int(cut)
        any_cut = any(flags.values())
        require(any_cut == bool(audit_row.any_truncated), f"Row {i}: any-truncated flag differs")
        field_counts["any_truncated"] += int(any_cut)
        field_counts["none_truncated"] += int(not any_cut)
        field_counts["multiple_fields_truncated"] += int(sum(flags.values()) > 1)

        result = {
            "record_id": record_id, "source_row_index": source_index,
            "review_key": str(row[data.REVIEW_COLUMN]), "gold_label": gold,
            "any_truncated": any_cut,
            **{f"{field}_truncated": flags[field] for field in FIELDS},
            **{f"{field}_original_tokens": int(audit_row[f"{field}_original_tokens"]) for field in FIELDS},
            **{f"{field}_retained_tokens": int(audit_row[f"{field}_retained_tokens"]) for field in FIELDS},
        }
        for mode in ("full", "truncated"):
            value = predictions[mode][i]
            require(value["record_id"] == record_id and int(value["source_row_index"]) == source_index, f"Row {i}: {mode} prediction identity differs")
            require(value["review_name"] == result["review_key"] and int(value["gold_label"]) == gold, f"Row {i}: {mode} prediction review/label differs")
            expected_hash = hashlib.sha256(prompt[mode].encode("utf-8")).hexdigest()
            require(value.get("input_sha256") == expected_hash, f"Row {i}: {mode} exact prompt hash differs")
            input_hash_matches[mode] += 1
            parsed = (value.get("status"), value.get("decision"))
            require(parsed in POLICY_OUTCOMES, f"Row {i}: {mode} has unknown status/decision: {parsed}")
            retained, category = POLICY_OUTCOMES[parsed]
            status_counts[mode][category] += 1
            if value.get("transport_error"):
                anomaly_counts[mode]["api_failure"] += 1
            if value.get("parse_error"):
                anomaly_counts[mode]["parse_error"] += 1
            if value.get("finish_reason") == "length":
                anomaly_counts[mode]["finish_reason_length"] += 1
            if value.get("output_at_max_tokens"):
                anomaly_counts[mode]["output_at_max_tokens"] += 1
            result.update({
                f"{mode}_status": value["status"],
                f"{mode}_decision": value["decision"],
                f"{mode}_retained": retained,
                f"{mode}_outcome_category": category,
                f"{mode}_retry_count": int(value["retry_count"]),
                f"{mode}_input_sha256": expected_hash,
            })
        result["prediction_changed"] = result["full_retained"] != result["truncated_retained"]
        result["decision_changed"] = result["full_decision"] != result["truncated_decision"]
        result["fn_repaired_by_full"] = gold == 1 and result["truncated_retained"] == 0 and result["full_retained"] == 1
        result["fn_added_by_full"] = gold == 1 and result["truncated_retained"] == 1 and result["full_retained"] == 0
        paired.append(result)

    stored_truncation = read_json(paths["truncation_summary"])
    require(stored_truncation["records"] == 5000, "Stored truncation summary size differs")
    for field in (*FIELDS, "any_truncated", "none_truncated"):
        require(stored_truncation["overall"][field]["count"] == field_counts[field], f"Stored truncation summary differs for {field}")
    require(stored_truncation["overall"]["multiple_fields_truncated"] == field_counts["multiple_fields_truncated"], "Stored multiple-field count differs")

    groups = {
        "all": paired,
        "any_truncated": [row for row in paired if row["any_truncated"]],
        "none_truncated": [row for row in paired if not row["any_truncated"]],
    }
    summary_rows = []
    group_metrics = {}
    for group_name, group in groups.items():
        group_metrics[group_name] = {}
        for mode in ("truncated", "full"):
            calculated = metrics(group, mode)
            group_metrics[group_name][mode] = calculated
            summary_rows.append({"group": group_name, "mode": mode, **calculated})
    changes = {
        "final_prediction_changed": sum(row["prediction_changed"] for row in paired),
        "raw_decision_changed": sum(row["decision_changed"] for row in paired),
        "fn_repaired_by_full": sum(row["fn_repaired_by_full"] for row in paired),
        "fn_added_by_full": sum(row["fn_added_by_full"] for row in paired),
        "truncated_exclude_to_full_retain": sum(row["truncated_retained"] == 0 and row["full_retained"] == 1 for row in paired),
        "truncated_retain_to_full_exclude": sum(row["truncated_retained"] == 1 and row["full_retained"] == 0 for row in paired),
    }
    deltas = {name: group_metrics["all"]["full"][name] - group_metrics["all"]["truncated"][name]
              for name in ("tp", "fp", "tn", "fn", "recall", "precision", "fnr", "workload_reduction")}

    source_hashes = {name: sha256_file(path) for name, path in paths.items() if path.is_file()}
    verification = {
        "scope": "frozen paired Validation predictions; no API calls or model inference",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "validation_population": len(validation), "sample_records": len(paired),
        "sample_positive_records": group_metrics["all"]["full"]["positives"],
        "same_record_ids_order_review_labels": True,
        "prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_fingerprint(),
        "matching_exact_prompt_hashes": input_hash_matches,
        "same_api_configuration_except_mode": True,
        "api_configuration": common["api"],
        "requests_per_minute": common["requests_per_minute"],
        "request_timeout_seconds": 90,
        "screening_policy": config["screening_policy"],
        "run_created_utc": {mode: runs[mode]["created_utc"] for mode in runs},
        "prediction_first_last_utc": {mode: {
            "first": predictions[mode][0]["completed_utc"],
            "last": predictions[mode][-1]["completed_utc"],
        } for mode in predictions},
        "outcome_counts": {mode: dict(status_counts[mode]) for mode in status_counts},
        "anomaly_counts": {mode: dict(anomaly_counts[mode]) for mode in anomaly_counts},
        "source_files": {name: {"path": str(path), "sha256": source_hashes[name]} for name, path in paths.items() if name in source_hashes},
        "tokenizer_files_sha256": config["tokenizer"]["files_sha256"],
        "input_difference_note": "Full uses raw field strings; Truncated uses PubMedBERT WordPiece prefix decoding with section markers. The comparison is between these two input pipelines, not an isolated length-only intervention.",
        "sampling_note": "Validation sample is stratified to 300 positive and 4700 negative records; precision and workload reduction describe this sample, not the complete Validation population.",
    }
    summary = {
        "group_metrics": group_metrics,
        "difference_full_minus_truncated": deltas,
        "paired_changes": changes,
        "outcome_counts": verification["outcome_counts"],
        "anomaly_counts": verification["anomaly_counts"],
        "truncation_counts": dict(field_counts),
    }

    output.mkdir(parents=True)
    pd.DataFrame(paired).to_csv(output / "paired_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(output / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    for name, content in (
        ("metrics_summary.json", summary),
        ("truncation_summary.json", stored_truncation),
        ("source_verification.json", verification),
    ):
        (output / name).write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "metrics": group_metrics["all"], "changes": changes,
                      "truncation_counts": dict(field_counts), "outcome_counts": verification["outcome_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
