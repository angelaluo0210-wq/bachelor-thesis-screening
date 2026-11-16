"""Prepare, but never execute, the paired Qwen Validation input comparison.

The script samples exactly 300 included and 4,700 excluded Validation records
with seed 42, reuses the production PubMedBERT reconstruction for mode A, and
keeps raw source criteria/title/abstract for mode B.  It makes no network or
Qwen API request.  Run it once before the separately authorized executor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import transformers
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_qwen_baseline import _git_metadata, _tokenizer_identity
from src import screening_data as data
from src.llm_baseline.client import QwenConfig
from src.llm_baseline.prompt import PROMPT_VERSION, build_prompt_fields, prompt_fingerprint
from src.llm_baseline.records import configuration_fingerprint, record_id_for_row, sha256_file
from src.llm_baseline.reconstruction import reconstruct_input


TARGETS = {1: 300, 0: 4700}
SEED = 42


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _quota(counts: pd.Series, target: int) -> dict[str, int]:
    """Largest-remainder proportional allocation, guaranteed feasible."""
    total = int(counts.sum())
    if total < target:
        raise ValueError(f"Only {total} records are available for a target of {target}")
    exact = {str(key): target * int(value) / total for key, value in counts.items()}
    result = {key: int(math.floor(value)) for key, value in exact.items()}
    remainder = target - sum(result.values())
    # Review key resolves ties deterministically; all groups here are nonempty.
    for key in sorted(exact, key=lambda name: (-(exact[name] - result[name]), name))[:remainder]:
        result[key] += 1
    if any(result[str(key)] > int(value) for key, value in counts.items()):
        raise ValueError("Stratified allocation exceeds an available review/label stratum")
    return result


def sample_validation(validation: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    indexed = validation.copy().reset_index(drop=True)
    indexed["_qwen_source_row_index"] = range(len(indexed))
    data.validate_labels(indexed, "validation")
    rng = np.random.default_rng(SEED)
    selected: list[pd.DataFrame] = []
    quotas: dict[str, dict[str, int]] = {}
    available: dict[str, dict[str, int]] = {}
    for label, target in TARGETS.items():
        subset = indexed.loc[indexed[data.LABEL_COLUMN].eq(label)]
        counts = subset.groupby(data.REVIEW_COLUMN, sort=True).size()
        allocation = _quota(counts, target)
        quotas[str(label)] = allocation
        available[str(label)] = {str(key): int(value) for key, value in counts.items()}
        for review in sorted(allocation):
            candidates = subset.loc[subset[data.REVIEW_COLUMN].astype(str).eq(review)]
            positions = rng.choice(candidates.index.to_numpy(), size=allocation[review], replace=False)
            selected.append(indexed.loc[positions])
    sample = pd.concat(selected, ignore_index=False).sort_values("_qwen_source_row_index").reset_index(drop=True)
    if len(sample) != sum(TARGETS.values()) or sample["_qwen_source_row_index"].duplicated().any():
        raise ValueError("Sampling did not produce a unique 5,000-record manifest")
    if sample.groupby(data.LABEL_COLUMN).size().to_dict() != TARGETS:
        raise ValueError("Sampling did not meet the requested label totals")
    summary = {
        "method": "stratified random sample by review_key and gold_label; largest-remainder proportional quotas",
        "random_seed": SEED, "targets": {"include": TARGETS[1], "exclude": TARGETS[0]},
        "available_by_label_review": available, "allocated_by_label_review": quotas,
        "selected_by_label_review": {
            str(label): {str(key): int(value) for key, value in sample.loc[sample[data.LABEL_COLUMN].eq(label)].groupby(data.REVIEW_COLUMN).size().items()}
            for label in (1, 0)
        },
        "records": int(len(sample)), "unique_source_records": int(sample["_qwen_source_row_index"].nunique()),
        "reviews": sorted(sample[data.REVIEW_COLUMN].astype(str).unique()),
        "note": "This is a positive-enriched stratified sample; its class mix is not the Validation population prevalence.",
    }
    return sample, summary


def _raw_fields(row: dict[str, Any]) -> dict[str, str]:
    # Only safe_text's documented NA-to-empty/basic-trim behaviour is allowed.
    return {"inclusion": data.safe_text(row.get("_inclusion_text", "")),
            "exclusion": data.safe_text(row.get("_exclusion_text", "")),
            "title": data.safe_text(row.get(data.TITLE_COLUMN, "")),
            "abstract": data.safe_text(row.get(data.ABSTRACT_COLUMN, ""))}


def _legacy_field_source_text(row: dict[str, Any], raw: dict[str, str], field: str) -> str:
    """Return exactly the pre-tokenization section text of the frozen path."""
    if field == "title":
        return f"[TITLE]\n{raw['title']}"
    if field == "abstract":
        return f"[ABSTRACT]\n{raw['abstract']}"
    if field == "exclusion":
        return f"[EXCLUSION]\n{raw['exclusion']}"
    review = data.safe_text(row.get("_review_text", ""))
    parts = [f"[REVIEW]\n{review}"] if review else []
    parts.append(f"[INCLUSION]\n{raw['inclusion']}")
    return "\n\n".join(parts)


def audit_truncation(sample: pd.DataFrame, tokenizer: Any, max_length: int) -> tuple[pd.DataFrame, dict[str, Any], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    prompt_chars = {"truncated": [], "full": []}
    for fallback_index, (_, source) in enumerate(sample.iterrows()):
        row = source.to_dict()
        reconstructed = reconstruct_input(tokenizer, row, max_length)
        raw = _raw_fields(row)
        record = {"record_id": record_id_for_row(row, fallback_index), "review_key": str(row[data.REVIEW_COLUMN]),
                  "gold_label": int(row[data.LABEL_COLUMN]), "abstract_missing": not raw["abstract"]}
        for field in ("title", "abstract", "inclusion", "exclusion"):
            # Reconstruction token IDs include exactly the field marker and, for
            # inclusion, any legacy review text, matching the original pipeline.
            original = len(tokenizer.encode(_legacy_field_source_text(row, raw, field), add_special_tokens=False))
            retained = len(reconstructed.token_ids[field])
            record[f"{field}_original_tokens"] = original
            record[f"{field}_retained_tokens"] = retained
            record[f"{field}_truncated"] = original > retained
        record["any_truncated"] = any(record[f"{field}_truncated"] for field in ("title", "abstract", "inclusion", "exclusion"))
        rows.append(record)
        prompt_chars["truncated"].append(len(build_prompt_fields(inclusion=reconstructed.inclusion, exclusion=reconstructed.exclusion, title=reconstructed.title, abstract=reconstructed.abstract)))
        prompt_chars["full"].append(len(build_prompt_fields(**raw)))
    audit = pd.DataFrame(rows)
    fields = ("title", "abstract", "inclusion", "exclusion")
    def group_stats(frame: pd.DataFrame) -> dict[str, Any]:
        return {field: {"count": int(frame[f"{field}_truncated"].sum()), "proportion": float(frame[f"{field}_truncated"].mean())} for field in fields} | {
            "any_truncated": {"count": int(frame["any_truncated"].sum()), "proportion": float(frame["any_truncated"].mean())},
            "none_truncated": {"count": int((~frame["any_truncated"]).sum()), "proportion": float((~frame["any_truncated"]).mean())},
            "multiple_fields_truncated": int((frame[[f"{field}_truncated" for field in fields]].sum(axis=1) > 1).sum()),
        }
    summary = {"records": len(audit), "overall": group_stats(audit),
               "by_gold_label": {"include": group_stats(audit.loc[audit.gold_label.eq(1)]), "exclude": group_stats(audit.loc[audit.gold_label.eq(0)])},
               "by_review": {str(key): group_stats(group) for key, group in audit.groupby("review_key", sort=True)},
               "token_distributions": {field: {"original": audit[f"{field}_original_tokens"].describe(percentiles=[.25, .5, .75, .95, .99]).to_dict(), "retained": audit[f"{field}_retained_tokens"].describe(percentiles=[.25, .5, .75, .95, .99]).to_dict()} for field in fields}}
    prompt = {mode: {"count": len(values), "characters_total": int(sum(values)), "characters_max": int(max(values)), "characters_mean": float(np.mean(values))} for mode, values in prompt_chars.items()}
    return audit, summary, {mode: values["characters_total"] for mode, values in prompt.items()} | {f"{mode}_max": values["characters_max"] for mode, values in prompt.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "processed" / "splits")
    parser.add_argument("--criteria-path", type=Path, default=ROOT / "data" / "eligibility_criteria(排纳标准).xlsx")
    parser.add_argument("--model-name-or-path", type=Path, default=ROOT / "model" / "pubmedbert")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "qwen_full_input_validation_comparison")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args(argv)
    if args.output_dir.exists(): raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    validation = data.read_split(args.split_dir / "validation.csv")
    criteria, schema = data.load_eligibility_criteria(args.criteria_path)
    validation = data.join_criteria(validation, criteria, schema, "validation")
    sample, sampling = sample_validation(validation)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    audit, truncation, prompt_chars = audit_truncation(sample, tokenizer, args.max_length)
    args.output_dir.mkdir(parents=True)
    manifest = sample[["_qwen_source_row_index", data.REVIEW_COLUMN, data.LABEL_COLUMN, *data.IDENTIFIER_COLUMNS]].copy()
    manifest.insert(0, "record_id", [record_id_for_row(row.to_dict(), i) for i, (_, row) in enumerate(sample.iterrows())])
    manifest = manifest.rename(columns={data.REVIEW_COLUMN: "review_key", data.LABEL_COLUMN: "gold_label", "_qwen_source_row_index": "source_row_index"})
    manifest.to_csv(args.output_dir / "sample_manifest.csv", index=False)
    audit.to_csv(args.output_dir / "truncation_audit.csv", index=False)
    _json(args.output_dir / "sampling_summary.json", sampling)
    _json(args.output_dir / "truncation_summary.json", truncation)
    config = {"experiment": "qwen_full_input_validation_comparison", "phase": "prepared_not_executed", "sample": {"records": 5000, "include": 300, "exclude": 4700, "seed": SEED}, "modes": {"truncated": "frozen PubMedBERT token reconstruction at 512", "full": "raw title/abstract and complete matched inclusion/exclusion criteria; no truncation"}, "model": QwenConfig(max_tokens=128).public_dict(), "difference_from_legacy_test": {"max_tokens": {"legacy": None, "comparison": 128}}, "prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_fingerprint(), "screening_policy": "retain_unresolved_v1", "input_sha256": {"validation.csv": sha256_file(args.split_dir / "validation.csv"), "criteria.xlsx": sha256_file(args.criteria_path)}, "tokenizer": _tokenizer_identity(str(args.model_name_or_path)), "runtime": {"python": platform.python_version(), "transformers": transformers.__version__}, "source": _git_metadata(), "qwen_context_window": {"status": "provider verified; no matching managed-service tokenizer is published", "context_length_tokens": 1000000, "max_input_tokens": 991808, "max_output_tokens": 131072, "thinking_mode": False, "execution_guard": "Apply a conservative UTF-8-byte-plus-chat-envelope bound and refuse, never truncate, an over-limit input."}, "estimated_requests": 10000, "estimated_token_usage": {"status": "requires Qwen tokenizer/provider accounting; character totals are provided only as a non-billing proxy", "prompt_characters": prompt_chars}, "cost_usd": None}
    _json(args.output_dir / "experiment_config.json", config)
    report = "# Qwen full-input Validation comparison: preflight\n\n"
    report += "Prepared only: no API request or prediction was made.\n\n"
    report += f"- Sample: {len(sample)} unique Validation rows; {int(sample[data.LABEL_COLUMN].eq(1).sum())} include and {int(sample[data.LABEL_COLUMN].eq(0).sum())} exclude.\n"
    report += f"- Modes share the exact frozen prompt fingerprint `{prompt_fingerprint()}` and Qwen API configuration.\n"
    report += "- The provider-documented limit is 991,808 input tokens (1,000,000-token context; 131,072 maximum output). Exact per-prompt counts still require a matching local Qwen tokenizer; execution must refuse any over-limit prompt rather than truncate it.\n"
    report += "- API requests planned after authorization: 10,000 (5,000 records × 2 modes); retries are additional attempts, not new record-level requests.\n"
    (args.output_dir / "preflight_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "records": len(sample), "sampling": sampling["selected_by_label_review"], "truncation": truncation["overall"], "blocking_context_window": True}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
