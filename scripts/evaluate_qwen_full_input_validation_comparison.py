"""Offline evaluation for the completed paired Qwen Validation A/B experiment."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.llm_baseline.evaluation import apply_screening_policy, evaluate_policy_records

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "qwen_full_input_validation_comparison"
INPUT_PRICE_CNY_PER_M = 0.8
OUTPUT_PRICE_CNY_PER_M = 2.7


def _load(mode: str) -> pd.DataFrame:
    path = OUT / f"predictions_{mode}" / "predictions.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    frame = pd.DataFrame(values)
    required = {"record_id", "review_name", "gold_label", "status", "decision"}
    if missing := required - set(frame):
        raise ValueError(f"{mode} predictions are missing {sorted(missing)}")
    if len(frame) != 5000 or frame.record_id.duplicated().any():
        raise ValueError(f"{mode} predictions are incomplete or duplicate")
    frame = frame.rename(columns={"review_name": "review_key", "gold_label": "label_included"})
    return apply_screening_policy(frame)


def _usage(frame: pd.DataFrame) -> dict[str, float | int]:
    prompt = sum((value or {}).get("prompt_tokens", 0) for value in frame.get("usage", []))
    completion = sum((value or {}).get("completion_tokens", 0) for value in frame.get("usage", []))
    return {"prompt_tokens": int(prompt), "completion_tokens": int(completion),
            "estimated_cost_cny": prompt * INPUT_PRICE_CNY_PER_M / 1_000_000 + completion * OUTPUT_PRICE_CNY_PER_M / 1_000_000}


def main() -> int:
    a, b = _load("truncated"), _load("full")
    if set(a.record_id) != set(b.record_id):
        raise ValueError("A/B record IDs differ")
    audit = pd.read_csv(OUT / "truncation_audit.csv", usecols=["record_id", "abstract_truncated", "any_truncated"])
    paired = a[["record_id", "review_key", "label_included", "decision", "status", "retained"]].merge(
        b[["record_id", "decision", "status", "retained"]], on="record_id", suffixes=("_truncated", "_full"), validate="one_to_one"
    ).merge(audit, on="record_id", validate="one_to_one")
    paired["change"] = paired["retained_truncated"].map({0: "exclude" , 1: "retain"}) + "_to_" + paired["retained_full"].map({0: "exclude", 1: "retain"})
    paired["fn_repaired"] = (paired.label_included.eq(1) & paired.retained_truncated.eq(0) & paired.retained_full.eq(1))
    paired["fn_added"] = (paired.label_included.eq(1) & paired.retained_truncated.eq(1) & paired.retained_full.eq(0))
    subset = {}
    for name, group in {"abstract_truncated": paired[paired.abstract_truncated], "abstract_not_truncated": paired[~paired.abstract_truncated]}.items():
        subset[name] = {"records": len(group), "truncated": evaluate_policy_records(group.rename(columns={"status_truncated": "status", "decision_truncated": "decision"})[["review_key", "label_included", "status", "decision"]]), "full": evaluate_policy_records(group.rename(columns={"status_full": "status", "decision_full": "decision"})[["review_key", "label_included", "status", "decision"]])}
    summary = {"sample_note": "Positive-enriched fixed Validation sample; unweighted precision, F1 and workload reduction describe this sample only, not population prevalence.",
               "truncated": {"metrics": evaluate_policy_records(a[["review_key", "label_included", "status", "decision"]]), "usage": _usage(a)},
               "full": {"metrics": evaluate_policy_records(b[["review_key", "label_included", "status", "decision"]]), "usage": _usage(b)},
               "paired_changes": paired.change.value_counts().to_dict(), "false_negative_changes": {"repaired": int(paired.fn_repaired.sum()), "added": int(paired.fn_added.sum())},
               "by_abstract_truncation": subset}
    paired.to_csv(OUT / "paired_prediction_changes.csv", index=False)
    (OUT / "evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(paired), "changes": summary["paired_changes"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
