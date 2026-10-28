"""Read-only validation-only summary for completed supplementary experiments.

This utility deliberately rejects Test metrics and only writes files explicitly
named by --output-csv / --output-json. It never invokes a model or loader.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("best_validation_pr_auc", "best_validation_roc_auc", "best_validation_recall",
           "best_validation_precision", "best_validation_fnr", "best_validation_f1",
           "best_validation_workload_reduction")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, default=ROOT / "supplementary_202610")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def read_summary(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("test_evaluated") is not False:
        raise ValueError(f"refusing non-validation-only result: {path}")
    row = {"summary_path": str(path), "seed": payload.get("seed"),
           "pooling": payload.get("pooling"), "training_strategy": payload.get("training_strategy"),
           "class_weighting": payload.get("class_weighting", "weighted")}
    for metric in METRICS:
        row[metric.removeprefix("best_validation_")] = payload.get(metric)
    return row


def main(argv=None) -> int:
    args = parse_args(argv)
    # Only formal supplementary experiment roots are eligible.  Disposable
    # AutoDL smoke artifacts and unrelated historical results are excluded.
    experiment_roots = (args.root / "mean_lora", args.root / "cls_lora_multiseed",
                        args.root / "cls_lora_unweighted_bce")
    paths = sorted(path for root in experiment_roots if root.is_dir()
                   for path in root.glob("**/run_summary.json"))
    rows = [read_summary(path) for path in paths]
    if not rows:
        raise SystemExit("No supplementary run_summary.json files found")
    frame = pd.DataFrame(rows).sort_values(["pooling", "class_weighting", "seed"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.output_csv.exists() or args.output_json.exists():
        raise SystemExit("Refusing to overwrite an existing summary output")
    frame.to_csv(args.output_csv, index=False)
    numeric = [name.removeprefix("best_validation_") for name in METRICS]
    grouped = frame.groupby(["pooling", "training_strategy", "class_weighting"], dropna=False)[numeric].agg(["count", "mean", "std", "min", "max"])
    report = {"source_root": str(args.root), "test_metrics_included": False,
              "rows": len(frame), "aggregate": json.loads(grouped.to_json())}
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(frame), "csv": str(args.output_csv), "json": str(args.output_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
