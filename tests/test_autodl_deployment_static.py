"""CPU-only static checks for AutoDL deployment helpers; no datasets are opened."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts import summarize_supplementary_results as summary

ROOT = Path(__file__).resolve().parents[1]


def test_deployment_scripts_are_guarded_and_test_sealed():
    setup = (ROOT / "scripts" / "setup_autodl.sh").read_text(encoding="utf-8")
    validate = (ROOT / "scripts" / "validate_autodl.sh").read_text(encoding="utf-8")
    orchestration = (ROOT / "scripts" / "run_autodl_supplementary.sh").read_text(encoding="utf-8")
    for content in (setup, validate, orchestration):
        assert "set -Eeuo pipefail" in content
        assert "test.csv" not in content
    assert '"status") != "passed"' in orchestration
    assert "Ambiguous or partial artifacts" in orchestration
    assert "--evaluate-test" not in orchestration


def test_validation_only_summary_excludes_smoke_and_refuses_test_metrics(tmp_path):
    formal = tmp_path / "mean_lora" / "batch" / "seed_42"
    smoke = tmp_path / "autodl_validation" / "smoke" / "seed_42"
    formal.mkdir(parents=True); smoke.mkdir(parents=True)
    payload = {"test_evaluated": False, "seed": 42, "pooling": "mean", "training_strategy": "lora",
               "class_weighting": "weighted", "best_validation_pr_auc": .1, "best_validation_roc_auc": .2,
               "best_validation_recall": .3, "best_validation_precision": .4, "best_validation_fnr": .7,
               "best_validation_f1": .5, "best_validation_workload_reduction": .6}
    (formal / "run_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    payload["test_evaluated"] = True
    (smoke / "run_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    csv, report = tmp_path / "summary.csv", tmp_path / "summary.json"
    assert summary.main(["--root", str(tmp_path), "--output-csv", str(csv), "--output-json", str(report)]) == 0
    frame = pd.read_csv(csv)
    assert len(frame) == 1 and frame.loc[0, "seed"] == 42
    assert json.loads(report.read_text(encoding="utf-8"))["test_metrics_included"] is False
