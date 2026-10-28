"""Acceptance tests for the Train/Validation-only TF-IDF supplementary baselines."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_tfidf_logreg_supplementary.py"
SPEC = importlib.util.spec_from_file_location("tfidf_supplementary", SCRIPT)
baseline = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(baseline)


def make_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    train = pd.DataFrame({
        "review_key": ["train_review"] * 8, "title": ["included human"] * 4 + ["excluded animal"] * 4,
        "abstract": ["clinical trial evidence"] * 4 + ["mouse study"] * 4,
        "label_included": [1] * 4 + [0] * 4,
    })
    validation = pd.DataFrame({
        "review_key": ["validation_review"] * 4, "title": ["validation unique included", "validation unique excluded"] * 2,
        "abstract": ["validationunique trial", "validationunique mouse"] * 2, "label_included": [1, 0, 1, 0],
    })
    train.to_csv(split_dir / "train.csv", index=False)
    validation.to_csv(split_dir / "validation.csv", index=False)
    criteria = tmp_path / "criteria.xlsx"
    pd.DataFrame({"review_key": ["train_review", "validation_review"],
                  "inclusion_criteria": ["human clinical trial", "human clinical trial"],
                  "exclusion_criteria": ["animal", "animal"]}).to_excel(criteria, index=False)
    canonical = tmp_path / "canonical_split.json"
    canonical.write_text("{}", encoding="utf-8")
    return split_dir, criteria, canonical


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_parse_rejects_test_options():
    with pytest.raises(SystemExit):
        baseline.parse_args(["--evaluate-test"])
    with pytest.raises(SystemExit):
        baseline.parse_args(["--test-split", "test.csv"])


def test_cpu_only_import_surface_has_no_model_stack_dependency():
    source = inspect.getsource(baseline)
    assert "train_classifier_head" not in source
    assert "import torch" not in source
    assert "transformers" not in source
    assert "peft" not in source


def test_train_only_fit_selection_calibration_and_artifacts(tmp_path, monkeypatch):
    split_dir, criteria, canonical = make_inputs(tmp_path)
    calls = []
    real_read = baseline.data.read_split
    def guarded_read(path, **kwargs):
        assert Path(path).name != "test.csv"
        calls.append(Path(path).name)
        return real_read(path, **kwargs)
    monkeypatch.setattr(baseline.data, "read_split", guarded_read)
    fitted_documents = []
    original_fit = baseline.TfidfVectorizer.fit_transform
    def recorded_fit(self, documents, *args, **kwargs):
        fitted_documents.extend(documents)
        return original_fit(self, documents, *args, **kwargs)
    monkeypatch.setattr(baseline.TfidfVectorizer, "fit_transform", recorded_fit)
    output = tmp_path / "supplementary_202610" / "tfidf_logreg"
    args = ["--output-root", str(output), "--split-dir", str(split_dir), "--criteria-path", str(criteria),
            "--canonical-split", str(canonical), "--min-df", "1", "--max-features", "100", "--c-grid", "0.01", "0.1", "1"]
    before = {path: hash_file(path) for path in (split_dir / "train.csv", split_dir / "validation.csv", canonical)}
    protected_before = baseline.protected_hashes(ROOT)
    assert baseline.main(args) == 0
    assert calls == ["train.csv", "validation.csv"]
    assert fitted_documents and all("validationunique" not in text for text in fitted_documents)
    after = {path: hash_file(path) for path in before}
    assert before == after
    assert protected_before == baseline.protected_hashes(ROOT)
    summary = pd.read_csv(output / "validation_summary.csv")
    assert set(summary["baseline"]) == set(baseline.BASELINES)
    for relative in ("config.json", "run_manifest.json", "run.log", "validation_summary.csv"):
        assert (output / relative).is_file()
    for name in baseline.BASELINES:
        run_dir = output / name
        for relative in ("model/tfidf_vectorizer.joblib", "model/logistic_regression.joblib", "validation_predictions.csv",
                         "validation_metrics.json", "threshold_calibration.json", "config.json", "run_manifest.json"):
            assert (run_dir / relative).is_file()
        predictions = pd.read_csv(run_dir / "validation_predictions.csv")
        assert len(predictions) == 4
        metrics = json.loads((run_dir / "validation_metrics.json").read_text())
        recomputed = average_precision_score(predictions["true_label"], predictions["predicted_probability"])
        assert metrics["threshold_independent"]["pr_auc"] == pytest.approx(recomputed)
        calibration = json.loads((run_dir / "threshold_calibration.json").read_text())
        threshold = calibration["selected"]["threshold"]
        assert threshold == pytest.approx(predictions["threshold"].iloc[0])
        assert calibration["selected"]["recall"] >= .95
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["test_access"] == "forbidden"
    for run in manifest["runs"]:
        expected = max(run["c_selection_trials"], key=lambda trial: trial["validation_pr_auc"])["C"]
        assert run["selected_C"] == expected


def test_review_disjoint_validation_and_threshold_rule():
    train = pd.DataFrame({"review_key": ["train", "train"], "label_included": [1, 0]})
    validation = pd.DataFrame({"review_key": ["validation", "validation"], "label_included": [1, 0]})
    baseline.data.validate_splits({"train": train, "validation": validation}, evaluate_test=False)
    labels = np.array([1, 1, 0, 0])
    probabilities = np.array([.8, .6, .7, .1])
    calibration = baseline.calibrate_threshold(labels, probabilities, .5)
    assert calibration["selected"]["threshold"] == pytest.approx(.8)


def test_c_selection_uses_validation_ap_only(tmp_path, monkeypatch):
    args = baseline.parse_args(["--min-df", "1", "--c-grid", "0.01", "0.1", "1"])
    scores = iter([0.2, 0.9, 0.4])
    monkeypatch.setattr(baseline, "average_precision_score", lambda labels, probabilities: next(scores))
    logger = baseline.logging.getLogger("test_c_selection")
    _, _, selected_c, _, trials = baseline.fit_and_select(
        ["positive study", "negative animal", "positive trial", "negative mouse"], np.array([1, 0, 1, 0]),
        ["validation positive", "validation negative"], np.array([1, 0]), args, logger,
    )
    assert selected_c == 0.1
    assert [trial["validation_pr_auc"] for trial in trials] == [0.2, 0.9, 0.4]
