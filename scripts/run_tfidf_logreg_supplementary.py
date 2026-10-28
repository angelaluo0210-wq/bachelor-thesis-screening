"""Train the supplementary TF-IDF + Logistic Regression baselines.

This is intentionally a Train/Validation-only entry point.  It never opens a
Test file, exposes no Test-related option, and refuses to overwrite a prior
supplementary run.  TF-IDF is fit on Train text only; Validation is used only
for C selection and threshold calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import screening_data as data


BASELINES = {
    "baseline_a_title_abstract": "title_abstract",
    "baseline_b_criteria_title_abstract": "criteria_title_abstract",
}
DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0)
PROTECTED_RELATIVE_PATHS = (
    "config/canonical_split.json",
    "data/processed/splits/train.csv",
    "data/processed/splits/validation.csv",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-root", type=Path, default=ROOT / "supplementary_202610" / "tfidf_logreg")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "processed" / "splits")
    parser.add_argument("--criteria-path", type=Path, default=ROOT / "data" / "eligibility_criteria(\u6392\u7eb3\u6807\u51c6).xlsx")
    parser.add_argument("--canonical-split", type=Path, default=ROOT / "config" / "canonical_split.json")
    parser.add_argument("--ngram-range", nargs=2, type=int, default=(1, 2), metavar=("MIN_N", "MAX_N"))
    parser.add_argument("--max-features", type=int, default=200000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--c-grid", nargs="+", type=float, default=DEFAULT_C_GRID)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--target-recall", type=float, default=0.95)
    args = parser.parse_args(argv)
    if args.ngram_range[0] < 1 or args.ngram_range[0] > args.ngram_range[1]:
        parser.error("ngram-range must be positive and ordered")
    if args.max_features < 1 or args.min_df < 1 or args.max_iter < 1:
        parser.error("max-features, min-df and max-iter must be positive")
    if not args.c_grid or any(not math.isfinite(value) or value <= 0 for value in args.c_grid):
        parser.error("c-grid values must be finite and positive")
    if not 0 < args.target_recall <= 1:
        parser.error("target-recall must be in (0, 1]")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_hashes(root: Path) -> dict[str, str]:
    """Hash only Train/Validation-era immutable inputs, never Test artifacts."""
    paths = {name: root / name for name in PROTECTED_RELATIVE_PATHS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Protected input(s) missing: {missing}")
    return {name: sha256(path) for name, path in paths.items()}


def configure_logging(output_root: Path) -> logging.Logger:
    logger = logging.getLogger("supplementary_tfidf_logreg")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(output_root / "run.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def title_abstract_text(frame: pd.DataFrame) -> list[str]:
    return [
        "\n\n".join((f"[TITLE]\n{data.safe_text(title)}", f"[ABSTRACT]\n{data.safe_text(abstract)}"))
        for title, abstract in zip(frame[data.TITLE_COLUMN], frame[data.ABSTRACT_COLUMN])
    ]


def criteria_title_abstract_text(frame: pd.DataFrame) -> list[str]:
    required = {"_inclusion_text", "_exclusion_text"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Criteria-joined frame is missing columns: {sorted(missing)}")
    return [
        "\n\n".join((
            f"[INCLUSION]\n{data.safe_text(inclusion)}", f"[EXCLUSION]\n{data.safe_text(exclusion)}",
            f"[TITLE]\n{data.safe_text(title)}", f"[ABSTRACT]\n{data.safe_text(abstract)}",
        ))
        for inclusion, exclusion, title, abstract in zip(
            frame["_inclusion_text"], frame["_exclusion_text"],
            frame[data.TITLE_COLUMN], frame[data.ABSTRACT_COLUMN],
        )
    ]


def make_text(frame: pd.DataFrame, input_variant: str) -> list[str]:
    if input_variant == "title_abstract":
        return title_abstract_text(frame)
    if input_variant == "criteria_title_abstract":
        return criteria_title_abstract_text(frame)
    raise ValueError(f"Unknown input variant: {input_variant}")


def calculate_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    # Reuse the project's metric definition, including workload reduction.
    return data.calculate_metrics(labels, probabilities, threshold)


def calibrate_threshold(labels: np.ndarray, probabilities: np.ndarray, target_recall: float) -> dict[str, Any]:
    """Return the largest observed-score threshold with Validation recall >= target."""
    candidates = np.unique(probabilities[np.isfinite(probabilities)])
    candidates.sort()
    candidates = candidates[::-1]
    eligible: list[tuple[float, dict[str, Any]]] = []
    for threshold in candidates:
        metrics = calculate_metrics(labels, probabilities, float(threshold))
        if not np.isnan(metrics["recall"]) and metrics["recall"] >= target_recall:
            eligible.append((float(threshold), metrics))
    if not eligible:
        raise ValueError("No Validation threshold achieves the requested recall target")
    threshold, metrics = eligible[0]  # candidates are descending: first is largest.
    return {
        "split": "validation",
        "target_recall": float(target_recall),
        "selection_rule": "largest_observed_score_threshold_with_validation_recall_at_least_target",
        "candidate_threshold_count": int(len(candidates)),
        "selected": {"threshold": threshold, **metrics},
    }


def prediction_frame(frame: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> pd.DataFrame:
    columns = [data.REVIEW_COLUMN, data.TITLE_COLUMN, data.ABSTRACT_COLUMN]
    columns.extend(column for column in data.IDENTIFIER_COLUMNS if column in frame.columns)
    output = frame.loc[:, columns].copy().reset_index().rename(columns={
        "index": "source_row_index", data.REVIEW_COLUMN: "review_name",
    })
    output["true_label"] = frame[data.LABEL_COLUMN].to_numpy(dtype=np.int64)
    output["predicted_probability"] = probabilities
    output["predicted_label"] = (probabilities >= threshold).astype(np.int64)
    output["threshold"] = threshold
    return output


def fit_and_select(
    train_text: list[str], train_labels: np.ndarray, validation_text: list[str], validation_labels: np.ndarray,
    args: argparse.Namespace, logger: logging.Logger,
) -> tuple[TfidfVectorizer, LogisticRegression, float, np.ndarray, list[dict[str, float]]]:
    vectorizer = TfidfVectorizer(
        ngram_range=tuple(args.ngram_range), max_features=args.max_features, min_df=args.min_df,
        sublinear_tf=True,
    )
    train_matrix = vectorizer.fit_transform(train_text)
    validation_matrix = vectorizer.transform(validation_text)
    trials: list[dict[str, float]] = []
    best: tuple[float, float, LogisticRegression, np.ndarray] | None = None
    for c_value in args.c_grid:
        model = LogisticRegression(
            C=c_value, class_weight="balanced", solver="liblinear", max_iter=args.max_iter,
            random_state=args.random_state,
        )
        model.fit(train_matrix, train_labels)
        probabilities = model.predict_proba(validation_matrix)[:, 1]
        ap = float(average_precision_score(validation_labels, probabilities))
        trial = {"C": float(c_value), "validation_pr_auc": ap}
        trials.append(trial)
        logger.info("C=%s validation_pr_auc=%.12f", c_value, ap)
        if best is None or ap > best[0]:
            best = (ap, float(c_value), model, probabilities)
    assert best is not None
    return vectorizer, best[2], best[1], best[3], trials


def run_baseline(
    baseline_name: str, input_variant: str, train: pd.DataFrame, validation: pd.DataFrame,
    args: argparse.Namespace, output_root: Path, logger: logging.Logger,
) -> dict[str, Any]:
    baseline_dir = output_root / baseline_name
    model_dir = baseline_dir / "model"
    model_dir.mkdir(parents=True)
    train_text, validation_text = make_text(train, input_variant), make_text(validation, input_variant)
    labels = train[data.LABEL_COLUMN].to_numpy(dtype=np.int64)
    validation_labels = validation[data.LABEL_COLUMN].to_numpy(dtype=np.int64)
    vectorizer, model, selected_c, probabilities, trials = fit_and_select(
        train_text, labels, validation_text, validation_labels, args, logger
    )
    calibration = calibrate_threshold(validation_labels, probabilities, args.target_recall)
    threshold = float(calibration["selected"]["threshold"])
    metrics = calculate_metrics(validation_labels, probabilities, threshold)
    predictions = prediction_frame(validation, probabilities, threshold)
    if len(predictions) != len(validation):
        raise RuntimeError("Validation prediction count does not match Validation input")
    joblib.dump(vectorizer, model_dir / "tfidf_vectorizer.joblib")
    joblib.dump(model, model_dir / "logistic_regression.joblib")
    predictions.to_csv(baseline_dir / "validation_predictions.csv", index=False)
    (baseline_dir / "validation_metrics.json").write_text(
        json.dumps({
            "split": "validation", "selection_metric": "validation_pr_auc", "selected_C": selected_c,
            "threshold_independent": {"pr_auc": metrics["pr_auc"], "roc_auc": metrics["roc_auc"]},
            "at_calibrated_validation_threshold": metrics,
        }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (baseline_dir / "threshold_calibration.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    result = {
        "baseline": baseline_name, "input_variant": input_variant, "directory": str(baseline_dir),
        "selected_C": selected_c, "c_selection_trials": trials,
        "validation_predictions": str(baseline_dir / "validation_predictions.csv"),
        "validation_rows": int(len(validation)), "threshold": threshold,
        **metrics,
    }
    (baseline_dir / "config.json").write_text(json.dumps({
        "baseline": baseline_name, "input_variant": input_variant, "test_access": "forbidden",
        "tfidf": {"ngram_range": list(args.ngram_range), "max_features": args.max_features,
                  "min_df": args.min_df, "sublinear_tf": True,
                  "truncation": "none; complete available text, not PubMedBERT's 512-token limit"},
        "logistic_regression": {"class_weight": "balanced", "solver": "liblinear", "max_iter": args.max_iter,
                                "C_grid": list(args.c_grid), "random_state": args.random_state},
        "selection_metric": "validation_pr_auc", "target_recall": args.target_recall,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (baseline_dir / "run_manifest.json").write_text(
        json.dumps({"status": "completed", "test_access": "forbidden", **result}, ensure_ascii=False,
                   indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing supplementary output directory: {args.output_root}")
    before_hashes = protected_hashes(ROOT)
    splits = data.load_splits(args.split_dir, evaluate_test=False)
    data.validate_splits(splits, evaluate_test=False)
    criteria, schema = data.load_eligibility_criteria(args.criteria_path)
    train = data.join_criteria(splits["train"], criteria, schema, "train")
    validation = data.join_criteria(splits["validation"], criteria, schema, "validation")
    args.output_root.mkdir(parents=True)
    logger = configure_logging(args.output_root)
    logger.info("Train/Validation-only TF-IDF baseline started; Test was not opened")
    summaries = {
        "train": data.summarize_split(train, "train"),
        "validation": data.summarize_split(validation, "validation"),
    }
    runs = [run_baseline(name, variant, train, validation, args, args.output_root, logger)
            for name, variant in BASELINES.items()]
    summary_columns = ["baseline", "input_variant", "selected_C", "threshold", "pr_auc", "roc_auc", "recall", "precision", "fnr", "f1", "workload_reduction"]
    pd.DataFrame(runs).loc[:, summary_columns].to_csv(args.output_root / "validation_summary.csv", index=False)
    after_hashes = protected_hashes(ROOT)
    if before_hashes != after_hashes:
        raise RuntimeError("A protected Train/Validation input changed during execution")
    config = {
        "experiment": "supplementary_202610_tfidf_logreg", "test_access": "forbidden", "test_evaluated": False,
        "canonical_split": str(args.canonical_split), "split_dir": str(args.split_dir),
        "criteria_path": str(args.criteria_path), "input_variants": BASELINES,
        "tfidf": {"ngram_range": list(args.ngram_range), "max_features": args.max_features,
                  "min_df": args.min_df, "sublinear_tf": True,
                  "truncation": "none; TF-IDF uses complete available text, unlike PubMedBERT max_length=512"},
        "logistic_regression": {"class_weight": "balanced", "solver": "liblinear", "max_iter": args.max_iter,
                                "C_grid": list(args.c_grid), "random_state": args.random_state},
        "model_selection": "Validation AP only", "threshold_calibration": f"Validation recall >= {args.target_recall}; largest eligible observed score",
        "data_qc": summaries, "protected_input_sha256": before_hashes,
    }
    (args.output_root / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
                "test_access": "forbidden", "protected_input_sha256_before": before_hashes,
                "protected_input_sha256_after": after_hashes, "runs": runs}
    (args.output_root / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    logger.info("Completed two supplementary TF-IDF baselines; Test was not opened")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
