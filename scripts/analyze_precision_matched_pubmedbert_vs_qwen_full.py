"""Post-hoc precision-matched analysis on the frozen 5,000-record Test subset.

This script never loads a model or makes an API call.  It scans only effective
PubMedBERT thresholds induced by the stored probabilities, and labels the
result as Test-label-selected exploratory analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "outputs" / "pubmedbert_vs_qwen_full_test_subset"
FROZEN_THRESHOLD = 0.016082723


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_divide(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def metric_row(tp: int, fp: int, tn: int, fn: int) -> dict[str, int | float | None]:
    retained = tp + fp
    total = tp + fp + tn + fn
    precision = safe_divide(tp, retained)
    recall = safe_divide(tp, tp + fn)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "retained_for_manual_screening": retained,
        "precision": precision,
        "recall": recall,
        "fnr": safe_divide(fn, tp + fn),
        "f1": None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall),
        "workload_reduction": safe_divide(tn + fn, total),
    }


def metrics_from_labels(labels: np.ndarray, predicted_positive: np.ndarray) -> dict[str, int | float | None]:
    tp = int(np.sum((labels == 1) & predicted_positive))
    fp = int(np.sum((labels == 0) & predicted_positive))
    tn = int(np.sum((labels == 0) & ~predicted_positive))
    fn = int(np.sum((labels == 1) & ~predicted_positive))
    return metric_row(tp, fp, tn, fn)


def scan_effective_thresholds(labels: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """One row per distinct score, representing the rule score >= threshold."""
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores, sorted_labels = scores[order], labels[order]
    total_positive = int(labels.sum())
    total_negative = int(len(labels) - total_positive)
    rows: list[dict[str, Any]] = []
    tp = fp = index = 0
    while index < len(sorted_scores):
        threshold = float(sorted_scores[index])
        end = index
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[index]:
            if sorted_labels[end] == 1:
                tp += 1
            else:
                fp += 1
            end += 1
        row: dict[str, Any] = {
            "threshold": threshold,
            "threshold_rule": "predicted_probability >= threshold",
            "threshold_type": "observed_score",
            "score_tie_count": end - index,
            **metric_row(tp, fp, total_negative - fp, total_positive - tp),
        }
        rows.append(row)
        index = end
    # This additional effective decision rule retains no records.  Its precision is
    # undefined, but retaining it makes the threshold scan explicit and complete.
    rows.append({
        "threshold": float(np.nextafter(float(sorted_scores[0]), np.inf)),
        "threshold_rule": "predicted_probability >= threshold",
        "threshold_type": "above_max_score_no_records_retained",
        "score_tie_count": 0,
        **metric_row(0, 0, total_negative, total_positive),
    })
    return pd.DataFrame(rows)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_qwen(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid Qwen JSON at line {line_number}") from exc
    frame = pd.DataFrame(rows)
    required = {"record_id", "gold_label", "review_name", "status", "decision"}
    require(required.issubset(frame.columns), "Qwen predictions lack required fields")
    require(frame.status.eq("completed").all(), "Qwen predictions are not all completed")
    require(frame.decision.isin(["include", "exclude", "uncertain"]).all(), "Qwen decision is invalid")
    require(not frame.record_id.duplicated().any(), "Qwen record_id is duplicated")
    return frame


def write_pr_curve_svg(path: Path, thresholds: pd.DataFrame, original: dict[str, Any], matched: dict[str, Any], qwen: dict[str, Any]) -> None:
    """Write a dependency-free, publication-readable PR curve as SVG."""
    width, height, left, top, right, bottom = 820, 560, 82, 48, 34, 78
    plot_width, plot_height = width - left - right, height - top - bottom
    x = lambda value: left + float(value) * plot_width
    y = lambda value: top + (1 - float(value)) * plot_height
    points = [(0.0, 1.0)] + [(float(row.recall), float(row.precision)) for row in thresholds.itertuples() if pd.notna(row.precision) and row.threshold_type == "observed_score"]
    polyline = " ".join(f"{x(recall):.2f},{y(precision):.2f}" for recall, precision in points)
    def circle(metrics: dict[str, Any], color: str, label: str, marker: str = "circle") -> str:
        px, py = x(metrics["recall"]), y(metrics["precision"])
        if marker == "square":
            shape = f'<rect x="{px-4:.2f}" y="{py-4:.2f}" width="8" height="8" fill="{color}"/>'
        elif marker == "triangle":
            shape = f'<path d="M {px:.2f} {py-5:.2f} L {px-5:.2f} {py+4:.2f} L {px+5:.2f} {py+4:.2f} Z" fill="{color}"/>'
        else:
            shape = f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4.5" fill="{color}"/>'
        return shape + f'<text x="{px+7:.2f}" y="{py-7:.2f}" font-size="11" fill="{color}">{label}</text>'
    ticks = "".join(f'<line x1="{x(v):.2f}" y1="{top}" x2="{x(v):.2f}" y2="{top+plot_height}" stroke="#e5e7eb"/><line x1="{left}" y1="{y(v):.2f}" x2="{left+plot_width}" y2="{y(v):.2f}" stroke="#e5e7eb"/><text x="{x(v):.2f}" y="{top+plot_height+22}" text-anchor="middle" font-size="11">{v:.1f}</text><text x="{left-12}" y="{y(v)+4:.2f}" text-anchor="end" font-size="11">{v:.1f}</text>' for v in np.linspace(0, 1, 6))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2}" y="25" text-anchor="middle" font-family="Arial, sans-serif" font-size="17" font-weight="bold">Precision–Recall Curve: PubMedBERT vs Qwen Full</text>
<g font-family="Arial, sans-serif" fill="#111827">{ticks}</g>
<line x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top+plot_height}" stroke="#111827"/><line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#111827"/>
<polyline points="{polyline}" fill="none" stroke="#1f77b4" stroke-width="1.7"/>
<g font-family="Arial, sans-serif">{circle(original, '#2ca02c', 'Frozen threshold')} {circle(matched, '#d62728', 'Precision-matched', 'square')} {circle(qwen, '#9467bd', 'Qwen Full', 'triangle')}</g>
<text x="{left+plot_width/2}" y="{height-22}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13">Recall</text>
<text x="20" y="{top+plot_height/2}" transform="rotate(-90 20 {top+plot_height/2})" text-anchor="middle" font-family="Arial, sans-serif" font-size="13">Precision</text>
<g font-family="Arial, sans-serif" font-size="11"><line x1="{left+10}" y1="{top+15}" x2="{left+32}" y2="{top+15}" stroke="#1f77b4" stroke-width="2"/><text x="{left+38}" y="{top+19}">PubMedBERT PR curve</text></g>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT / "precision_matched_comparison")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis directory: {output_dir}")

    manifest_path = EXPERIMENT / "sample_manifest.csv"
    pub_path = EXPERIMENT / "pubmedbert_predictions_subset.csv"
    qwen_path = EXPERIMENT / "predictions_qwen_full" / "predictions.jsonl"
    manifest = pd.read_csv(manifest_path, dtype={"record_id": "string"})
    pub = pd.read_csv(pub_path, dtype={"record_id": "string"})
    qwen = load_qwen(qwen_path)
    for name, frame in (("manifest", manifest), ("pubmedbert", pub), ("qwen", qwen)):
        require(len(frame) == 5000, f"{name} does not contain 5,000 records")
        require(not frame.record_id.duplicated().any(), f"{name} record_id is duplicated")
    require(set(manifest.record_id) == set(pub.record_id) == set(qwen.record_id), "Record IDs are not perfectly aligned")

    pair = manifest[["record_id", "review_key", "gold_label"]].merge(
        pub[["record_id", "review_key", "gold_label", "predicted_probability", "predicted_label"]],
        on="record_id", suffixes=("_manifest", "_pub"), validate="one_to_one",
    ).merge(
        qwen[["record_id", "review_name", "gold_label", "decision"]], on="record_id", validate="one_to_one"
    )
    require((pair.review_key_manifest == pair.review_key_pub).all(), "PubMedBERT review keys disagree with manifest")
    require((pair.review_key_manifest == pair.review_name).all(), "Qwen review keys disagree with manifest")
    require((pair.gold_label_manifest.astype(int) == pair.gold_label_pub.astype(int)).all(), "PubMedBERT labels disagree with manifest")
    require((pair.gold_label_manifest.astype(int) == pair.gold_label.astype(int)).all(), "Qwen labels disagree with manifest")

    labels = pair.gold_label_manifest.to_numpy(dtype=int)
    scores = pair.predicted_probability.to_numpy(dtype=float)
    require(np.isfinite(scores).all(), "PubMedBERT probability contains non-finite values")
    require(((scores >= FROZEN_THRESHOLD).astype(int) == pair.predicted_label.astype(int).to_numpy()).all(), "Stored PubMedBERT labels violate frozen threshold")
    qwen_positive = pair.decision.ne("exclude").to_numpy()
    qwen_metrics = metrics_from_labels(labels, qwen_positive)
    original_metrics = metrics_from_labels(labels, scores >= FROZEN_THRESHOLD)

    analysis = scan_effective_thresholds(labels, scores)
    target_precision = qwen_metrics["precision"]
    require(target_precision is not None, "Qwen precision is undefined; matching is impossible")
    eligible = analysis[analysis.precision.notna()].copy()
    eligible["absolute_precision_difference"] = (eligible.precision - target_precision).abs()
    # The documented first two criteria are target precision, then recall.  The
    # final descending-threshold tie-break is deterministic and maximizes exclusion.
    selected = eligible.sort_values(
        ["absolute_precision_difference", "recall", "threshold"], ascending=[True, False, False], kind="mergesort"
    ).iloc[0].to_dict()
    matched_threshold = float(selected["threshold"])
    matched_metrics = metrics_from_labels(labels, scores >= matched_threshold)
    require(all(matched_metrics[key] == selected[key] for key in ("tp", "fp", "tn", "fn")), "Threshold scan inconsistency")

    analysis["is_precision_matched_selection"] = analysis.threshold.eq(matched_threshold)
    analysis["is_original_frozen_threshold"] = analysis.threshold.eq(FROZEN_THRESHOLD)
    output_dir.mkdir(parents=True)
    analysis.to_csv(output_dir / "threshold_analysis.csv", index=False)

    comparison = pd.DataFrame([
        {"system": "Qwen Full", "decision_rule": "exclude=0; include/uncertain=1", "threshold": np.nan, "precision_target": True, **qwen_metrics},
        {"system": "PubMedBERT + LoRA (original frozen threshold)", "decision_rule": "predicted_probability >= threshold", "threshold": FROZEN_THRESHOLD, "precision_target": False, **original_metrics},
        {"system": "PubMedBERT + LoRA (precision-matched; post-hoc)", "decision_rule": "predicted_probability >= threshold", "threshold": matched_threshold, "precision_target": True, **matched_metrics},
    ])
    comparison.to_csv(output_dir / "comparison_table.csv", index=False)

    write_pr_curve_svg(output_dir / "precision_recall_curve.svg", analysis, original_metrics, matched_metrics, qwen_metrics)

    precision_difference = float(matched_metrics["precision"] - target_precision)
    summary = {
        "analysis": "precision_matched_post_hoc_exploratory_test_subset",
        "warning": "The matching threshold was selected using Test labels. It is not an independently validated or pre-specified operating threshold.",
        "records": int(len(pair)), "positive_records": int(labels.sum()), "negative_records": int(len(labels) - labels.sum()),
        "alignment": {"manifest_pubmedbert_qwen_record_ids_identical": True, "gold_labels_identical": True, "frozen_prediction_labels_reproduced": True},
        "inputs": {str(manifest_path.relative_to(ROOT)): sha256_file(manifest_path), str(pub_path.relative_to(ROOT)): sha256_file(pub_path), str(qwen_path.relative_to(ROOT)): sha256_file(qwen_path)},
        "original_frozen_threshold": FROZEN_THRESHOLD,
        "qwen_full": qwen_metrics,
        "pubmedbert_original": original_metrics,
        "pubmedbert_precision_matched": {"threshold": matched_threshold, "target_precision_qwen": target_precision, "precision_match_difference": precision_difference, "absolute_precision_match_difference": abs(precision_difference), **matched_metrics},
        "candidate_thresholds": int(len(analysis)),
        "selection_rule": "minimize absolute difference from Qwen Full precision; among ties maximize recall; remaining exact ties choose largest threshold",
    }
    (output_dir / "precision_matched_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report = f"""# Precision-Matched Comparison: PubMedBERT + LoRA vs Qwen Full

## Purpose

This post-hoc exploratory analysis compares the frozen PubMedBERT + CLS + LoRA predictions with Qwen Full on the same fixed 5,000-record Test subset. PubMedBERT thresholds induced by its stored probabilities were scanned to find the operating point whose precision is closest to Qwen Full's observed precision.

## Data and alignment

The fixed subset contains {len(pair):,} records: {int(labels.sum()):,} positive and {int(len(labels) - labels.sum()):,} negative. The manifest, PubMedBERT predictions, and Qwen predictions have identical unique record IDs and identical gold labels. Stored PubMedBERT labels are reproduced by the original frozen threshold `{FROZEN_THRESHOLD}`.

## Threshold selection

Qwen Full precision is {target_precision:.6%}. The closest effective PubMedBERT threshold is `{matched_threshold:.12g}` (rule: probability ≥ threshold), yielding precision {matched_metrics['precision']:.6%}. The signed matching error is {precision_difference:+.6%} ({abs(precision_difference):.6%} in absolute value). Selection minimized absolute precision difference across observed-score thresholds; ties were broken by higher recall, then by a larger threshold.

## Comparison

| System | Precision | Recall | FN | F1 | Workload reduction | Manual screening |
|---|---:|---:|---:|---:|---:|---:|
| Qwen Full | {qwen_metrics['precision']:.2%} | {qwen_metrics['recall']:.2%} | {qwen_metrics['fn']} | {qwen_metrics['f1']:.2%} | {qwen_metrics['workload_reduction']:.2%} | {qwen_metrics['retained_for_manual_screening']:,} |
| PubMedBERT original frozen threshold | {original_metrics['precision']:.2%} | {original_metrics['recall']:.2%} | {original_metrics['fn']} | {original_metrics['f1']:.2%} | {original_metrics['workload_reduction']:.2%} | {original_metrics['retained_for_manual_screening']:,} |
| PubMedBERT precision-matched threshold | {matched_metrics['precision']:.2%} | {matched_metrics['recall']:.2%} | {matched_metrics['fn']} | {matched_metrics['f1']:.2%} | {matched_metrics['workload_reduction']:.2%} | {matched_metrics['retained_for_manual_screening']:,} |

## Methodological limitation

The precision-matched threshold was selected with gold labels from this Test subset. It therefore constitutes an exploratory, optimistic in-sample operating-point analysis, not a pre-registered or independently validated Test result. It must not replace the original Validation-selected frozen threshold in the primary evaluation. The Qwen output and the stored model probabilities were not changed, and no model/API inference was performed.
"""
    (output_dir / "precision_matched_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "matched_threshold": matched_threshold, "qwen_precision": target_precision, "matched_precision": matched_metrics["precision"], "matched_recall": matched_metrics["recall"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
