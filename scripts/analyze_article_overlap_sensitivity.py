"""Post-hoc article-overlap sensitivity analysis using frozen predictions only.

This script never imports model-training code, loads a checkpoint, performs
inference, or changes the frozen threshold.  It validates the canonical final
Test artifacts, uses exact normalized bibliographic identifiers to classify
Test rows, and writes new analysis artifacts to a separate directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


MISSING = {"", "na", "n/a", "nan", "none", "null", "-"}
ID_KINDS = ("doi", "pmid", "openalex_id")
SPLITS = ("train", "validation", "test")
OUTPUT_FILES = (
    "overlap_summary.json",
    "test_overlap_audit.csv",
    "article_disjoint_metrics.json",
    "primary_vs_article_disjoint.csv",
    "false_negative_overlap_audit.csv",
    "identifier_conflicts.csv",
    "README.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _clean(raw: Any) -> str:
    value = "" if raw is None else str(raw).strip()
    return "" if value.lower() in MISSING else value


def normalize_doi(raw: Any) -> tuple[str, bool]:
    value = _clean(raw).lower()
    if not value:
        return "", False
    value = re.sub(r"^(?:doi\s*:\s*)", "", value)
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value)
    value = value.strip()
    valid = bool(re.fullmatch(r"10\.\d{4,9}/\S+", value)) and not any(c.isspace() for c in value)
    return (value if valid else ""), valid


def normalize_pmid(raw: Any) -> tuple[str, bool]:
    value = _clean(raw).lower()
    if not value:
        return "", False
    value = re.sub(r"^pmid\s*:\s*", "", value)
    value = re.sub(r"^https?://(?:www\.)?pubmed\.ncbi\.nlm\.nih\.gov/", "", value)
    value = value.strip().strip("/")
    valid = bool(re.fullmatch(r"\d+", value))
    return (value if valid else ""), valid


def normalize_openalex(raw: Any) -> tuple[str, bool]:
    value = _clean(raw).lower()
    if not value:
        return "", False
    value = re.sub(r"^openalex\s*:\s*", "", value)
    value = re.sub(r"^https?://(?:www\.)?openalex\.org/", "", value)
    value = value.strip().strip("/")
    valid = bool(re.fullmatch(r"w\d+", value))
    return (value if valid else ""), valid


NORMALIZERS = {
    "doi": normalize_doi,
    "pmid": normalize_pmid,
    "openalex_id": normalize_openalex,
}


def normalized_ids(row: dict[str, str]) -> tuple[dict[str, str], list[tuple[str, str]]]:
    result: dict[str, str] = {}
    malformed: list[tuple[str, str]] = []
    for kind in ID_KINDS:
        raw = _clean(row.get(kind, ""))
        normalized, valid = NORMALIZERS[kind](raw)
        result[kind] = normalized
        if raw and not valid:
            malformed.append((kind, raw))
    return result, malformed


def safe_div(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def average_precision(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    tp = fp = 0
    previous_recall = 0.0
    ap = 0.0
    index = 0
    while index < len(ordered):
        score = ordered[index][0]
        group_tp = group_fp = 0
        while index < len(ordered) and ordered[index][0] == score:
            if ordered[index][1] == 1:
                group_tp += 1
            else:
                group_fp += 1
            index += 1
        tp += group_tp
        fp += group_fp
        recall = tp / positives
        precision = tp / (tp + fp)
        ap += (recall - previous_recall) * precision
        previous_recall = recall
    return ap


def roc_auc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def calculate_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    labels = [int(row["true_label"]) for row in rows]
    scores = [float(row["predicted_probability"]) for row in rows]
    predictions = [int(score >= threshold) for score in scores]
    tp = sum(label == 1 and pred == 1 for label, pred in zip(labels, predictions))
    tn = sum(label == 0 and pred == 0 for label, pred in zip(labels, predictions))
    fp = sum(label == 0 and pred == 1 for label, pred in zip(labels, predictions))
    fn = sum(label == 1 and pred == 0 for label, pred in zip(labels, predictions))
    n = len(rows)
    positive = sum(labels)
    negative = n - positive
    return {
        "n": n,
        "positive": positive,
        "negative": negative,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "recall": safe_div(tp, tp + fn),
        "fnr": safe_div(fn, tp + fn),
        "precision": safe_div(tp, tp + fp),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "specificity": safe_div(tn, tn + fp),
        "npv": safe_div(tn, tn + fn),
        "accuracy": safe_div(tp + tn, n),
        "predicted_include_rate": safe_div(tp + fp, n),
        "workload_reduction": safe_div(tn + fn, n),
        "pr_auc": average_precision(labels, scores),
        "roc_auc": roc_auc(labels, scores),
        "threshold": threshold,
    }


def require_close(actual: Any, expected: Any, name: str, tolerance: float = 1e-7) -> None:
    if actual is None or expected is None:
        if actual != expected:
            raise ValueError(f"{name} mismatch: {actual!r} != {expected!r}")
        return
    if not math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"{name} mismatch: {actual!r} != {expected!r}")


class DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0
            return item
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/article_overlap_sensitivity"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis directory: {output_dir}")

    paths = {
        "train": root / "data/processed/splits/train.csv",
        "validation": root / "data/processed/splits/validation.csv",
        "test": root / "data/processed/splits/test.csv",
        "canonical_split": root / "config/canonical_split.json",
        "predictions": root / "outputs/final_test/final_test_predictions.csv",
        "metrics": root / "outputs/final_test/final_test_metrics.json",
        "final_config": root / "outputs/final_test/final_test_config.json",
        "threshold": root / "results/formal_training_strategy_20260910/thresholds/selected_validation_threshold.json",
        "manifest": root / "outputs/training_strategy_comparison/20260910_150708_3f3f0eb0/comparison_manifest.json",
        "checkpoint": root / "outputs/training_strategy_comparison/20260910_150708_3f3f0eb0/lora/adapter_lr_5e-5/lora/seed_42/cls/run_20260910_191547/best_pr_auc_classifier_head.pt",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing canonical inputs: {missing}")
    before_hashes = {name: sha256(path) for name, path in paths.items()}

    canonical = read_json(paths["canonical_split"])
    final_config = read_json(paths["final_config"])
    canonical_metrics = read_json(paths["metrics"])
    threshold_config = read_json(paths["threshold"])
    manifest = read_json(paths["manifest"])
    threshold = float(threshold_config["selected"]["threshold"])
    require_close(threshold, final_config["threshold"], "frozen threshold", 1e-12)
    if threshold_config.get("split") != "validation":
        raise ValueError("Threshold source is not marked as validation")
    if final_config.get("selection_data") != "validation_only":
        raise ValueError("Final config does not attest validation-only selection")
    if not final_config.get("test_was_evaluation_only") or final_config.get("training_performed"):
        raise ValueError("Final Test was not recorded as evaluation-only")
    if manifest.get("test_disabled") is not True:
        raise ValueError("Training-strategy manifest does not attest Test disabled")
    if before_hashes["test"] != final_config["test_split_sha256"]:
        raise ValueError("Current canonical Test CSV hash differs from final Test config")
    if before_hashes["checkpoint"] != final_config["checkpoint_sha256"]:
        raise ValueError("Current final checkpoint hash differs from final Test config")

    split_rows: dict[str, list[dict[str, Any]]] = {}
    malformed_records: list[dict[str, Any]] = []
    required_data_columns = {"doi", "pmid", "openalex_id", "title", "label_included", "review_key"}
    for split in SPLITS:
        fields, rows = read_csv(paths[split])
        if not required_data_columns.issubset(fields):
            raise ValueError(f"{split} lacks required columns: {required_data_columns - set(fields)}")
        enriched: list[dict[str, Any]] = []
        for row_index, row in enumerate(rows, start=1):
            norm, malformed = normalized_ids(row)
            item: dict[str, Any] = {**row, "_normalized": norm, "_row_number": row_index}
            enriched.append(item)
            for kind, raw in malformed:
                malformed_records.append({
                    "conflict_type": "malformed_identifier",
                    "identifier_type": kind,
                    "identifier_value": raw,
                    "associated_values": "",
                    "splits": split,
                    "reviews": row["review_key"],
                    "row_count": 1,
                    "note": "Non-empty identifier excluded from exact matching by conservative validation.",
                })
        split_rows[split] = enriched

    prediction_fields, prediction_rows_raw = read_csv(paths["predictions"])
    required_prediction_columns = {
        "review_name", "title", "pmid", "doi", "openalex_id",
        "true_label", "predicted_probability", "predicted_label",
    }
    if not required_prediction_columns.issubset(prediction_fields):
        raise ValueError("Final Test predictions lack required alignment columns")
    if len(prediction_rows_raw) != len(split_rows["test"]):
        raise ValueError("Test/prediction row-count mismatch")

    aligned_predictions: list[dict[str, Any]] = []
    for index, (test_row, prediction) in enumerate(
        zip(split_rows["test"], prediction_rows_raw), start=1
    ):
        exact_pairs = (
            (test_row["review_key"], prediction["review_name"], "review"),
            (test_row["title"], prediction["title"], "title"),
            (test_row["doi"], prediction["doi"], "doi"),
            (test_row["pmid"], prediction["pmid"], "pmid"),
            (test_row["openalex_id"], prediction["openalex_id"], "openalex_id"),
            (str(test_row["label_included"]), str(prediction["true_label"]), "label"),
        )
        for left, right, name in exact_pairs:
            if left != right:
                raise ValueError(f"Ambiguous row alignment at row {index}: {name} differs")
        probability = float(prediction["predicted_probability"])
        expected_prediction = int(probability >= threshold)
        if int(prediction["predicted_label"]) != expected_prediction:
            raise ValueError(f"Stored predicted label violates frozen threshold at row {index}")
        aligned_predictions.append({
            **prediction,
            "true_label": int(prediction["true_label"]),
            "predicted_label": int(prediction["predicted_label"]),
            "predicted_probability": probability,
            "_test_row": test_row,
        })

    recomputed_primary = calculate_metrics(aligned_predictions, threshold)
    canonical_primary = canonical_metrics["at_locked_threshold"]
    for key in (
        "tp", "tn", "fp", "fn", "recall", "fnr", "precision", "f1",
        "specificity", "npv", "accuracy", "predicted_include_rate",
        "workload_reduction", "pr_auc", "roc_auc",
    ):
        require_close(recomputed_primary[key], canonical_primary[key], f"primary metric {key}")
    if recomputed_primary["n"] != int(final_config["test_row_count"]):
        raise ValueError("Recomputed primary Test N differs from final config")
    qc = final_config["test_data_qc"]
    if recomputed_primary["positive"] != int(qc["positive"]) or recomputed_primary["negative"] != int(qc["negative"]):
        raise ValueError("Recomputed Test class counts differ from final config")

    id_sets: dict[str, dict[str, set[str]]] = {
        split: {kind: set() for kind in ID_KINDS} for split in SPLITS
    }
    label_maps: dict[str, dict[str, dict[str, set[int]]]] = {
        split: {kind: defaultdict(set) for kind in ID_KINDS} for split in SPLITS
    }
    dsu = DisjointSet()
    row_component_nodes: dict[str, list[list[str]]] = {split: [] for split in SPLITS}
    for split in SPLITS:
        for row in split_rows[split]:
            nodes = []
            label = int(row["label_included"])
            for kind, value in row["_normalized"].items():
                if value:
                    id_sets[split][kind].add(value)
                    label_maps[split][kind][value].add(label)
                    nodes.append(f"{kind}:{value}")
            if nodes:
                dsu.find(nodes[0])
                for node in nodes[1:]:
                    dsu.union(nodes[0], node)
            row_component_nodes[split].append(nodes)

    component_sets = {
        split: {dsu.find(nodes[0]) for nodes in row_component_nodes[split] if nodes}
        for split in SPLITS
    }
    pair_summaries: dict[str, Any] = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        key = f"{left}_vs_{right}"
        pair: dict[str, Any] = {"by_identifier": {}}
        for kind in ID_KINDS:
            shared = id_sets[left][kind] & id_sets[right][kind]
            pair["by_identifier"][kind] = {
                "unique_normalized_identifiers": len(shared),
                f"affected_{left}_rows": sum(
                    row["_normalized"][kind] in shared for row in split_rows[left]
                ),
                f"affected_{right}_rows": sum(
                    row["_normalized"][kind] in shared for row in split_rows[right]
                ),
            }
        pair["any_exact_identifier"] = {
            f"affected_{left}_rows": sum(
                any(row["_normalized"][kind] in id_sets[right][kind]
                    for kind in ID_KINDS if row["_normalized"][kind])
                for row in split_rows[left]
            ),
            f"affected_{right}_rows": sum(
                any(row["_normalized"][kind] in id_sets[left][kind]
                    for kind in ID_KINDS if row["_normalized"][kind])
                for row in split_rows[right]
            ),
            "identifier_linked_article_components": len(component_sets[left] & component_sets[right]),
        }
        pair_summaries[key] = pair

    audit_rows: list[dict[str, Any]] = []
    classified_predictions: dict[str, list[dict[str, Any]]] = {
        "confirmed_overlap": [], "confirmed_non_overlap": [], "identity_unknown": []
    }
    label_conflict_summary = {
        "train_test": defaultdict(int),
        "validation_test": defaultdict(int),
    }
    for index, prediction in enumerate(aligned_predictions, start=1):
        test_row = prediction["_test_row"]
        norm = test_row["_normalized"]
        matches: dict[tuple[str, str], bool] = {}
        sources: list[str] = []
        for kind in ID_KINDS:
            for source_split in ("train", "validation"):
                hit = bool(norm[kind] and norm[kind] in id_sets[source_split][kind])
                matches[(kind, source_split)] = hit
                if hit:
                    sources.append(f"{kind}:{source_split}")
        overlap_train = any(matches[(kind, "train")] for kind in ID_KINDS)
        overlap_validation = any(matches[(kind, "validation")] for kind in ID_KINDS)
        has_identifier = any(norm.values())
        if overlap_train or overlap_validation:
            identity_status = "confirmed_overlap"
        elif has_identifier:
            identity_status = "confirmed_non_overlap"
        else:
            identity_status = "identity_unknown"

        test_label = int(prediction["true_label"])
        comparison: dict[str, dict[str, bool]] = {}
        for source_split in ("train", "validation"):
            counterpart_labels: set[int] = set()
            for kind in ID_KINDS:
                if matches[(kind, source_split)]:
                    counterpart_labels.update(label_maps[source_split][kind][norm[kind]])
            same = test_label in counterpart_labels
            conflict = any(label != test_label for label in counterpart_labels)
            comparison[source_split] = {"same": same, "conflict": conflict}
            if counterpart_labels:
                bucket = label_conflict_summary[f"{source_split}_test"]
                bucket["overlap_rows"] += 1
                bucket["same_label_rows"] += int(same)
                bucket["conflicting_label_rows"] += int(conflict)
                bucket["same_only_rows"] += int(same and not conflict)
                bucket["conflicting_only_rows"] += int(conflict and not same)
                bucket["both_same_and_conflicting_rows"] += int(same and conflict)

        row_audit = {
            "test_row_id": f"test_row_{index:06d}",
            "source_original_row_index": test_row.get("_original_row_index", ""),
            "review": prediction["review_name"],
            "title": prediction["title"],
            "label": test_label,
            "probability": prediction["predicted_probability"],
            "predicted_label": prediction["predicted_label"],
            "normalized_doi": norm["doi"],
            "normalized_pmid": norm["pmid"],
            "normalized_openalex_id": norm["openalex_id"],
            "doi_overlap_train": matches[("doi", "train")],
            "doi_overlap_validation": matches[("doi", "validation")],
            "pmid_overlap_train": matches[("pmid", "train")],
            "pmid_overlap_validation": matches[("pmid", "validation")],
            "openalex_overlap_train": matches[("openalex_id", "train")],
            "openalex_overlap_validation": matches[("openalex_id", "validation")],
            "overlap_train_any": overlap_train,
            "overlap_validation_any": overlap_validation,
            "overlap_any": overlap_train or overlap_validation,
            "identity_status": identity_status,
            "same_label_train": comparison["train"]["same"],
            "conflicting_label_train": comparison["train"]["conflict"],
            "same_label_validation": comparison["validation"]["same"],
            "conflicting_label_validation": comparison["validation"]["conflict"],
            "label_conflict_detected": comparison["train"]["conflict"] or comparison["validation"]["conflict"],
            "overlap_source": ";".join(sources),
        }
        audit_rows.append(row_audit)
        classified_predictions[identity_status].append(prediction)

    if sum(len(rows) for rows in classified_predictions.values()) != len(aligned_predictions):
        raise ValueError("Identity classes do not partition Test")
    strict_rows = classified_predictions["confirmed_non_overlap"]
    for audit in audit_rows:
        if audit["identity_status"] == "confirmed_non_overlap" and audit["overlap_any"]:
            raise ValueError("Strict article-disjoint row has an exact identifier overlap")

    strict_metrics = calculate_metrics(strict_rows, threshold)
    descriptive_metrics = {
        group: calculate_metrics(rows, threshold)
        for group, rows in classified_predictions.items()
    }

    identity_summary: dict[str, Any] = {}
    for status, rows in classified_predictions.items():
        positive = sum(int(row["true_label"]) for row in rows)
        identity_summary[status] = {
            "rows": len(rows),
            "percentage": len(rows) / len(aligned_predictions),
            "positive": positive,
            "negative": len(rows) - positive,
        }
    overlap_source_counts = {
        "train_only": sum(r["overlap_train_any"] and not r["overlap_validation_any"] for r in audit_rows),
        "validation_only": sum(r["overlap_validation_any"] and not r["overlap_train_any"] for r in audit_rows),
        "both_train_and_validation": sum(r["overlap_train_any"] and r["overlap_validation_any"] for r in audit_rows),
    }

    # Bibliographic one-to-many mappings are audit flags, not automatic repairs.
    association_specs = (
        ("doi", "pmid", "same_doi_multiple_pmids"),
        ("pmid", "doi", "same_pmid_multiple_dois"),
        ("openalex_id", "doi", "same_openalex_multiple_dois"),
    )
    association_maps: dict[tuple[str, str], dict[str, dict[str, set[str]]]] = {}
    for primary_kind, secondary_kind, _ in association_specs:
        mapping: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: {"values": set(), "splits": set(), "reviews": set(), "rows": set()}
        )
        for split in SPLITS:
            for row in split_rows[split]:
                primary = row["_normalized"][primary_kind]
                secondary = row["_normalized"][secondary_kind]
                if primary and secondary:
                    mapping[primary]["values"].add(secondary)
                    mapping[primary]["splits"].add(split)
                    mapping[primary]["reviews"].add(row["review_key"])
                    mapping[primary]["rows"].add(f"{split}:{row['_row_number']}")
        association_maps[(primary_kind, secondary_kind)] = mapping
        for primary, details in mapping.items():
            if len(details["values"]) > 1:
                malformed_records.append({
                    "conflict_type": _,
                    "identifier_type": primary_kind,
                    "identifier_value": primary,
                    "associated_values": ";".join(sorted(details["values"])),
                    "splits": ";".join(sorted(details["splits"])),
                    "reviews": ";".join(sorted(details["reviews"])),
                    "row_count": len(details["rows"]),
                    "note": "Audited only; no bibliographic mapping was repaired.",
                })

    false_negatives = [row for row in audit_rows if row["label"] == 1 and row["predicted_label"] == 0]
    if len(false_negatives) != canonical_primary["fn"]:
        raise ValueError("False-negative audit count differs from canonical metrics")

    primary_for_comparison = {
        **recomputed_primary,
        "n": recomputed_primary["n"],
        "positive": recomputed_primary["positive"],
        "negative": recomputed_primary["negative"],
    }
    comparison_metrics = (
        "n", "positive", "negative", "tp", "tn", "fp", "fn", "recall", "fnr",
        "precision", "f1", "specificity", "npv", "accuracy",
        "predicted_include_rate", "workload_reduction", "pr_auc", "roc_auc",
    )
    comparison_rows = []
    for metric in comparison_metrics:
        primary_value = primary_for_comparison[metric]
        sensitivity_value = strict_metrics[metric]
        difference = (
            None if primary_value is None or sensitivity_value is None
            else sensitivity_value - primary_value
        )
        comparison_rows.append({
            "metric": metric,
            "original_review_disjoint_test": primary_value,
            "strict_identifier_confirmed_article_disjoint": sensitivity_value,
            "difference_sensitivity_minus_primary": difference,
        })

    # Hash all canonical inputs again before writing any new artifact.
    after_hashes = {name: sha256(path) for name, path in paths.items()}
    if before_hashes != after_hashes:
        raise RuntimeError("A canonical input changed during analysis; refusing to write outputs")

    summary = {
        "analysis_type": "post_hoc_article_disjoint_sensitivity_using_frozen_predictions",
        "primary_evaluation_name": "review-disjoint evaluation",
        "sensitivity_subset_name": "identifier-confirmed article-disjoint subset",
        "identity_rule": "A Test row overlaps when any valid normalized exact DOI, PMID, or OpenAlex ID occurs in Train or Validation.",
        "normalization": {
            "doi": "lowercase; trim; remove doi:, doi.org, or dx.doi.org prefix; require conservative DOI syntax",
            "pmid": "lowercase; trim; remove PMID or PubMed URL prefix; require digits",
            "openalex_id": "lowercase; trim; remove OpenAlex prefix/URL; require W followed by digits",
            "title_or_semantic_matching": False,
        },
        "canonical_paths": {name: str(path.relative_to(root)).replace("\\", "/") for name, path in paths.items()},
        "canonical_sha256_before_and_after": {
            name: {"before": before_hashes[name], "after": after_hashes[name], "unchanged": True}
            for name in paths
        },
        "provenance_checks": {
            "test_prediction_rows_exactly_aligned_to_test_csv": True,
            "test_csv_hash_matches_final_config": True,
            "checkpoint_hash_matches_final_config": True,
            "threshold_matches_validation_selection_and_final_config": True,
            "stored_predicted_labels_match_frozen_threshold": True,
            "recomputed_primary_metrics_match_canonical_metrics": True,
            "model_inference_performed": False,
        },
        "cross_split_overlap": pair_summaries,
        "test_identity_classification": identity_summary,
        "confirmed_overlap_sources": overlap_source_counts,
        "label_conflict_analysis": {
            key: dict(value) for key, value in label_conflict_summary.items()
        },
        "malformed_identifier_rows": sum(r["conflict_type"] == "malformed_identifier" for r in malformed_records),
        "identifier_conflict_records": len(malformed_records),
        "false_negative_count": len(false_negatives),
        "sanity_checks": {
            "identity_groups_sum_to_test_n": True,
            "strict_subset_has_no_exact_normalized_identifier_overlap": True,
            "undefined_metrics_are_serialized_as_null": True,
            "canonical_inputs_unchanged_during_analysis": True,
        },
    }
    metrics_payload = {
        "frozen_threshold": threshold,
        "strict_subset_definition": "identity_status == confirmed_non_overlap",
        "primary_recomputed_from_frozen_predictions": recomputed_primary,
        "strict_identifier_confirmed_article_disjoint": strict_metrics,
        "secondary_descriptive_groups": descriptive_metrics,
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "overlap_summary.json", summary)
    audit_fields = list(audit_rows[0])
    write_csv(output_dir / "test_overlap_audit.csv", audit_fields, audit_rows)
    write_json(output_dir / "article_disjoint_metrics.json", metrics_payload)
    write_csv(
        output_dir / "primary_vs_article_disjoint.csv",
        ["metric", "original_review_disjoint_test", "strict_identifier_confirmed_article_disjoint", "difference_sensitivity_minus_primary"],
        comparison_rows,
    )
    write_csv(output_dir / "false_negative_overlap_audit.csv", audit_fields, false_negatives)
    conflict_fields = [
        "conflict_type", "identifier_type", "identifier_value", "associated_values",
        "splits", "reviews", "row_count", "note",
    ]
    write_csv(output_dir / "identifier_conflicts.csv", conflict_fields, malformed_records)

    readme = f"""# Post-hoc Article-Overlap Sensitivity Analysis

This directory is separate from the canonical Final Test artifacts. The analysis used
stored frozen predictions only. It performed no training, model loading, inference,
checkpoint selection, threshold calibration, or canonical-data modification.

## Definitions

- Primary result: review-disjoint evaluation.
- Strict sensitivity subset: identifier-confirmed article-disjoint Test rows.
- Overlap: at least one valid normalized exact DOI, PMID, or OpenAlex ID occurs in
  Train or Validation.
- `identity_unknown` rows have no usable identifier and are not included in the strict subset.
- Titles, abstracts, fuzzy matching, embeddings, and semantic matching were not used.

## Frozen threshold

`{threshold}` from Validation target-recall calibration; unchanged.

## Test identity groups

| Group | N | Positive | Negative | Percent |
|---|---:|---:|---:|---:|
"""
    for status in ("confirmed_overlap", "confirmed_non_overlap", "identity_unknown"):
        item = identity_summary[status]
        readme += f"| {status} | {item['rows']} | {item['positive']} | {item['negative']} | {item['percentage']:.6%} |\n"
    readme += "\n## Primary vs strict sensitivity\n\n"
    readme += "| Metric | Review-disjoint Test | Strict article-disjoint | Difference |\n|---|---:|---:|---:|\n"
    for metric in ("n", "positive", "negative", "recall", "fnr", "precision", "f1", "pr_auc", "roc_auc", "workload_reduction"):
        p = primary_for_comparison[metric]
        s = strict_metrics[metric]
        d = None if p is None or s is None else s - p
        fmt = lambda value: "NA" if value is None else (str(value) if isinstance(value, int) else f"{value:.9f}")
        readme += f"| {metric} | {fmt(p)} | {fmt(s)} | {fmt(d)} |\n"
    readme += """

## Interpretation constraints

This exact-identifier audit cannot detect duplicates when identifiers are absent or
incorrect. The sensitivity result is post-hoc and does not replace the primary
review-disjoint evaluation. The same article may legitimately have different labels
under different reviews because eligibility criteria are review-specific.

## Files

- `overlap_summary.json`: provenance, overlap, identity, and label-conflict summaries.
- `test_overlap_audit.csv`: one row per canonical Test prediction.
- `article_disjoint_metrics.json`: primary, strict, and descriptive group metrics.
- `primary_vs_article_disjoint.csv`: side-by-side metrics and differences.
- `false_negative_overlap_audit.csv`: bibliographic status of every canonical FN.
- `identifier_conflicts.csv`: malformed IDs and one-to-many identifier mappings; no repairs.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    produced = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    if produced != sorted(OUTPUT_FILES):
        raise RuntimeError(f"Unexpected output manifest: {produced}")
    print(json.dumps({
        "status": "completed",
        "output_dir": str(output_dir),
        "identity_summary": identity_summary,
        "strict_metrics": strict_metrics,
        "false_negatives": len(false_negatives),
        "files": produced,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
