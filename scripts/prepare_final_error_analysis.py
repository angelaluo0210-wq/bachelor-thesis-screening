"""Prepare factual review sheets for the canonical Final Test false negatives.

This script uses stored Test data and frozen predictions only. It performs no
training, model loading, inference, threshold change, label change, external
retrieval, or automatic causal/qualitative error explanation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from openpyxl import load_workbook


OUTPUT_NAMES = (
    "README.md",
    "error_analysis_summary.md",
    "false_negative_case_1.md",
    "false_negative_case_2.md",
    "false_negative_cases.csv",
    "missing_abstract_context.json",
)

MANUAL_QUESTIONS = (
    "Does the title alone provide enough information to establish eligibility?",
    "Which eligibility criteria can be identified from the title?",
    "Which eligibility criteria require information normally found in the abstract/full text?",
    "Is the positive gold label understandable from the available title alone?",
    "Could missing abstract information plausibly make this case difficult for title/abstract screening?",
    "Is the probability far below or close to the frozen threshold?",
    "Is there any indication of a data-quality issue?",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))


def yes_no(value: bool) -> str:
    return "YES" if value else "NO"


def load_criteria(path: Path, review: str) -> dict[str, Any]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    if "Eligibility criteria" not in workbook.sheetnames:
        raise ValueError("Eligibility criteria worksheet is missing")
    sheet = workbook["Eligibility criteria"]
    rows = list(sheet.iter_rows(values_only=True))
    headers = [str(value).strip() if value is not None else "" for value in rows[0]]
    required = {"Review key", "Eligibility criteria", "Inclusion criteria", "Exclusion criteria"}
    if not required.issubset(headers):
        raise ValueError(f"Eligibility workbook lacks columns: {required - set(headers)}")
    matches = []
    for excel_row, values in enumerate(rows[1:], start=2):
        record = dict(zip(headers, values))
        if str(record.get("Review key", "")).strip() == review:
            matches.append((excel_row, record))
    if len(matches) != 1:
        raise ValueError(f"Expected one criteria row for {review}; found {len(matches)}")
    excel_row, record = matches[0]
    return {
        "sheet": "Eligibility criteria",
        "excel_row": excel_row,
        "review_doi": record.get("Review DOI") or "",
        "dataset": record.get("Dataset") or "",
        "eligibility": record.get("Eligibility criteria") or "",
        "inclusion": record.get("Inclusion criteria") or "",
        "exclusion": record.get("Exclusion criteria") or "",
        "needs_manual_review": record.get("Needs manual review") or "",
    }


def render_case(case_number: int, case: dict[str, Any], criteria: dict[str, Any], source_path: str) -> str:
    abstract_display = case["abstract"] if case["abstract"] else "NOT AVAILABLE IN REPOSITORY ARTIFACTS"
    questions = "\n".join(f"{index}. {question}" for index, question in enumerate(MANUAL_QUESTIONS, 1))
    return f"""# FN CASE {case_number}

## Article

Title: {case['title']}

Review: {case['review']}

DOI: {case['doi'] or 'NOT AVAILABLE'}

PMID: {case['pmid'] or 'NOT AVAILABLE'}

OpenAlex ID: {case['openalex_id'] or 'NOT AVAILABLE'}

Stable Test row ID: {case['test_row_id']}

Canonical Test source row: `{case['source_file']}`, original row index `{case['source_original_row_index']}`

## Model output

True label: {case['true_label']}

Predicted label: {case['predicted_label']}

Probability: {case['probability']:.10f}

Frozen threshold: {case['threshold']:.9f}

Probability - threshold: {case['distance_from_threshold']:.10f}

No qualitative distance category was assigned automatically.

## Available screening text

Title:

> {case['title']}

Abstract:

> {abstract_display}

## Data completeness

- Title available: {yes_no(case['title_present'])}
- Abstract available: {yes_no(case['abstract_present'])}
- Abstract character count: {case['abstract_character_count']}
- Abstract word count: {case['abstract_word_count']}
- Model effectively had title-only article text: {yes_no(case['title_only'])}

## Review eligibility criteria

Source: `{source_path}`, sheet `{criteria['sheet']}`, row {criteria['excel_row']}.

The following is repository free text. It has not been converted into inferred PICO fields.

### Inclusion criteria

{criteria['inclusion']}

### Exclusion criteria

{criteria['exclusion']}

### Complete eligibility text

{criteria['eligibility']}

## Existing inclusion rationale

Explicit inclusion reason available: **NO**

No explicit article-level inclusion rationale was found in the repository. The available artifacts establish the positive label but do not state why this specific article was included.

## Overlap status

- Identity status: {case['identity_status']}
- Train overlap: {yes_no(case['overlap_train'])}
- Validation overlap: {yes_no(case['overlap_validation'])}
- Available identifier used for identity audit: {case['identity_identifiers']}
- Same-label counterpart: {yes_no(case['same_label_counterpart'])}
- Conflicting-label counterpart: {yes_no(case['conflicting_label_counterpart'])}

## Review-level context

- Review: {case['review']}
- N: {case['review_n']}
- Positives: {case['review_positive']}
- Negatives: {case['review_negative']}
- TP: {case['review_tp']}
- FN: {case['review_fn']}
- Recall: {case['review_recall']:.9f}
- Precision: {case['review_precision']:.9f}
- PR-AUC: {case['review_pr_auc']:.9f}

## Questions for manual researcher review

{questions}

## MANUAL RESEARCHER NOTES


"""


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "results/final_error_analysis"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis directory: {output_dir}")

    paths = {
        "test": root / "data/processed/splits/test.csv",
        "predictions": root / "outputs/final_test/final_test_predictions.csv",
        "metrics": root / "outputs/final_test/final_test_metrics.json",
        "config": root / "outputs/final_test/final_test_config.json",
        "per_review": root / "outputs/final_test/per_review_test_metrics.csv",
        "threshold": root / "results/formal_training_strategy_20260910/thresholds/selected_validation_threshold.json",
        "criteria": root / "data/eligibility_criteria(排纳标准).xlsx",
        "overlap_fn": root / "results/article_overlap_sensitivity/false_negative_overlap_audit.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required source artifacts are missing: {missing}")
    before_hashes = {name: sha256(path) for name, path in paths.items()}

    test_fields, test_rows = read_csv(paths["test"])
    prediction_fields, predictions = read_csv(paths["predictions"])
    if len(test_rows) != len(predictions):
        raise ValueError("Canonical Test and prediction row counts differ")
    required_test = {"doi", "pmid", "openalex_id", "title", "abstract", "label_included", "review_key"}
    required_predictions = {
        "doi", "pmid", "openalex_id", "title", "true_label", "predicted_label",
        "predicted_probability", "review_name",
    }
    if not required_test.issubset(test_fields) or not required_predictions.issubset(prediction_fields):
        raise ValueError("Canonical Test or prediction schema is incomplete")

    config = read_json(paths["config"])
    canonical_metrics = read_json(paths["metrics"])["at_locked_threshold"]
    threshold_config = read_json(paths["threshold"])
    threshold = float(threshold_config["selected"]["threshold"])
    if not math.isclose(threshold, float(config["threshold"]), rel_tol=0, abs_tol=1e-12):
        raise ValueError("Frozen threshold differs between Validation selection and Final Test config")
    if sha256(paths["test"]) != config["test_split_sha256"]:
        raise ValueError("Canonical Test hash differs from Final Test config")

    aligned = []
    for row_number, (test_row, prediction) in enumerate(zip(test_rows, predictions), start=1):
        checks = (
            (test_row["review_key"], prediction["review_name"], "review"),
            (test_row["title"], prediction["title"], "title"),
            (test_row["doi"], prediction["doi"], "doi"),
            (test_row["pmid"], prediction["pmid"], "pmid"),
            (test_row["openalex_id"], prediction["openalex_id"], "openalex"),
            (test_row["label_included"], prediction["true_label"], "label"),
        )
        for actual, predicted, field in checks:
            if actual != predicted:
                raise ValueError(f"Ambiguous Test/prediction alignment at row {row_number}: {field}")
        probability = float(prediction["predicted_probability"])
        frozen_prediction = int(probability >= threshold)
        if frozen_prediction != int(prediction["predicted_label"]):
            raise ValueError(f"Stored predicted label violates frozen threshold at row {row_number}")
        aligned.append((test_row, prediction, row_number))

    fn_aligned = [
        item for item in aligned
        if int(item[1]["true_label"]) == 1 and int(item[1]["predicted_label"]) == 0
    ]
    if len(fn_aligned) != 2 or int(canonical_metrics["fn"]) != 2:
        raise ValueError(f"Expected exactly two canonical false negatives; found {len(fn_aligned)}")

    _, overlap_rows = read_csv(paths["overlap_fn"])
    if len(overlap_rows) != 2:
        raise ValueError("Overlap FN audit does not contain exactly two records")
    overlap_by_title = {row["title"]: row for row in overlap_rows}
    if len(overlap_by_title) != 2:
        raise ValueError("Overlap FN audit titles are not unique")

    _, per_review_rows = read_csv(paths["per_review"])
    per_review = {row["review_name"]: row for row in per_review_rows}
    reviews = {item[1]["review_name"] for item in fn_aligned}
    if reviews != {"van_Dis_2020"}:
        raise ValueError(f"Unexpected FN reviews: {reviews}")
    criteria = load_criteria(paths["criteria"], "van_Dis_2020")

    cases: list[dict[str, Any]] = []
    for test_row, prediction, _ in fn_aligned:
        overlap = overlap_by_title.get(prediction["title"])
        if overlap is None:
            raise ValueError("FN is missing from article-overlap audit")
        review_metrics = per_review[prediction["review_name"]]
        abstract = test_row["abstract"].strip()
        title = test_row["title"].strip()
        identifier_parts = []
        for label, value in (("DOI", test_row["doi"]), ("PMID", test_row["pmid"]), ("OpenAlex", test_row["openalex_id"])):
            if value.strip():
                identifier_parts.append(f"{label}: {value.strip()}")
        cases.append({
            "test_row_id": overlap["test_row_id"],
            "source_original_row_index": test_row.get("_original_row_index", ""),
            "source_file": test_row.get("_source_file", ""),
            "review": prediction["review_name"],
            "title": title,
            "abstract": abstract,
            "doi": test_row["doi"].strip(),
            "pmid": test_row["pmid"].strip(),
            "openalex_id": test_row["openalex_id"].strip(),
            "true_label": int(prediction["true_label"]),
            "predicted_label": int(prediction["predicted_label"]),
            "probability": float(prediction["predicted_probability"]),
            "threshold": threshold,
            "distance_from_threshold": float(prediction["predicted_probability"]) - threshold,
            "title_present": bool(title),
            "abstract_present": bool(abstract),
            "abstract_character_count": len(abstract),
            "abstract_word_count": word_count(abstract),
            "title_only": bool(title) and not bool(abstract),
            "identity_status": overlap["identity_status"],
            "overlap_train": overlap["overlap_train_any"].lower() == "true",
            "overlap_validation": overlap["overlap_validation_any"].lower() == "true",
            "same_label_counterpart": overlap["same_label_train"].lower() == "true" or overlap["same_label_validation"].lower() == "true",
            "conflicting_label_counterpart": overlap["conflicting_label_train"].lower() == "true" or overlap["conflicting_label_validation"].lower() == "true",
            "identity_identifiers": "; ".join(identifier_parts) or "NONE",
            "review_n": int(review_metrics["n_samples"]),
            "review_positive": int(review_metrics["n_positive"]),
            "review_negative": int(review_metrics["n_negative"]),
            "review_tp": int(review_metrics["tp"]),
            "review_fn": int(review_metrics["fn"]),
            "review_recall": float(review_metrics["recall"]),
            "review_precision": float(review_metrics["precision"]),
            "review_pr_auc": float(review_metrics["pr_auc"]),
            "explicit_inclusion_reason_available": False,
            "eligibility_criteria_source": "data/eligibility_criteria(排纳标准).xlsx; Eligibility criteria!A24:H24",
        })

    # Missing-abstract context uses existing Test labels and stored FN identities only.
    test_positives = [row for row in test_rows if int(row["label_included"]) == 1]
    review_positives = [row for row in test_positives if row["review_key"] == "van_Dis_2020"]
    test_positive_missing = [row for row in test_positives if not row["abstract"].strip()]
    review_positive_missing = [row for row in review_positives if not row["abstract"].strip()]
    fn_missing = sum(not case["abstract_present"] for case in cases)
    context = {
        "analysis_type": "descriptive_only",
        "causal_conclusion_drawn": False,
        "relevant_review": {
            "review": "van_Dis_2020",
            "total_positive_records": len(review_positives),
            "positive_records_with_missing_abstract": len(review_positive_missing),
            "false_negatives_with_missing_abstract": fn_missing,
        },
        "complete_test": {
            "total_positive_records": len(test_positives),
            "positive_records_with_missing_abstract": len(test_positive_missing),
            "false_negatives_with_missing_abstract": fn_missing,
        },
        "contingency": [
            {
                "group": "Abstract available",
                "included_articles": len(test_positives) - len(test_positive_missing),
                "false_negatives": len(cases) - fn_missing,
            },
            {
                "group": "Abstract missing",
                "included_articles": len(test_positive_missing),
                "false_negatives": fn_missing,
            },
        ],
        "interpretation_limit": "Both FN records had missing abstracts. This descriptive fact does not establish causality.",
    }
    if len(review_positives) != int(per_review["van_Dis_2020"]["n_positive"]):
        raise ValueError("Review positive count differs from canonical per-review metrics")
    if len(test_positives) != int(config["test_data_qc"]["positive"]):
        raise ValueError("Test positive count differs from Final Test config")
    if len(test_positive_missing) != int(config["test_data_qc"]["missing_abstract_positive"]):
        raise ValueError("Test positive missing-abstract count differs from Final Test config")

    after_hashes = {name: sha256(path) for name, path in paths.items()}
    if before_hashes != after_hashes:
        raise RuntimeError("A canonical source changed during analysis; refusing to write outputs")

    output_dir.mkdir(parents=True, exist_ok=False)
    case_fields = [
        "test_row_id", "source_original_row_index", "source_file", "review", "title", "abstract",
        "doi", "pmid", "openalex_id", "true_label", "predicted_label", "probability",
        "threshold", "distance_from_threshold", "title_present", "abstract_present",
        "abstract_character_count", "abstract_word_count", "title_only", "identity_status",
        "overlap_train", "overlap_validation", "same_label_counterpart",
        "conflicting_label_counterpart", "identity_identifiers", "review_n", "review_positive",
        "review_negative", "review_tp", "review_fn", "review_recall", "review_precision",
        "review_pr_auc", "explicit_inclusion_reason_available", "eligibility_criteria_source",
    ]
    write_csv(output_dir / "false_negative_cases.csv", case_fields, cases)
    (output_dir / "missing_abstract_context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    criteria_source = "data/eligibility_criteria(排纳标准).xlsx"
    for index, case in enumerate(cases, start=1):
        (output_dir / f"false_negative_case_{index}.md").write_text(
            render_case(index, case, criteria, criteria_source), encoding="utf-8"
        )

    table_lines = [
        "| Case | Review | Abstract available | Probability | Threshold | Probability - threshold | Train/Validation overlap | Notes |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for index, case in enumerate(cases, start=1):
        table_lines.append(
            f"| {index} | {case['review']} | {yes_no(case['abstract_present'])} | "
            f"{case['probability']:.10f} | {threshold:.9f} | {case['distance_from_threshold']:.10f} | "
            f"None confirmed | Positive label only; no article-level inclusion rationale available |"
        )
    summary = f"""# Final False Negative Error Analysis Summary

## Verified facts

- Canonical Final Test false negatives: 2.
- Both records are from `van_Dis_2020`.
- Both records have titles and lack abstracts.
- Both records are `confirmed_non_overlap` with no Train or Validation match.
- Both records remain false negatives at the frozen Validation-selected threshold `{threshold}`.
- No explicit article-level inclusion rationale was found in repository artifacts.
- No causal or qualitative error category has been assigned automatically.

## Thesis-ready factual table

{chr(10).join(table_lines)}

## Missing-abstract contingency

| Group | Included articles | False negatives |
|---|---:|---:|
| Abstract available | {context['contingency'][0]['included_articles']} | {context['contingency'][0]['false_negatives']} |
| Abstract missing | {context['contingency'][1]['included_articles']} | {context['contingency'][1]['false_negatives']} |

Interpretation is limited to: **Both FN records had missing abstracts.** This does not establish that missing abstracts caused the errors.

## Evidence boundary

Available evidence includes the exact title, identifiers, positive label, frozen model probability and threshold, review-level eligibility criteria, overlap status, and per-review performance. Missing evidence includes article abstracts, article-level inclusion rationales, full texts, and any repository-supported explanation of why the model assigned these probabilities.

## Source integrity

All source hashes were unchanged before and after extraction. No canonical artifact was modified.
"""
    (output_dir / "error_analysis_summary.md").write_text(summary, encoding="utf-8")

    readme = f"""# Final False Negative Error Analysis Preparation

This directory contains factual evidence and blank manual-review sheets for all two
canonical Final Test false negatives. It does not contain an automatic explanation
or causal claim.

## Sources

- `outputs/final_test/final_test_predictions.csv`
- `data/processed/splits/test.csv`
- `outputs/final_test/final_test_metrics.json`
- `outputs/final_test/per_review_test_metrics.csv`
- `results/formal_training_strategy_20260910/thresholds/selected_validation_threshold.json`
- `data/eligibility_criteria(排纳标准).xlsx`, `{criteria['sheet']}!A{criteria['excel_row']}:H{criteria['excel_row']}`
- `results/article_overlap_sensitivity/false_negative_overlap_audit.csv`

## Integrity

- No training or model inference.
- No prediction regeneration or threshold change.
- No label or canonical-data modification.
- No external article retrieval.
- No automatic causal explanation or qualitative error category.
- All source hashes remained unchanged during extraction.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    produced = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    if produced != sorted(OUTPUT_NAMES):
        raise RuntimeError(f"Unexpected output manifest: {produced}")
    print(json.dumps({
        "status": "completed",
        "false_negatives": len(cases),
        "threshold": threshold,
        "cases": [
            {
                "test_row_id": case["test_row_id"],
                "probability": case["probability"],
                "distance": case["distance_from_threshold"],
                "abstract_present": case["abstract_present"],
                "identity_status": case["identity_status"],
            }
            for case in cases
        ],
        "missing_abstract_context": context,
        "files": produced,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
