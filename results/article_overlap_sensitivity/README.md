# Post-hoc Article-Overlap Sensitivity Analysis

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

`0.016082723` from Validation target-recall calibration; unchanged.

## Test identity groups

| Group | N | Positive | Negative | Percent |
|---|---:|---:|---:|---:|
| confirmed_overlap | 2012 | 40 | 1972 | 8.528315% |
| confirmed_non_overlap | 21580 | 432 | 21148 | 91.471685% |
| identity_unknown | 0 | 0 | 0 | 0.000000% |

## Primary vs strict sensitivity

| Metric | Review-disjoint Test | Strict article-disjoint | Difference |
|---|---:|---:|---:|
| n | 23592 | 21580 | -2012 |
| positive | 472 | 432 | -40 |
| negative | 23120 | 21148 | -1972 |
| recall | 0.995762712 | 0.995370370 | -0.000392341 |
| fnr | 0.004237288 | 0.004629630 | 0.000392341 |
| precision | 0.025477016 | 0.025771651 | 0.000294635 |
| f1 | 0.049682875 | 0.050242449 | 0.000559574 |
| pr_auc | 0.155312775 | 0.169193557 | 0.013880781 |
| roc_auc | 0.906307731 | 0.912206259 | 0.005898528 |
| workload_reduction | 0.218040014 | 0.226830399 | 0.008790385 |


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
