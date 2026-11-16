# Experiment audit notes

This is a read-only, artifact-based audit. No training, inference, threshold change, or Test re-evaluation was performed.

## Material findings

- **Canonical split.** `config/canonical_split.json` supersedes the historical `config/review_split.json` / `reports/proposed_review_split.csv`. It declares review-disjoint Train (18), Validation (4), and Test (4) splits. Cite the canonical file.
- **Formal evidence.** The pooling comparison (`outputs/pooling_comparison/formal/20260905_122936/`) and five strategy runs (`results/formal_training_strategy_20260910/`) are formal Validation-only experiments. The selected run is CLS + LoRA, adapter LR 5e-5, epoch 4, seed 42.
- **Threshold boundary.** The 0.016082723 operating threshold was selected on Validation only, using the largest threshold whose Validation recall was at least 0.95. Do not call default-threshold (0.5) metrics calibrated metrics.
- **Final Test.** The frozen primary Test result is from `outputs/final_test/`. The two recorded false negatives are both van_Dis_2020 records with missing abstracts; this association does not establish causality. Details: `results/final_error_analysis/false_negative_cases.csv`.
- **Overlap.** The post-hoc exact identifier analysis detected 2,012 Test records (40 positives) overlapping Train and/or Validation by normalised DOI, PMID, or OpenAlex ID. The strict subset uses the original frozen predictions and threshold; it is sensitivity analysis, not a new independent Test. Sources: `results/article_overlap_sensitivity/overlap_summary.json`, `article_disjoint_metrics.json`.

## Data quality and provenance

- Processed source: 26 reviews, 169,277 rows, 2,833 label-1 rows, 166,432 label-0 rows, and 12 other/malformed labels preserved. Source: `reports/processed_data_summary.json`.
- All 26 review keys have an exact eligibility-workbook match; one review (Donners_2021) has no separate exclusion field, although the unified eligibility criterion is available. Source: `reports/criteria_matching_report.csv`.
- QC artifacts exist for missing values, duplicates, label conflicts, mojibake, and the Leenaars repair under `reports/`. The processed summary records 58 modified text fields/records and 106 confirmed repair operations. These are historical data-preparation evidence, not experimental results.
- Missing title/abstract counts are split-specific in `data/processed/splits/split_summary.csv`: Train 239/11,331; Validation 17/1,650; Test 18/2,516. Test positives with missing abstracts: 12.

## Important limitations / manual checks

1. Selected final PubMedBERT performance is a single-seed Test result. Additional seeds are Validation-only supplementary runs; do not report a multi-seed Test estimate.
2. The formal artifacts name the checkpoint-selection metric `validation_pr_auc`; this report preserves it as Validation AP rather than claiming a different PR-AUC calculation. Confirm the metric implementation before fixing thesis terminology.
3. Classical TF-IDF/logistic-regression baselines are Validation-only and use full available text, while the transformer uses a 512-token construction. They are not strict input-identical Test comparisons. Sources: `supplementary_202610/tfidf_logreg/`.
4. Qwen results are a separate complete-Test hard-decision experiment (`results/qwen_final_test/`), not a ranking-score experiment; AP and ROC-AUC are N/A. Its end-to-end policy retains uncertain and API-failed records. Never combine its metrics with PubMedBERT metrics.
5. Some historic manifests contain unavailable Linux absolute paths. Local copied JSON/CSV artifacts cited in this report were used instead.
6. Workload reduction is the fraction predicted for exclusion. It is not a measured reduction in human time, cost, or screening risk.
