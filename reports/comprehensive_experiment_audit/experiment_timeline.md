# Experiment timeline

Dates are taken from run names/manifests where available. “Unknown” is used instead of inferring a date from an unrelated file timestamp.

| Date | Experiment(s) | Main configuration / finding | Included in thesis now? | Evidence |
|---|---|---|---|---|
| Unknown | Corpus QC and split | Canonical 18/4/4 review-disjoint split; QC artifacts available | Yes, methodological description | `config/canonical_split.json`; `reports/processed_data_summary.json` |
| 2026-09-05 | Pooling comparison | Frozen PubMedBERT; CLS highest Validation AP (.0944) | Yes | `outputs/pooling_comparison/formal/20260905_122936/final_comparison.csv` |
| 2026-09-10 | Training-strategy comparison | CLS LoRA 5e-5 highest Validation AP (.1901) | Yes | `results/formal_training_strategy_20260910/comparison_manifest.json` |
| 2026-09-10 | Threshold calibration | Validation-only .016082723 | Yes | `results/formal_training_strategy_20260910/thresholds/selected_validation_threshold.json` |
| 2026-09-19 | Frozen PubMedBERT final Test | 470/472 retained; 21.80% workload reduction | Yes | `outputs/final_test/*` |
| Unknown | Article-overlap sensitivity and PubMedBERT error analysis | frozen subset analysis; two FNs audited | Yes | `results/article_overlap_sensitivity/*`; `results/final_error_analysis/*` |
| 2026-10-28 | TF-IDF baselines, seeds, mean LoRA, unweighted loss | Validation-only supplementary evidence | No | `supplementary_202610/*` |
| 2026-10-28 | Qwen pilot and positive diagnostic | diagnostics only, not performance estimates | No | `supplementary_202610/llm_baseline/*` |
| 2026-10-29 | Qwen full Test | recall 50%, workload reduction 89.95% | No | `results/qwen_final_test/*` |
| 2026-11-04 | Qwen FN audit (partial) | all 236 extracted; only 11 individually reviewed, so no population-level cause claim | No | `reports/qwen_fn_audit_20261104/*` |
| 2026-11-13 | Qwen Full 5,000-record stratified Test subset | recall 47%; not population-prevalence representative | No | `outputs/pubmedbert_vs_qwen_full_test_subset/*` |
| 2026-11-13 | Precision-matched analysis | threshold .85556966 selected on subset labels; exploratory | No | `outputs/pubmedbert_vs_qwen_full_test_subset/precision_matched_comparison/*` |

Smoke runs, API-connectivity checks, AutoDL validation, and preflight-only artifacts are not placed on the performance timeline because they do not establish an estimand-level model result.
