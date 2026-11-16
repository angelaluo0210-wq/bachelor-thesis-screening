# Comprehensive Experimental Results Audit

Audit date: 13 November 2026.  This is a static, artifact-based audit: no training, inference, Test evaluation, or API request was performed.  `AP` is used where the artifact uses `validation_pr_auc` as the selection metric; the repository does not establish a different numerical PR-AUC definition for these runs.

## Executive summary

The evidence supports one primary, prospectively selected PubMedBERT result: CLS pooling + LoRA (adapter LR `5e-5`, seed 42), checkpoint selected by Validation AP, with a threshold selected on Validation only.  On the sealed 23,592-record / four-review Test set, the frozen system achieved AP 0.1553, ROC-AUC 0.9063, recall 99.58% (470/472), FNR 0.42% (2 FN), precision 2.55%, F1 4.97%, and workload reduction 21.80% at threshold 0.016082723.  These are the official PubMedBERT Test results.

The completed Qwen comparison is `qwen3.8-flash`, zero-shot, temperature 0, with `retain_unresolved_v1`.  It used the exact same Test-file SHA-256 and all 23,592 records. Its end-to-end policy retained uncertain (1,398) and API-failed (105) records, obtaining recall 50.00%, precision 9.95%, F1 16.60%, and workload reduction 89.95%; 105 API transport failures occurred. It is a same-population model-family comparison, but its hard categorical policy is not the same decision mechanism as a calibrated probability threshold.

The newer 5,000-record Qwen Full subset is stratified by review and gold label, rather than a prevalence-preserving independent Test. Its precision-matched PubMedBERT threshold (0.85556966) was selected by scanning the subset's gold labels. At closely matched precision (12.0827% versus Qwen's 12.0823%), PubMedBERT recall was 76% (FN 24) and Qwen recall was 47% (FN 53). This is useful exploratory paired evidence only; it must not replace the Validation-selected threshold or the primary Test result.

## Dataset and experimental setup

`config/canonical_split.json` is the source of truth: 18 Train, 4 Validation, and 4 Test reviews, with review sets declared disjoint. The primary Test reviews are Leenaars_2019, Muthu_2021, Radjenovic_2013, and van_Dis_2020. `outputs/final_test/final_test_config.json` records 23,592 Test records (472 positive, 23,120 negative), Test SHA-256 `23728...ab658`, and `test_was_evaluation_only: true`.

| Split | Reviews | Records | Positives | Role |
|---|---:|---:|---:|---|
| Train | 18 | 123,291 | 1,945 | fitting |
| Validation | 4 | 22,397 | 414 | configuration/checkpoint/threshold selection |
| Test | 4 | 23,592 | 472 | sealed final evaluation |

The selected checkpoint is CLS + LoRA, rank 8/alpha 16/dropout .05, adapters in query/value projections, seed 42, max length 512. The threshold rule was the largest Validation threshold with recall >= .95. Its Validation confusion matrix was TP=394, FN=20, TN=6,351, FP=15,632 (recall 95.17%, precision 2.46%, workload reduction 28.45%). Sources: `results/formal_training_strategy_20260910/thresholds/selected_validation_threshold.json`; `outputs/final_test/final_test_config.json`.

## Complete experiment inventory and results

The machine-readable complete inventory is `experiment_inventory.csv`; each row has an evidence status. “Completed” means an output artifact records an executed result. “Exploratory” means it is not eligible to replace the primary Test inference. Smoke, preflight, deployment, and connectivity artifacts are retained in the inventory as non-performance evidence rather than counted as formal experiments.

### Data quality and split controls

Data processing produced QC artifacts for missing values, duplicates, conflicting labels, mojibake, and the Leenaars repair (`reports/processed_data_summary.json`, `reports/duplicate_articles.csv`, `reports/conflicting_labels_classified.csv`, and related reports). The audit verifies their presence, not every historical data-repair decision. `config/review_split.json` is explicitly deprecated; the canonical split must be cited instead. Final-Test QC recorded 18 missing titles and 2,517 missing abstracts, including 12 positive records with missing abstracts.

### Model selection on Validation

| Stage / condition | Dataset | AP | ROC-AUC | Recall@.5 | Precision@.5 | Interpretation / evidence |
|---|---|---:|---:|---:|---:|---|
| Frozen CLS pooling | Validation | .0944 | .6767 | 58.45% | 2.92% | Highest pooling AP; `outputs/pooling_comparison/formal/20260905_122936/final_comparison.csv` |
| Frozen concat pooling | Validation | .0909 | .7235 | 57.25% | 4.29% | Lower AP; same source |
| Frozen mean pooling | Validation | .0391 | .6572 | 67.63% | 2.40% | Higher .5 recall but lower ranking AP; same source |
| Frozen attention pooling | Validation | .0387 | .6365 | 41.55% | 3.67% | Lower AP; same source |
| Classifier head only | Validation | .0892 | artifact | 62.08% | 2.70% | Formal strategy comparison |
| Partial unfreeze, LR 1e-5 | Validation | .0553 | artifact | 44.69% | 4.64% | Formal strategy comparison |
| Partial unfreeze, LR 2e-5 | Validation | .0552 | artifact | 38.65% | 4.59% | Formal strategy comparison |
| LoRA, adapter LR 5e-5 | Validation | **.1901** | artifact | 57.97% | 5.89% | Selected configuration and epoch 4 |
| LoRA, adapter LR 1e-4 | Validation | .1579 | artifact | 50.72% | 8.43% | Alternative formal run |

Formal strategy sources: `results/formal_training_strategy_20260910/comparison_manifest.json` and its five `*/metrics.json` files. Only the bold row is carried to primary Test.

### Traditional baselines and supplementary ablations

| Experiment | Dataset | AP / ROC-AUC | Threshold result | Status |
|---|---|---|---|---|
| TF-IDF + logistic regression, title+abstract | Validation | .0450 / .6474 | recall 95.17%, precision 2.00%, workload reduction 12.18% at .0009869893 | Completed; Validation-only |
| TF-IDF + logistic regression, criteria+title+abstract | Validation | .0403 / .5791 | recall 95.17%, precision 2.08%, workload reduction 15.59% at .0007978570 | Completed; Validation-only |
| Mean-LoRA, seeds 42/123/2026 | Validation | .0859/.0791/.0596 AP | all threshold .5; no calibrated/Test result | Completed supplementary |
| CLS-LoRA additional seeds 123/2026 | Validation | .1794/.1250 AP | no Test result | Completed supplementary |
| CLS-LoRA unweighted BCE | Validation | .1129 AP | TP=2, FN=412 at .5 | Completed ablation |

Sources: `supplementary_202610/tfidf_logreg/validation_summary.csv`, `supplementary_202610/mean_lora/.../metrics.json`, `supplementary_202610/cls_lora_multiseed/.../metrics.json`, and `supplementary_202610/cls_lora_unweighted_bce/.../metrics.json`. The TF-IDF systems use full available text while the encoder uses max length 512, so neither baseline is a strictly input-identical Test comparison.

### Primary PubMedBERT final Test and robustness

| Population / analysis | n / positives | Threshold | TP / FP / TN / FN | AP / ROC-AUC | Recall / precision / F1 | Workload reduction |
|---|---:|---:|---|---:|---:|---:|
| Official frozen final Test | 23,592 / 472 | .016082723 | 470 / 17,978 / 5,142 / 2 | .1553 / .9063 | 99.58% / 2.55% / 4.97% | 21.80% |
| Strict identifier-confirmed article-disjoint subset | 21,580 / 432 | .016082723 | 430 / artifact / artifact / 2 | .1692 / artifact | 99.54% / 2.58% / artifact | 22.68% |

The second row is post-hoc sensitivity analysis over frozen predictions, not an independent Test. Exact DOI/PMID/OpenAlex overlap was detected for 2,012 Test records (40 positives). Sources: `outputs/final_test/final_test_metrics.json`, `outputs/final_test/per_review_test_metrics.csv`, `results/article_overlap_sensitivity/article_disjoint_metrics.json`, and `results/article_overlap_sensitivity/overlap_summary.json`.

Per-review official PubMedBERT Test recalls: Leenaars 100% (17/17), Muthu 100% (335/335), Radjenovic 100% (48/48), van_Dis 97.22% (70/72). Both recorded false negatives are in van_Dis; the stored error analysis associates both with missing abstracts, but does not establish causality. Sources: `outputs/final_test/per_review_test_metrics.csv`; `results/final_error_analysis/error_analysis_summary.md`.

### Qwen experiments

The validation pilot (25 label-blind sampled records) and positive-case diagnostic (12 known positive Validation records) are completed diagnostic/API exercises, not population estimates. The Qwen final Test was a complete population run: model `qwen3.8-flash`, prompt `qwen-zero-shot-screening-v2`, zero-shot, JSON output, temperature 0, thinking disabled, three retries, 30 RPM. The checked-in prompt/parser implementation is in `src/llm_baseline/prompt.py`, `src/llm_baseline/evaluation.py`, and `src/llm_baseline/runner.py`; because the run manifest records a dirty worktree, current source alone cannot independently prove byte-identical prompt code at execution. The manifest's prompt SHA-256 remains the primary configuration evidence.

| Qwen condition | Population | TP / FP / TN / FN | Recall | Precision | F1 | Workload reduction | Evidence |
|---|---:|---|---:|---:|---:|---:|---|
| Full Test, end-to-end retain-unresolved policy | 23,592 / 472 positive | 236 / 2,135 / 20,985 / 236 | 50.00% | 9.95% | 16.60% | 89.95% | `results/qwen_final_test/qwen_final_test_metrics.json` |
| Full Test, valid-response-only | 23,487 / 472 positive | 236 / 2,030 / 20,985 / 236 | 50.00% | 10.41% | 17.24% | 90.35% | supplementary only; excludes 105 failures |
| Stratified 5,000-record Test subset, Qwen Full | 5,000 / 100 positive | 47 / 342 / 4,558 / 53 | 47.00% | 12.0823% | 19.22% | 92.22% | `outputs/pubmedbert_vs_qwen_full_test_subset/comparison_evaluation.json` |

Full-Test output outcomes were 868 explicit includes, 21,221 excludes, 1,398 uncertain, 105 API transport failures, and zero malformed outputs. End-to-end handling retains unresolved/failed records; do not replace it by valid-response-only figures.

### Precision-matched exploratory comparison

All three artifacts confirm matching of subset record IDs and gold labels. Qwen’s 12.082262% precision was the target. Of 4,996 observed-score candidate thresholds, .85556966 minimized absolute precision difference; ties were specified to prefer higher recall and then larger threshold. The achieved PubMedBERT precision was 12.082671%, a +0.000409 percentage-point signed difference.

| System / rule | Retained | TP / FP / TN / FN | Precision | Recall | F1 | Workload reduction |
|---|---:|---|---:|---:|---:|---:|
| Qwen Full, exclude=auto-exclude | 389 | 47 / 342 / 4,558 / 53 | 12.0823% | 47% | 19.22% | 92.22% |
| PubMedBERT original frozen threshold .016082723 | 3,859 | 100 / 3,759 / 1,141 / 0 | 2.5913% | 100% | 5.05% | 22.82% |
| PubMedBERT matched threshold .85556966 | 629 | 76 / 553 / 4,347 / 24 | 12.0827% | 76% | 20.85% | 87.42% |

**Evidence boundary:** the matched threshold uses Test-subset gold labels and is therefore post-hoc, in-sample, exploratory, and optimistically selected. It is not an independently validated threshold, cannot be advertised as a pre-specified Test result, and cannot alter the original frozen model/threshold result. Sources: `outputs/pubmedbert_vs_qwen_full_test_subset/precision_matched_comparison/{precision_matched_summary.json,comparison_table.csv,precision_matched_report.md}`.

## Logical relation, comparability, and risks

* Model-selection evidence: pooling and formal training-strategy comparisons, all Validation only. Threshold calibration is a separate Validation-only decision.
* Final performance evidence: only PubMedBERT frozen final Test and Qwen full-Test end-to-end results. They share the same Test SHA-256/population but differ in score-vs-categorical decision process and unresolved-response policy.
* Supplementary evidence: traditional baselines, seeds, loss ablation, overlap sensitivity, error analyses, Qwen pilot/diagnostic, Qwen subset, and precision matching. None authorizes changing the primary result.
* The Test subset was deliberately stratified by label (100/5,000 positives versus 472/23,592 in the full Test), so subset precision/workload must not be generalized as full-Test prevalence performance.
* Article-overlap sensitivity reduces but cannot eliminate leakage concerns: it is post-hoc and detects exact identifiers only. Precision matching introduces explicit Test-label threshold-selection bias. The final Qwen and original PubMedBERT Tests reuse the same Test data for two model-family evaluations; this is acceptable descriptive comparison but prohibits further model/prompt/threshold tuning against those labels.

## Overall findings

Within the documented search, CLS-LoRA was the highest-AP Validation configuration. Its Validation-calibrated operating point retained 470 of 472 eligible Test records, but forwarded 78.20% of records for manual screening. Qwen's full-Test policy excluded substantially more records but missed 236 of 472 eligible records. The supported claim is a retrospective decision-support trade-off, not autonomous exclusion, real-world time savings, or universal transfer. Repeated seeds and a broader split protocol remain needed; newer seed results specifically weaken the former thesis wording that no repeat runs existed.

