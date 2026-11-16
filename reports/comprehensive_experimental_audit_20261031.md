# Comprehensive Experimental Audit — 31 October 2026

## 1. Executive Summary

This audit uses files present in the workspace and the remote \`origin/work\` ref checked on 31 October 2026. The remote and local \`work\` heads both resolve to \`150c1334e5f1e35a127b246bcdbb24d971c0c7b1\`. No training, Test evaluation, or API request was run.

The completed primary experiment is a review-disjoint PubMedBERT evaluation. The selected system is CLS pooling plus LoRA (\`adapter_lr=5e-5\`, seed 42), selected by Validation AP. Its Validation-only frozen threshold is 0.016082723, the largest threshold with recall >=0.95. On the complete four-review Test set: TP=470, FN=2, TN=5,142, FP=17,978; Recall=99.58%, FNR=0.42%, Precision=2.55%, F1=4.97%, workload reduction=21.80%.

The completed Qwen3.8-Flash zero-shot Test is a distinct hard-decision experiment on the same 23,592 records. Under \`retain_unresolved_v1\`: TP=236, FN=236, TN=20,985, FP=2,135; Recall=50.00%, FNR=50.00%, Precision=9.95%, F1=16.60%, workload reduction=89.95%. It had 105 API transport failures and 1,398 uncertain responses; both were retained for screening in its end-to-end denominator.

The PubMedBERT result supports decision support with a measured 21.80% exclusion-side workload reduction and high observed Test recall, not autonomous exclusion. Qwen reduces more manual volume but misses half of eligible Test records, and does not meet the high-recall objective.

## 2. Dataset and Evaluation Protocol

| Split | Reviews | Records | Positive | Negative | Role |
|---|---:|---:|---:|---:|---|
| Train | 18 | 123,291 | 1,945 | 121,346 | fitting |
| Validation | 4 | 22,397 | 414 | 21,983 | selection and threshold calibration |
| Test | 4 | 23,592 | 472 | 23,120 | locked final evaluation |

The Test reviews are \`Leenaars_2019\` (5,811; 17 positive), \`Muthu_2021\` (2,718; 335), \`Radjenovic_2013\` (5,935; 48), and \`van_Dis_2020\` (9,128; 72). Splits are review-disjoint. Validation figures use threshold 0.5 unless a calibrated threshold is named. AP/PR-AUC is score-based; Recall/FNR/Precision/F1/workload reduction are operating-point metrics.

The final PubMedBERT checkpoint was selected by Validation AP. Calibration selected 0.016082723 on Validation only as the largest threshold with recall >=95%: TP=394, FN=20, TN=6,351, FP=15,632, Recall=95.17%, FNR=4.83%, Precision=2.46%, workload reduction=28.45%. No Test labels selected the checkpoint, hyperparameters, or threshold.

## 3. Original PubMedBERT Experiment Tables

### 3.1 Pooling strategies — completed formal Validation experiment

All runs: frozen PubMedBERT backbone, seed 42, batch size 32, five epochs, max length 512, AdamW, learning rate 1e-4, dropout 0.1, weighted BCEWithLogitsLoss with \`pos_weight=62.388688946\`. Selection was Validation PR-AUC/AP, not Test.

| Pooling | Best epoch | Validation PR-AUC | Recall | FNR | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|
| CLS | 1 | 0.0944 | 58.45% | 41.55% | 2.92% | 5.56% |
| Concat | 1 | 0.0909 | 57.25% | 42.75% | 4.29% | 7.98% |
| Mean | 5 | 0.0391 | 67.63% | 32.37% | 2.40% | 4.63% |
| Attention | 3 | 0.0387 | 41.55% | 58.45% | 3.67% | 6.74% |

CLS was selected because it has the highest Validation PR-AUC. Mean has higher threshold-0.5 recall, but lower ranking performance.

### 3.2 Fine-tuning strategies — completed formal Validation experiment

All rows use CLS pooling, seed 42, five epochs, batch size 32, max length 512, weighted BCE, classifier LR=1e-4, and Validation AP checkpoint selection.

| Strategy | Variable LR | Trainable params | Best epoch | Validation AP | Recall@0.5 | FNR@0.5 | Precision@0.5 | F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Classifier head only | 1e-4 head | 769 | 2 | 0.0892 | 62.08% | 37.92% | 2.70% | 5.17% |
| Partial unfreezing | 1e-5 backbone | 14,176,513 | 2 | 0.0553 | 44.69% | 55.31% | 4.64% | 8.41% |
| Partial unfreezing | 2e-5 backbone | 14,176,513 | 2 | 0.0552 | 38.65% | 61.35% | 4.59% | 8.20% |
| LoRA | 5e-5 adapter | 295,681 | 4 | **0.1901** | 57.97% | 42.03% | 5.89% | 10.69% |
| LoRA | 1e-4 adapter | 295,681 | 1 | 0.1579 | 50.72% | 49.28% | 8.43% | 14.46% |

The selected configuration is CLS LoRA, adapter LR=5e-5. LoRA uses rank 8, alpha 16, dropout 0.05, query/value adapters in all encoder layers, and its epoch-4 checkpoint.

### 3.3 Frozen final PubMedBERT Test — completed primary result

| Metric | Micro-global Test result |
|---|---:|
| AP / ROC-AUC | 0.1553 / 0.9063 |
| Frozen threshold | 0.016082723 |
| TP / FN / TN / FP | 470 / 2 / 5,142 / 17,978 |
| Recall / FNR | 99.58% / 0.42% |
| Precision / F1 | 2.55% / 4.97% |
| Predicted include / workload reduction | 78.20% / 21.80% |
| 95% CIs | N/A: not in final PubMedBERT artifact |

| Test review | TP / FN / TN / FP | Recall | FNR | Precision | F1 | Workload reduction | AP / ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| Leenaars_2019 | 17 / 0 / 1,427 / 4,367 | 100.00% | 0.00% | 0.39% | 0.77% | 24.56% | 0.7581 / 0.9990 |
| Muthu_2021 | 335 / 0 / 10 / 2,373 | 100.00% | 0.00% | 12.37% | 22.02% | 0.37% | 0.2525 / 0.7355 |
| Radjenovic_2013 | 48 / 0 / 315 / 5,572 | 100.00% | 0.00% | 0.85% | 1.69% | 5.31% | 0.1128 / 0.9381 |
| van_Dis_2020 | 70 / 2 / 3,390 / 5,666 | 97.22% | 2.78% | 1.22% | 2.41% | 37.16% | 0.0521 / 0.8144 |
| Macro review-level | N/A | 99.31% | 0.69% | 3.71% | 6.72% | 16.85% | N/A |

A completed post-hoc article-identity sensitivity analysis, using the same frozen scores and threshold, yields n=21,580, 432 positives, TP=430, FN=2, Recall=99.54%, FNR=0.46%, Precision=2.58%, workload reduction=22.68%.

## 4. Supplementary Experiment Tables

### 4.1 TF-IDF + Logistic Regression — completed Validation only

Both baselines use word TF-IDF (1–2 grams, max 200,000 features, min_df=2, sublinear TF), balanced liblinear logistic regression, max_iter=2000, random state 42, C grid {0.01, 0.1, 1, 10}, and select C=10 by Validation PR-AUC. Their full available input is not subject to the transformer 512-token limit; Test access is forbidden.

| Input | Val AP / ROC-AUC | Calibrated threshold | TP / FN / TN / FP | Recall / FNR | Precision / F1 | Workload reduction |
|---|---:|---:|---|---:|---:|---:|
| Title + abstract | 0.0450 / 0.6474 | 0.0009869893 | 394 / 20 / 2,707 / 19,276 | 95.17% / 4.83% | 2.00% / 3.92% | 12.18% |
| Criteria + title + abstract | 0.0403 / 0.5791 | 0.0007978570 | 394 / 20 / 3,471 / 18,512 | 95.17% / 4.83% | 2.08% / 4.08% | 15.59% |

### 4.2 Mean pooling + LoRA — completed Validation only

Configuration: weighted BCE (\`pos_weight=62.388688946\`), adapter LR=5e-5, head LR=1e-4, rank 8, alpha 16, query/value targets, five epochs, batch size 32, max length 512. Results use threshold 0.5; no calibrated threshold or Test evaluation exists.

| Seed | Best epoch | Validation AP / ROC-AUC | TP / FN / TN / FP | Recall / FNR | Precision / F1 | Workload reduction |
|---:|---:|---:|---|---:|---:|---:|
| 42 | 3 | 0.0859 / 0.7720 | 166 / 248 / 18,928 / 3,055 | 40.10% / 59.90% | 5.15% / 9.13% | 85.62% |
| 123 | 4 | 0.0791 / 0.7573 | 181 / 233 / 18,613 / 3,370 | 43.72% / 56.28% | 5.10% / 9.13% | 84.15% |
| 2026 | 1 | 0.0596 / 0.7465 | 260 / 154 / 15,213 / 6,770 | 62.80% / 37.20% | 3.70% / 6.99% | 68.61% |

Mean LoRA is below selected CLS-LoRA seed-42 Validation AP (0.1901) and varies substantially by seed. It does not support replacing CLS pooling; no mean-pooling Test claim is available.

### 4.3 CLS LoRA random-seed checks — completed Validation only

The formal selected configuration supplies seed 42 (AP=0.1901). Added seeds 123 and 2026 give AP=0.1794 and 0.1250; Recall@0.5=35.99% and 46.38%. The three-seed AP range is 0.1250–0.1901. This is evidence of repeatability but not tight seed stability; only seed 42 has final Test results.

### 4.4 Class weighting / loss function — completed Validation ablation

Train prevalence is 1.58% (1,945/123,291). Weighted BCE uses \`pos_weight=train_negative/train_positive=62.388688946\`. The matched unweighted run keeps CLS LoRA, seed 42, architecture, learning rates, and epoch budget fixed.

| Loss | Validation AP | TP / FN / TN / FP @0.5 | Recall / FNR | Precision / F1 | Workload reduction |
|---|---:|---|---:|---:|---:|
| Weighted BCE | 0.1901 | 240 / 174 / 18,148 / 3,835 | 57.97% / 42.03% | 5.89% / 10.69% | 82.08% |
| Unweighted BCE | 0.1129 | 2 / 412 / 21,982 / 1 | 0.48% / 99.52% | 66.67% / 0.96% | 99.99% |

This supports class weighting: without it, the default threshold gives almost no positive predictions and is unsuitable for high-recall exclusion. Other numeric weights and an unweighted calibrated threshold were not tested.

### 4.5 Non-formal / incomplete artifacts

Smoke runs, AutoDL deployment validation, API connectivity checks, and preflight plans exist but are not performance experiments. No final Test evaluation exists for TF-IDF, mean LoRA, added CLS-LoRA seeds, or unweighted BCE.

## 5. Qwen LLM Baseline Results

### 5.1 Validation pilot and diagnostic

The completed label-blind Validation pilot sampled 25/22,397 Validation records (seed 202610); all 25 responses were explicit excludes. It is an API/prompt pilot, not a population performance estimate. The completed positive-case diagnostic queried 12 known-positive Validation records: 6 explicit includes and 6 explicit excludes. As a positive-conditioned sample, it has no valid overall Recall, FNR, Precision, workload-reduction, or Test-performance estimate.

### 5.2 Qwen3.8-Flash zero-shot final Test — completed

Configuration: \`qwen3.8-flash\`, \`qwen-zero-shot-screening-v2\`, temperature 0, thinking disabled, JSON responses, max three retries, 30 requests/minute, \`retain_unresolved_v1\`. Complete population verification: expected=completed=23,592; missing=extra=0.

Outcomes: 868 explicit includes, 21,221 explicit excludes, 1,398 uncertain, 105 API transport failures, 0 malformed outputs. Uncertain and failed records are retained for screening. Therefore primary predicted include=868+1,398+105=2,371 and primary exclusion=21,221.

| End-to-end primary metric | Result and 95% CI where available |
|---|---:|
| TP / FN / TN / FP | 236 / 236 / 20,985 / 2,135 |
| Recall | 50.00% [45.51%, 54.49%] |
| FNR | 50.00% [45.51%, 54.49%] |
| Precision | 9.95% [8.81%, 11.22%] |
| F1 | 16.60% |
| Workload reduction | 89.95% [89.56%, 90.33%] |
| AP / ROC-AUC | N/A: hard binary decisions |
| Macro Recall / FNR / Precision / F1 / workload reduction | 71.54% / 28.46% / 12.90% / 18.57% / 89.25% |

| Test review | TP / FN / TN / FP | Recall (95% CI) | FNR (95% CI) | Precision | F1 | Workload reduction (95% CI) |
|---|---|---:|---:|---:|---:|---:|
| Leenaars_2019 | 15 / 2 / 5,260 / 534 | 88.24% [65.66, 96.71] | 11.76% [3.29, 34.34] | 2.73% | 5.30% | 90.55% [89.77, 91.28] |
| Muthu_2021 | 128 / 207 / 2,044 / 339 | 38.21% [33.17, 43.52] | 61.79% [56.48, 66.83] | 27.41% | 31.92% | 82.82% [81.35, 84.19] |
| Radjenovic_2013 | 44 / 4 / 5,672 / 215 | 91.67% [80.45, 96.71] | 8.33% [3.29, 19.55] | 16.99% | 28.66% | 95.64% [95.09, 96.13] |
| van_Dis_2020 | 49 / 23 / 8,009 / 1,047 | 68.06% [56.61, 77.67] | 31.94% [22.33, 43.39] | 4.47% | 8.39% | 87.99% [87.31, 88.64] |

The valid-response-only calculation (n=23,487) excludes 105 technical failures and is supplementary only: same TP/FN=236/236, Precision=10.41%, workload reduction=90.35%.

## 6. Cross-model Comparison

| Condition | Population | Decision source | Recall / FNR | Precision / F1 | Workload reduction | Comparability |
|---|---|---|---:|---:|---:|---|
| PubMedBERT CLS-LoRA | same 23,592 Test | score threshold frozen on Validation | 99.58% / 0.42% | 2.55% / 4.97% | 21.80% | primary model |
| Qwen end-to-end | same 23,592 Test | zero-shot hard decisions, unresolved retained | 50.00% / 50.00% | 9.95% / 16.60% | 89.95% | same population; policy/output type differ |
| TF-IDF baselines | Validation only, 22,397 | Validation-calibrated scores | 95.17% / 4.83% | 2.00–2.08% / 3.92–4.08% | 12.18–15.59% | not a Test comparison |

PubMedBERT and Qwen share complete Test population and have end-to-end outcome records, but no paired error analysis was requested. PubMedBERT is score-thresholded for a Validation recall target; Qwen has no ranking score and retains uncertain/API-failed cases. Use this as observed Test operating performance, not a significance claim.

## 7. Responses to Supervisor Feedback

1. **Baselines:** two completed traditional TF-IDF/logistic baselines cover title/abstract-only and criteria-augmented input. Four pooling and five fine-tuning configurations provide neural within-family comparisons. Traditional baselines are Validation-only.
2. **Genuine LLM:** yes. Qwen3.8-Flash zero-shot ran on all 23,592 Test records. It gives Recall 50.00% and FNR 50.00%, hence fails the high-recall objective.
3. **Seeds:** only partial stability evidence. CLS-LoRA AP is 0.1901/0.1794/0.1250 for seeds 42/123/2026; mean-LoRA is 0.0859/0.0791/0.0596. Only seed 42 has a Test result.
4. **Mean + LoRA:** all three runs are below selected CLS-LoRA AP; no Test evaluation exists.
5. **Weighting:** the matched unweighted ablation finds 2/414 Validation positives at threshold 0.5. Weighting is necessary to obtain a recall-oriented operating point before calibration.

## 8. Methodological Caveats

- The primary Test has four reviews and 472 positives. Final PubMedBERT artifacts contain no CIs.
- PubMedBERT forwards 78.20% of Test records; its 21.80% workload reduction is prediction-derived, not proof that exclusions can be made without human oversight.
- Qwen AP/ROC-AUC are N/A; valid-response-only metrics must not replace end-to-end metrics.
- Pooling, fine-tuning, baselines, mean-LoRA, seeds, and loss ablation are Validation-only unless explicitly labelled Test.
- TF-IDF uses full available text, not the 512-token transformer input. Qwen and PubMedBERT differ in decision mechanism; no paired error analysis exists.
- Article-overlap sensitivity is a frozen-prediction subset, not an independent Test run.

## 9. Remaining Tasks

- A Test-level classical baseline requires pre-specification before accessing Test labels; current manifests prohibit it.
- Stronger robustness claims need pre-specified seed aggregation and more seeds without using the locked Test set.
- Compute final PubMedBERT CIs from stored predictions if required, without changing the threshold.
- Any paired PubMedBERT–Qwen error analysis needs a separately specified aligned-prediction analysis.
- Keep smoke, preflight, connectivity, and positive-conditioned diagnostics out of performance tables.

## 10. Source Evidence Index

| Evidence | Primary file(s) |
|---|---|
| Split counts/reviews | \`data/processed/splits/split_summary.csv\`; \`config/canonical_split.json\` |
| Pooling | \`outputs/pooling_comparison/formal/20260905_122936/final_comparison.csv\`; \`outputs/pooling_comparison/formal/20260905_122936/pooling_comparison_manifest_20260905_122936.json\` |
| Fine-tuning/checkpoint | \`results/formal_training_strategy_20260910/comparison_manifest.json\`; \`results/formal_training_strategy_20260910/*/metrics.json\`; \`results/formal_training_strategy_20260910/lora_5e-5/config.json\` |
| Threshold | \`results/formal_training_strategy_20260910/thresholds/selected_validation_threshold.json\` |
| Primary Test | \`outputs/final_test/final_test_config.json\`; \`outputs/final_test/final_test_metrics.json\`; \`outputs/final_test/per_review_test_metrics.csv\` |
| Overlap sensitivity | \`results/article_overlap_sensitivity/article_disjoint_metrics.json\` |
| Traditional baselines | \`supplementary_202610/tfidf_logreg/*/config.json\`; \`supplementary_202610/tfidf_logreg/*/validation_metrics.json\`; \`supplementary_202610/tfidf_logreg/*/threshold_calibration.json\` |
| Mean/seed/loss runs | \`supplementary_202610/mean_lora/20261028_085641_aaeca089/**/metrics.json\`; \`supplementary_202610/cls_lora_multiseed/20261028_161144_dfbd2800/**/metrics.json\`; \`supplementary_202610/cls_lora_unweighted_bce/20261028_210437_bdbd9750/**/metrics.json\` |
| Qwen final | \`results/qwen_final_test/run_configuration.json\`; \`results/qwen_final_test/execution_summary.json\`; \`results/qwen_final_test/qwen_final_test_metrics.json\`; \`results/qwen_final_test/per_review_qwen_metrics.csv\` |
| Qwen pilot/diagnostic | \`supplementary_202610/llm_baseline/validation_pilot_api_20261028/execution_summary.json\`; \`supplementary_202610/llm_baseline/validation_positive_case_diagnostic_20261028/diagnostic_summary.json\`; \`supplementary_202610/llm_baseline/validation_positive_case_diagnostic_20261028/positive_case_error_analysis.md\` |
| Thesis statements checked | \`thesis/chapters/chapter4_results.tex\`; \`thesis/chapters/chapter5_discussion.tex\`; \`thesis/chapters/chapter6_conclusion.tex\`; \`thesis/thesis.tex\` |

### Audit findings on consistency

No numerical conflict was found between current thesis PubMedBERT final-Test statements (470/472, 99.58% Recall, 21.80% workload reduction) and its primary artifacts. Qwen is newer and not incorporated into the current thesis result chapters. The categorical risk is material: Qwen's 50.00% Recall / 89.95% workload reduction must never be attributed to PubMedBERT, and Validation-only results must not be presented as Test results. The pooling manifest says “PR-AUC” while later files say “AP”; this audit preserves those labels and does not treat them as distinct measurements.

