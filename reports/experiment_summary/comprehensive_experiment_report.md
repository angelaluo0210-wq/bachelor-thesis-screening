# Comprehensive Experimental Results Report

**Scope.** This English report is a read-only audit of saved artifacts. It does not retrain a model, run Test inference, alter a threshold, or overwrite any experimental output. `AP` below denotes the stored `validation_pr_auc` selection value unless explicitly labelled otherwise.

## 1. Dataset preparation

The corpus comprises 26 systematic reviews and 169,277 processed rows (2,833 positive, 166,432 negative, 12 other/malformed labels preserved). The active split is review-level and declared disjoint by `config/canonical_split.json`: Train: 18 reviews, 123,291 records (1,945 positive; 1.58%); Validation: 4 reviews, 22,397 records (414; 1.85%); Test: 4 reviews, 23,592 records (472; 2.00%). The Test reviews are Leenaars_2019, Muthu_2021, Radjenovic_2013, and van_Dis_2020; the complete memberships are in the canonical split file.

Eligibility criteria are loaded from `data/eligibility_criteria(排纳标准).xlsx`. All review keys have exact matches in `reports/criteria_matching_report.csv`. The project records QC for missing fields, duplicates, conflicting labels, character-encoding repair, and a Leenaars repair under `reports/`; it records 106 confirmed repairs and 58 modified text fields. Split QC (`data/processed/splits/split_summary.csv`) records missing titles/abstracts of 239/11,331 (Train), 17/1,650 (Validation), and 18/2,516 (Test). Exact DOI/PMID/OpenAlex overlap does exist across reviews; see Section 8.

## 2. Model architecture and input construction

The formal system uses the local PubMedBERT checkpoint corresponding in code to `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` (`src/train_classifier_head.py`). Its BERT configuration is 12 layers, 768 hidden units, 12 attention heads, 30,522 vocabulary items and 512 position embeddings (`model/pubmedbert/config.json`); the selected run reports 109,777,921 total parameters.

Input combines eligibility material with title and abstract as one tokenized BERT sequence. `src/train_classifier_head.py` constructs labelled sections and applies prefix truncation with an allocation policy that reserves capacity for title and abstract (`build_input_text` and tokenisation code); maximum length is 512. The selected model uses CLS pooling, a linear 768-to-1 classifier (769 parameters), weighted `BCEWithLogitsLoss` with `pos_weight=62.388688946` (= Train negatives / positives), AdamW (weight decay 0.01), batch size 32, five epochs, seed 42 and dropout 0.1. It uses LoRA rank 8, alpha 16, dropout 0.05 in query/value projections of all encoder layers; head LR is 1e-4 and adapter LR is 5e-5. Environment: CUDA on NVIDIA GeForce RTX 4080 SUPER; torch 2.6.0+cu124, transformers 5.16.1, PEFT 0.20.0. Source: `results/formal_training_strategy_20260910/lora_5e-5/config.json`.

## 3. Formal pooling comparison (Validation)

All four rows are formal frozen-encoder, seed-42 Validation experiments, with selection by AP; fixed-threshold statistics are at 0.5 and are not threshold-calibrated.

| Pooling | Best epoch | Validation AP | ROC-AUC | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|
| CLS | 1 | 0.0944 | 0.6767 | 58.45% | 2.92% | 5.56% |
| Concatenated | 1 | 0.0909 | 0.7235 | 57.25% | 4.29% | 7.98% |
| Mean | 5 | 0.0391 | 0.6572 | 67.63% | 2.40% | 4.63% |
| Attention | 3 | 0.0387 | 0.6365 | 41.55% | 3.67% | 6.74% |

CLS was selected because it had the largest Validation AP, despite mean pooling having higher recall at the uncalibrated 0.5 threshold. Sources: `outputs/pooling_comparison/formal/20260905_122936/final_comparison.csv`, per-pooling `metrics.json`, `config.json`, and `training_history.csv` files.

## 4. Formal training-strategy comparison (Validation)

All rows use CLS pooling, seed 42, five epochs, batch size 32, max length 512 and weighted BCE. The best checkpoint is selected by Validation AP; performance columns other than AP/ROC-AUC are at threshold 0.5.

| Strategy | Learning rate(s) | Best epoch | AP | ROC-AUC | Recall | Precision | F1 | Trainable parameters |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Classifier head only | head 1e-4 | 2 | 0.0892 | 0.6800 | 62.08% | 2.70% | 5.17% | 769 |
| Partial unfreezing | backbone 1e-5; head 1e-4 | 2 | 0.0553 | 0.7392 | 44.69% | 4.64% | 8.41% | 14,176,513 |
| Partial unfreezing | backbone 2e-5; head 1e-4 | 2 | 0.0552 | 0.7389 | 38.65% | 4.59% | 8.20% | 14,176,513 |
| LoRA | adapter 5e-5; head 1e-4 | 4 | **0.1901** | 0.7570 | 57.97% | 5.89% | 10.69% | 295,681 |
| LoRA | adapter 1e-4; head 1e-4 | 1 | 0.1579 | 0.7597 | 50.72% | 8.43% | 14.46% | 295,681 |

The selected strategy is CLS-LoRA with adapter LR 5e-5, epoch 4. Exact configurations and per-epoch AP histories are in `results/formal_training_strategy_20260910/comparison_manifest.json`, each run's `config.json`, `metrics.json`, and `training_history.csv`.

Supplementary—not formal replacements—include mean-LoRA multi-seed runs, CLS-LoRA added seeds, and unweighted-BCE ablation under `supplementary_202610/`. Their absence of calibrated/final-Test artifacts must be retained in any thesis description.

## 5. Baselines

Two executed TF-IDF + balanced logistic-regression baselines are Validation-only: title+abstract (AP 0.0450, ROC-AUC 0.6474) and criteria+title+abstract (AP 0.0403, ROC-AUC 0.5791). Both choose C=10 by Validation AP and have Validation-only recall-target threshold files. They use complete available text, not the transformer's 512-token construction, so they are not strictly input-identical comparisons and have no saved final-Test result. Sources: `supplementary_202610/tfidf_logreg/*/{config.json,validation_metrics.json,threshold_calibration.json}`.

The classifier-head-only row in Section 4 is the appropriate PubMedBERT within-family baseline. A separate Qwen3.8-Flash complete-Test experiment exists, but is a hard categorical zero-shot system with retained unresolved outputs—not a conventional scored baseline—and is reported separately in audit notes.

## 6. Validation-based threshold calibration

The selected CLS-LoRA epoch-4 checkpoint was calibrated on Validation predictions only. Rule: select the largest threshold with recall at least the target. The selected target was 0.95, yielding threshold **0.016082723**, TP=394, FP=15,632, TN=6,351, FN=20, recall 95.17%, FNR 4.83%, precision 2.46%, F1 4.79%, and workload reduction 28.45%. Alternative recorded targets 0.90 and 0.98 are in `threshold_comparison.csv`. Source: `results/formal_training_strategy_20260910/thresholds/`.

## 7. Frozen held-out Test evaluation

The final system is CLS-LoRA (adapter LR 5e-5; epoch 4) with the Validation-frozen threshold 0.016082723. The Test set has 23,592 records: 472 positive and 23,120 negative.

| Result | TP | FP | TN | FN | Recall | FNR | Precision | F1 | AP | ROC-AUC | Specificity | Workload reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Overall | 470 | 17,978 | 5,142 | 2 | 99.58% | 0.42% | 2.55% | 4.97% | 0.1553 | 0.9063 | 22.24% | 21.80% |

Per-review values and row meanings are in `final_test_summary.csv`; original sources are `outputs/final_test/final_test_metrics.json`, `per_review_test_metrics.csv`, and `final_test_config.json`. The two stored false-negative cases are both van_Dis_2020, title-present and abstract-missing; their probabilities were 0.0057630353 and 0.015676498. Source: `results/final_error_analysis/false_negative_cases.csv`.

## 8. Article-overlap sensitivity analysis (post-hoc)

The saved post-hoc analysis normalised and matched DOI, PMID and OpenAlex identifiers against Train/Validation. It classified 2,012 Test records (40 positives) as confirmed overlap and 21,580 (432 positives) as confirmed non-overlap. Using the original frozen scores and unchanged threshold, the strict article-disjoint subset produced TP=430, FP=16,255, TN=4,893, FN=2, recall 99.54%, FNR 0.46%, precision 2.58%, F1 5.02%, AP 0.1692, ROC-AUC 0.9122 and workload reduction 22.68%.

This is a sensitivity analysis, not a new independent Test: it is post-hoc, uses exact identifiers only, and cannot exclude unidentified or semantic duplication. Sources: `results/article_overlap_sensitivity/{overlap_summary.json,article_disjoint_metrics.json,primary_vs_article_disjoint.csv}`.

## 9. Limitations and consistency audit

The selected result has one final-Test seed and one fixed review-level split; it does not support statistical-significance claims. Formal strategy runs use saved copied artifacts while some manifests point to unavailable Linux paths. `validation_pr_auc` versus the prose label “AP” requires implementation confirmation. Test overlap exists despite review-level separation, and the strict sensitivity analysis is post-hoc. Traditional baselines have different input exposure. Workload reduction is a predicted exclusion fraction rather than observed human-time savings. PubMedBERT is an encoder classifier, not a generative LLM. See `experiment_audit_notes.md` for the complete evidence boundaries and manual checks.

## Evidence index

Primary data/split: `config/canonical_split.json`; `data/processed/splits/split_summary.csv`; QC: `reports/processed_data_summary.json`. Pooling: `outputs/pooling_comparison/formal/20260905_122936/`. Formal strategies/calibration: `results/formal_training_strategy_20260910/`. Final Test: `outputs/final_test/`. Error analysis: `results/final_error_analysis/`. Overlap sensitivity: `results/article_overlap_sensitivity/`. All tables in this report are reconciled with the companion CSV files in this directory.
