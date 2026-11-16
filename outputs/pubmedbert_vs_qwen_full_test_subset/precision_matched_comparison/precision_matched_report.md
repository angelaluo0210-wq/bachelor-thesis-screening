# Precision-Matched Comparison: PubMedBERT + LoRA vs Qwen Full

## Purpose

This post-hoc exploratory analysis compares the frozen PubMedBERT + CLS + LoRA predictions with Qwen Full on the same fixed 5,000-record Test subset. PubMedBERT thresholds induced by its stored probabilities were scanned to find the operating point whose precision is closest to Qwen Full's observed precision.

## Data and alignment

The fixed subset contains 5,000 records: 100 positive and 4,900 negative. The manifest, PubMedBERT predictions, and Qwen predictions have identical unique record IDs and identical gold labels. Stored PubMedBERT labels are reproduced by the original frozen threshold `0.016082723`.

## Threshold selection

Qwen Full precision is 12.082262%. The closest effective PubMedBERT threshold is `0.85556966` (rule: probability ≥ threshold), yielding precision 12.082671%. The signed matching error is +0.000409% (0.000409% in absolute value). Selection minimized absolute precision difference across observed-score thresholds; ties were broken by higher recall, then by a larger threshold.

## Comparison

| System | Precision | Recall | FN | F1 | Workload reduction | Manual screening |
|---|---:|---:|---:|---:|---:|---:|
| Qwen Full | 12.08% | 47.00% | 53 | 19.22% | 92.22% | 389 |
| PubMedBERT original frozen threshold | 2.59% | 100.00% | 0 | 5.05% | 22.82% | 3,859 |
| PubMedBERT precision-matched threshold | 12.08% | 76.00% | 24 | 20.85% | 87.42% | 629 |

## Methodological limitation

The precision-matched threshold was selected with gold labels from this Test subset. It therefore constitutes an exploratory, optimistic in-sample operating-point analysis, not a pre-registered or independently validated Test result. It must not replace the original Validation-selected frozen threshold in the primary evaluation. The Qwen output and the stored model probabilities were not changed, and no model/API inference was performed.
