# Audit issues and evidence boundaries

## Missing or incomplete artifacts

1. Several stored paths in historical manifests are Linux absolute paths that do not exist locally (for example the original training-comparison output path). The copied `results/formal_training_strategy_20260910` metrics/configuration artifacts are present and were used.
2. No final-Test classical baseline, mean-LoRA Test, additional-seed Test, or unweighted-BCE Test artifact exists. Do not imply such evaluations.
3. The Qwen final artifact is aggregate/per-review evidence; raw per-record final-Test responses are deliberately redacted from that release. It supports aggregate metrics but limits a new paired error analysis.
4. The precise execution date is Unknown for overlap sensitivity and PubMedBERT error analysis; directory timestamps are not treated as experiment dates.

## Conflicting or terminology-sensitive metrics

1. Pooling/training artifacts call the selection value `validation_pr_auc`, while thesis prose calls it AP. The audit treats this as the recorded selection metric and does not claim they are distinct calculations; code/metric implementation should be cited if a formal definition is required.
2. Qwen full-Test end-to-end and valid-response-only metrics use different denominators (23,592 vs 23,487). The former is primary because failures are retained by policy.
3. The 5,000-record subset's positive prevalence (2%) matches the full Test approximately by design but label-stratified sampling means it is still not an independent population Test. Its subset metrics must not replace full-Test metrics.

## Data leakage, Test reuse, and selection-bias risks

1. The primary PubMedBERT checkpoint and threshold have explicit Validation-only selection evidence; no direct Test-selection conflict was found.
2. The original Test and Qwen final-Test compare different methods on the same sealed population. They are valid descriptive results but must not be used for iterative selection of prompts/models/thresholds.
3. The precision-matched threshold .85556966 is selected using Test-subset gold labels. This is threshold-selection bias and makes that row strictly post-hoc/in-sample.
4. Article-overlap sensitivity is post-hoc, operates on frozen predictions, and only captures exact normalized DOI/PMID/OpenAlex matches; unresolved identifiers or semantic duplicates remain possible.

## Reproducibility and methodological limits

1. Formal selected Test evidence remains single-seed. Additional Validation seeds demonstrate variation, not a multi-seed Test estimate.
2. Traditional baselines consume full available text; transformer/Qwen inputs and policies are not identical in every respect.
3. Qwen has hard categorical outputs, no ranking score, and retains uncertain/failed outputs. AP and ROC-AUC are therefore N/A, not zero.
4. The primary Test has four reviews and 472 positives. PubMedBERT final artifacts provide no confidence intervals, whereas Qwen artifacts include Wilson intervals.
5. Workload reduction is an output-derived proportion, not measured screening time, cost, safety, or deployment effectiveness.

## Manual verification items

1. Confirm the exact AP implementation behind `validation_pr_auc` before final terminology is fixed.
2. Review the Qwen prompt template and parser source before quoting its full wording in a thesis appendix.
3. Decide whether adding Qwen changes the stated research scope; if yes, update abstract/RQs/method/results/discussion coherently.
4. If confidence intervals for PubMedBERT are desired, calculate them from stored predictions with the frozen threshold and label this as a read-only analysis.
