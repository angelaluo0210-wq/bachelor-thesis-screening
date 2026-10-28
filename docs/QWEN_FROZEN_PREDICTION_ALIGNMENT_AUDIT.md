# Frozen PubMedBERT prediction alignment audit

## Verified from code and artifact schema only

`src/train_classifier_head.py:prediction_frame` writes a row per evaluated
source row using its evaluation indices. Its schema is `review_name`, `title`,
any available `pmid`, `doi`, and `openalex_id`, followed by `true_label`,
`predicted_probability`, and `predicted_label`. The probability is produced by
applying sigmoid to the model's logits in `evaluate`; it is not a logit.
Binary predictions use `probability >= threshold`.

The committed final-release directory is expected to contain
`final_test_predictions.csv`, per-review metrics, metrics JSON, and config
JSON. Only the CSV header and JSON key sets were inspected in Phase 1; no Test
prediction row, title, label, score, or Test example was read. The header is
consistent with the writer above.

`scripts/evaluate_final_test.py` locks the PubMedBERT selection evidence:
LoRA, CLS pooling, seed 42, validation-selected checkpoint, and a threshold
loaded from a validation-only threshold configuration. It rejects changes to
those release invariants before it opens Test data. Overall and per-review
metrics use `train_classifier_head.calculate_metrics` and the frozen threshold.

## Future joining rule

`src/llm_baseline/alignment.py` provides a non-destructive validator. It joins
only on normalized review key plus available stable article identifiers
(`pmid`, `doi`, `openalex_id`). Titles and row order are never join keys. This
allows an article identifier to occur in different reviews while detecting a
duplicate within the review-plus-identifier key. Missing identifiers, duplicate
keys, missing keys, and extra keys are reported; nothing is deduplicated,
reordered, discarded, or repaired.

For the Qwen run, `stable_record_id` uses normalized review plus identifiers,
falling back to the source-row index only where identifiers are absent. The
fallback is safe for resume within one immutable input file but is insufficient
to establish cross-artifact alignment. The run manifest records input hashes.

## NOT VERIFIED

Actual Test prediction alignment is **NOT VERIFIED**. The real Test data and
prediction rows were deliberately not read. The later AutoDL procedure must
validate identifier completeness, uniqueness, equal key sets, label equality,
and exact population counts before any comparative metric is computed. If an
identifier is missing or ambiguous, stop rather than fall back to title or row
order; retain both artifacts and resolve the source data manually.

The future comparison is full-population only: Qwen must process every article
in all four sealed Test reviews, and PubMedBERT must be compared on that same
complete record set. Test subset sampling, reduced Test populations, and
article-level selection are prohibited.
