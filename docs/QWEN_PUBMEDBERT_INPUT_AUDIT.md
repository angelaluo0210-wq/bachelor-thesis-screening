# PubMedBERT input pipeline audit for the Qwen baseline

This is a Phase 1 code audit. It does not open the real Train, Validation, or
sealed Test data, and it does not run a model checkpoint.

## Verified from repository code

The authoritative input implementation is
`src/train_classifier_head.py:tokenize_with_section_budget`. The live training
and final evaluation paths both create loaders through `make_loader`, whose
`BudgetedCollator` invokes this function. The final release entry point
`scripts/evaluate_final_test.py` also uses `trainer.make_loader`; there is no
separate final-Test text construction path.

`src/screening_data.py` supplies the shared data conventions. CSV splits must
contain `review_key`, `title`, `abstract`, and `label_included`. `safe_text`
maps a pandas missing value to the empty string and otherwise applies
`str(value).strip()`. Consequently missing and whitespace-only abstracts reach
the tokenizer as an empty abstract after its field marker. No other text
normalization occurs in the input constructor.

Criteria are loaded from the sole workbook sheet containing `review_key` or
`review_id`. Its column names are normalized by lowercasing, replacing hyphens
with spaces, and joining whitespace with underscores. Review keys are likewise
normalized for joining (case folded; spaces and hyphens become underscores).
`join_criteria` maps selected inclusion/exclusion columns to every article.
If only unified eligibility text exists, it is preserved in the inclusion
field. Empty criteria after that fallback are rejected. Optional review title,
question, description, and systematic-review-title columns are concatenated
into `_review_text`; current repository comments say project data leaves it
empty.

The model input logical order is optional `[REVIEW]`, then `[INCLUSION]`,
`[EXCLUSION]`, `[TITLE]`, and `[ABSTRACT]`. The bracket labels are literal
ordinary text passed to `tokenizer.encode(..., add_special_tokens=False)`;
they are not registered tokenizer special tokens and do not change embeddings.
Sections are joined with literal newlines. Each is prefix-truncated at the
token-ID level.

For `max_length=512`, the two BERT special tokens reserve two positions, so
there are 510 content-token positions. The title gets at most 64 content
tokens. Inclusion (including optional review text and its marker) then gets at
most 128, and exclusion gets at most 128. Abstract gets all remaining capacity
(190 when the three earlier caps are full). If the abstract is shorter, unused
capacity is water-filled across truncated inclusion and exclusion: equal extra
tokens while both need them, one odd token goes to the greater unmet need, then
the remaining unmet field receives leftovers. `[CLS]` is prepended and `[SEP]`
appended after field truncation; the sequence is right-padded to `max_length`.
The maximum includes those two special tokens and padding. Criteria/article
masks are calculated from these retained segment lengths and exclude special
tokens and padding.

`build_input_text` is used only by `report_truncation` to report unbudgeted
lengths. It is not the tensor constructor. Treating its combined string with a
generic 512-token truncation as equivalent would be incorrect.

## Verified by offline unit tests

`tests/test_llm_baseline.py` creates a synthetic local BERT tokenizer and
checks that the reconstruction invokes the production tokenization function,
reassembles its active ID sequence, preserves field ordering, and handles
missing fields, long fields, Unicode/punctuation, and repeated runs. Existing
`tests/test_training_strategies.py` also exercises the real tokenizer/collator
on a synthetic 12-layer local BERT model.

## Reconstruction fidelity limits

`src/llm_baseline/reconstruction.py` uses the original tokenizer and confirms
its reconstructed retained IDs against `tokenize_with_section_budget`. It
decodes each retained field for Qwen; it never sends IDs to Qwen. Decoding can
be lossy relative to raw source characters: WordPiece tokenizers may normalize
whitespace/case and replace unsupported Unicode with `[UNK]`; decoded subwords
may also have different whitespace presentation. The module reports a
per-field re-encoding check. A passing check demonstrates retained token-ID
equivalence, not exact original character equivalence.

## Not verifiable in Phase 1

No real data, local PubMedBERT snapshot, or checkpoint was available for this
task. Actual criteria-column selection, actual retained-token distributions,
and character-level reconstruction on the scientific corpus remain unverified
until the authorized AutoDL Validation Pilot.
