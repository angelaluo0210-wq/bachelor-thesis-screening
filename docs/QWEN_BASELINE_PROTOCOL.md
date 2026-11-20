# Qwen zero-shot baseline Phase 2 protocol

## Scope and frozen boundaries

This package is separate from the completed PubMedBERT experiments. It does
not modify PubMedBERT inputs, checkpoints, calibration, frozen predictions,
canonical splits, TF-IDF artifacts, or historical supplementary outputs.
`scripts/run_qwen_baseline.py` is a Validation-Pilot-only command: it has no
Test argument and cannot open a Test file.

The Validation Pilot may use a small deterministic, label-blind sample only to
check program behavior and API compatibility. It is not an efficacy experiment.
The final Qwen baseline must evaluate the **complete sealed Test set covering
all four Test reviews**. Test sampling, a reduced Test subset, or selecting
only some articles is prohibited. The later full-Test Qwen and frozen
PubMedBERT comparison must use the same complete record set, gold labels, and
review set, with one-to-one record alignment before metrics are calculated.

## Frozen prompt and screening policy

The prompt is `qwen-zero-shot-screening-v2`; its SHA-256 covers both template
and version. It contains the PubMedBERT-retained inclusion criteria, exclusion
criteria, title, and abstract. It directs the model to treat field text as data
rather than instructions and permits only `include`, `exclude`, or `uncertain`
in an exact one-key JSON response.

The primary screening outcome is binary:

| Qwen outcome | Screening action | Primary metric value |
| --- | --- | --- |
| Explicit `include` | Retain for manual screening | retained (1) |
| Explicit `exclude` | Automatically exclude | excluded (0) |
| Explicit `uncertain` | Retain for manual screening | retained (1) |
| Malformed output after bounded retry | Retain for manual screening | retained (1) |
| API/transport failure after bounded retry | Retain for manual screening | retained (1) |

No unresolved record is dropped from the primary denominator or converted to an
automatic exclusion. The run and evaluation outputs separately count explicit
include, explicit exclude, uncertain, malformed output, and API/transport
failure. This conservative policy protects recall and FNR against technical
failure, at the expected cost of reduced workload reduction and potentially
lower precision.

## API behavior and reproducibility

The client requests `qwen3.8-flash` from the configured OpenAI-compatible
endpoint with `enable_thinking=false`, `stream=false`, `temperature=0`, and a
JSON-object request. `DASHSCOPE_API_KEY` is read only at request time and is
not written to manifests or prediction artifacts. Transport failures and
malformed JSON each receive at most three retries with bounded exponential
backoff. An explicit `uncertain` response is valid and is not retried.

Every immutable run manifest records API parameters, Qwen model ID, prompt
version/hash, Train/Validation/criteria hashes, local tokenizer file hashes,
Git commit and worktree state, Python and Transformers versions, deterministic
Pilot sample seed and size. Per-record JSONL records include parsing status,
retry count, token usage, returned model identity and provider request ID when
the provider supplies them. Cost remains null until provider pricing and usage
units are verified; it is never estimated from an undocumented rate.

The Pilot performs `reconstruction_audit` for every selected record. It invokes
the production PubMedBERT section-budget tokenizer and records whether each
decoded field re-encodes to the retained token sequence. This confirms token-ID
equivalence, not character-level identity: WordPiece decoding may alter spacing,
casing, punctuation display, or unsupported Unicode representation.

## Evaluation

`evaluate_policy_records` evaluates every input under the retain/exclude policy
and reuses the project's confusion-matrix definitions for Recall, FNR,
Precision, F1, and Workload Reduction. It reports micro aggregate, per-review
metrics, macro aggregate, and Wilson intervals for recall, FNR, precision and
workload reduction. Qwen hard decisions have no calibrated continuous score;
PR-AUC and ROC-AUC are explicitly null and must not be compared with
PubMedBERT's probability-based AP.

The report has three non-interchangeable layers. `response_quality` separates
valid model predictions (`include`, `exclude`, `uncertain`) from technical
failures (malformed output and API/transport failure), and reports valid-response
coverage, uncertain rate, and technical-failure rate. `end_to_end_primary`
reports the complete-population screening metrics after applying the frozen
retain/exclude policy. `supplementary_valid_responses` reports the same policy
only among valid model responses and states its reduced denominator; it is never
the primary result.

Before any future full-Test comparison, the alignment procedure must prove
unique review-plus-identifier keys, equal record sets, equal review sets, and
equal gold labels. It must report both systems' confusion matrices and each
record's paired predictions. No Test access is authorized until all Validation
settings, this prompt, the screening policy, and the full-Test rule are frozen.
