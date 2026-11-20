# Qwen Validation input comparison (5,000 paired records)

This report evaluates existing Qwen predictions only. Both runs use the same 5,000 Validation records, Prompt `qwen-zero-shot-screening-v2`, model ID `qwen3.8-flash`, temperature 0, disabled thinking, JSON-object response format, `max_tokens=128`, 30 requests per minute, and the same retry and decision-mapping code. Every stored input SHA-256 matches a prompt rebuilt from the frozen Validation data and local PubMedBERT tokenizer. No Qwen API call or model inference was made for this report.

The sample contains 300 included and 4,700 excluded records. It is label-stratified, so precision and workload reduction below describe this sample, not the complete Validation population.

## Overall results

| Input | N | TP | FP | TN | FN | Recall | Precision | FNR | Workload reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Truncated | 5,000 | 113 | 126 | 4,574 | 187 | 37.67% | 47.28% | 62.33% | 95.22% |
| Full | 5,000 | 103 | 86 | 4,614 | 197 | 34.33% | 54.50% | 65.67% | 96.22% |

Relative to Truncated, Full has 10 more false negatives, 3.33 percentage points lower recall, 7.22 points higher precision, and 1.00 point higher workload reduction. The final retain/exclude decision changes on 82 records: 16 change from exclude to retain and 66 from retain to exclude. Among positive records, Full repairs 4 false negatives and adds 14.

## By actual truncation

| Group | Input | N | Positives | TP | FP | TN | FN | Recall | Precision | FNR | Workload reduction | Final decisions changed in pair |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| At least one field truncated | Truncated | 1,612 | 105 | 15 | 15 | 1,492 | 90 | 14.29% | 50.00% | 85.71% | 98.14% | 19 |
| At least one field truncated | Full | 1,612 | 105 | 12 | 7 | 1,500 | 93 | 11.43% | 63.16% | 88.57% | 98.82% | 19 |
| No field truncated | Truncated | 3,388 | 195 | 98 | 111 | 3,082 | 97 | 50.26% | 46.89% | 49.74% | 93.83% | 63 |
| No field truncated | Full | 3,388 | 195 | 91 | 79 | 3,114 | 104 | 46.67% | 53.53% | 53.33% | 94.98% | 63 |

In the 1,612 records with actual truncation, Full has 3 more false negatives; in the 3,388 records without truncation, it has 7 more. The two strata differ in review and label composition, so their raw rates should not be interpreted as a causal estimate of truncation.

The stored truncation audit was independently reproduced: Title 3, Abstract 1,611, Inclusion Criteria 0, Exclusion Criteria 0; 1,612 records have any truncation and 2 have multiple fields truncated. The 512-token limit is the total PubMedBERT input budget, not Qwen's output limit.

## Output quality and interpretation

Both runs completed all 5,000 records. Full returned 121 `include`, 4,811 `exclude`, and 68 `uncertain`; Truncated returned 127 `include`, 4,761 `exclude`, and 112 `uncertain`. Both have zero final API failures, malformed outputs, `finish_reason=length` responses, and responses reaching 128 output tokens. All `uncertain` decisions are retained for manual screening under `retain_unresolved_v1` and are included in the metrics.

Full inserts raw Title, Abstract, Inclusion Criteria, and Exclusion Criteria into the Prompt. Truncated inserts PubMedBERT WordPiece-decoded section prefixes, which include section markers and may normalize text. All 5,000 paired prompt hashes differ, including the 3,388 records with no actual token truncation. The results therefore compare two existing input-construction pipelines; the performance difference cannot be attributed solely to input length. The runs occurred hours apart and the records do not contain an immutable provider model revision.

The preparation configuration still says `prepared_not_executed`; the two complete prediction files and their per-record completion timestamps establish that execution occurred. Completeness was checked directly from the JSONL files. Original API response bodies and per-attempt retry logs are not available. None of these record limitations changes the verified row counts or computed confusion matrices.

## Files

- `paired_predictions.csv`: one row per Validation record with both decisions, final retained labels, input hashes, and field-truncation data.
- `metrics_summary.csv` and `metrics_summary.json`: overall and truncation-stratified confusion matrices and metrics.
- `truncation_summary.json`: verified original truncation summary.
- `source_verification.json`: source paths, SHA-256 hashes, configuration, alignment, prompt-hash, and output-quality checks.

Reproduction command from the repository root (requires a new output path because the evaluator refuses overwriting):

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_qwen_input_length_sensitivity.py --output-dir results\qwen_input_length_sensitivity_recheck
```
