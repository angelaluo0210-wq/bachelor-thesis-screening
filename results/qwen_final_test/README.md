# Qwen Final Test Archive

This directory contains redacted, aggregate-only artifacts from the completed Qwen final Test evaluation.

## Frozen experiment

- Model: `qwen3.8-flash`
- Prompt: `qwen-zero-shot-screening-v2` (`92d5ec3e57208464fdfafece65202a68d3363b790a64a9821cdfdc335317625a`)
- API parameters: temperature 0, thinking disabled, non-streaming JSON-object response, serial 30 RPM, at most three retries.
- Input construction: PubMedBERT tokenizer identity recorded in `run_configuration.json`; max length 512.
- Population: all 23,592 records from four frozen Test reviews (`Leenaars_2019`, `Muthu_2021`, `Radjenovic_2013`, and `van_Dis_2020`); no sampling.
- Screening policy: `retain_unresolved_v1`. Explicit `exclude` records are automatically excluded. `include`, `uncertain`, malformed outputs, and transport failures remain retained for human screening.

## Outputs

- `qwen_final_test_metrics.json`: aggregate end-to-end and supplementary-valid-response metrics, including micro, macro, per-review, response-quality, and available Wilson confidence intervals.
- `per_review_qwen_metrics.csv`: one aggregate metric row per Test review.
- `execution_summary.json`: completion, outcome-count, token-usage, and elapsed-time summary.
- `run_configuration.json`: redacted frozen configuration, input hashes, tokenizer identity hash, and source commit.

## Completion and recovery

The run completed all 23,592 records with no missing, extra, or duplicate IDs. An initial Windows `PermissionError` while atomically replacing the progress snapshot interrupted the run after 11,950 durable append-only prediction records. The runner was repaired to use unique temporary names and bounded retry/backoff for `PermissionError`; the identical run then resumed from the same output directory without repeating completed requests. The final evaluation gate verified complete Qwen--gold alignment before formal metrics were generated.

## Deliberate exclusions

This archive contains no API key, endpoint/workspace URL, absolute local path, raw Test data, per-record prediction log, full API request/response log, model weight, or checkpoint. `predictions.jsonl` remains local and is intentionally not versioned.

## Interpretation

The primary metrics evaluate the full Test population under the frozen retention policy. Qwen emits hard decisions only, so PR-AUC and ROC-AUC are not reported for Qwen. Any PubMedBERT comparison uses results from its previously completed, independent final Test evaluation; no per-record paired analysis was performed here.
