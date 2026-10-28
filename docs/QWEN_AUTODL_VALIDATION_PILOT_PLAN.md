# Qwen3.8-Flash AutoDL Validation Pilot plan

This is a small API and program-validation Pilot, not a model-selection or
scientific efficacy experiment. It must never use Test data. The final baseline
will later evaluate the complete four-review Test set; Test subset sampling is
prohibited.

## Preconditions

1. Use the reviewed code with the frozen `qwen-zero-shot-screening-v2` prompt
   and retain/exclude policy.
2. Provide local Train and Validation CSVs, criteria workbook, and the exact
   local PubMedBERT tokenizer directory. Do not download a model from the Hub.
3. Set `DASHSCOPE_API_KEY` securely in the AutoDL environment. Do not write it
   to files, shell history, manifests, or Git configuration.
4. Confirm approved HTTPS access to the configured provider endpoint and retain
   TLS verification.

## No-network preflight

The default command loads only Train/Validation-era inputs. It deterministically
selects 25 Validation records using SHA-256 of the frozen seed and stable record
ID, never labels, and audits every selected reconstruction against the actual
PubMedBERT section-budget path.

```bash
python scripts/run_qwen_baseline.py \
  --model-name-or-path /workspace/models/pubmedbert \
  --max-records 25 \
  --sample-seed 202610 \
  --output-dir supplementary_202610/llm_baseline/validation_plan
```

Review the plan before any API call: prompt hash, API parameters, Git/runtime
metadata, tokenizer hashes, input hashes, selected record IDs, and the complete
reconstruction audit. Any token-ID disagreement stops the Pilot. Decoded-field
round-trip failures require documentation and researcher review before inference.

## Authorized API Pilot

After explicit authorization, create a new immutable directory and run:

```bash
python scripts/run_qwen_baseline.py \
  --execute \
  --model-name-or-path /workspace/models/pubmedbert \
  --split-dir data/processed/splits \
  --criteria-path 'data/eligibility_criteria(排纳标准).xlsx' \
  --max-records 25 \
  --sample-seed 202610 \
  --output-dir supplementary_202610/llm_baseline/validation_pilot_YYYYMMDD
```

Check provider compatibility with the exact model name, JSON-object format,
`enable_thinking=false`, `stream=false`, `temperature=0`, response content,
returned model identity, provider request ID, usage fields, malformed-response
behavior, rate limits, retry behavior, elapsed runtime, and token usage.

The Pilot reports explicit include, explicit exclude, uncertain, malformed
output, and API/transport failure separately, together with valid-response
coverage, uncertain rate, technical-failure rate, elapsed time, and token usage.
It must not be used to iterate on Recall, FNR, or another label-based metric. If
the API reveals an incompatibility that requires a prompt or policy change, stop
and obtain researcher approval; do not silently modify the frozen prompt.

## Future full-Test comparison design only

Do not execute this stage until settings are frozen and separately authorized.
The future Qwen run must process every article in the complete sealed Test set
for all four Test reviews. It may not sample, downsize, or select a subset.
Before it reads Test labels or frozen prediction rows, verify the predeclared
full-population rule. Before comparison, validate exact one-to-one alignment of
Qwen and PubMedBERT records, review membership, and gold labels. Report both
confusion matrices and per-record paired prediction differences.
