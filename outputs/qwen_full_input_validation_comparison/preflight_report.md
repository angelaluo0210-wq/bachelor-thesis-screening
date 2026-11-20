# Qwen full-input Validation comparison: preflight

Prepared only: no API request or prediction was made.

- Sample: 5000 unique Validation rows; 300 include and 4700 exclude.
- Modes share the exact frozen prompt fingerprint `92d5ec3e57208464fdfafece65202a68d3363b790a64a9821cdfdc335317625a`, model ID, and inference settings. The sole configured API-parameter difference from the legacy Test is `max_tokens=128`, applied equally to both modes to bound the JSON response.
- Provider verification, checked 2026-11-12: the actual returned model in the completed run was `qwen3.8-flash`; its official Alibaba Cloud Model Studio limits are a 1,000,000-token context, 991,808-token maximum input, and 131,072-token maximum output. The largest prepared Full prompt is 25,118 characters, far below the documented input ceiling even under a deliberately conservative UTF-8-byte bound. 
- No published tokenizer was verified as identical to the managed `qwen3.8-flash` deployment. The executor therefore uses a conservative UTF-8-byte bound: prompt UTF-8 bytes plus a 4,096-byte chat-envelope allowance must not exceed 900,000 bytes. This is a safety bound, not a token count; API `usage` is recorded as the billing source of truth. It refuses an over-limit request and never truncates text.
- Before batch execution, invoke each mode with `--technical-validation-records N --execute`. These isolated output directories stop on malformed JSON, an invalid label, `finish_reason="length"`, or an output that reaches 128 completion tokens.
- Estimated usage is calibrated from the previous same-model run and recorded in `experiment_config.json`; API-returned `usage` remains the billing source of truth.
- API requests planned after authorization: 10,000 (5,000 records × 2 modes); retries are additional attempts, not new record-level requests.
