# Qwen baseline Phase 1 test report

This report covers offline development only. All fixtures introduced for the
Qwen baseline are explicitly synthetic. No Qwen API call, external model
download, GPU run, real Train/Validation inference, or Test access occurred.

## Focused checks

```bash
PYTHONPATH=. python -m pytest -q tests/test_llm_baseline.py --tb=short
PYTHONPATH=. python -m compileall -q src/llm_baseline scripts/run_qwen_baseline.py
```

The focused suite verifies synthetic reconstruction field order and caps,
512-token boundaries, missing abstract, empty criteria, long fields, Unicode
and punctuation, deterministic reconstruction, prompt serialization, strict
parsing, no-key rejection, mocked transport retry, configuration fingerprints,
append-only resume storage, duplicate detection, alignment validation, and hard
decision evaluation. It does not demonstrate real-corpus reconstruction or
real Test alignment.

## Repository suite

Run with offline Hub mode:

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 python -m pytest -q --tb=short
```

The final run completed with 93 passing tests and three expected missing-input
failures: both tests in `tests/test_criteria_matching.py` require the excluded
criteria workbook and Train/Validation splits; one TF-IDF supplementary test
requires the excluded Train/Validation splits. These failures are unchanged
environment-data limitations and were not bypassed with fabricated files.
