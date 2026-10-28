# Qwen baseline Phase 2 protocol-finalization test report

No Qwen request, Test read, GPU command, formal Validation evaluation, commit,
or merge was performed while preparing Phase 2.

Focused offline verification:

```bash
PYTHONPATH=. python -m pytest -q tests/test_llm_baseline.py --tb=short
PYTHONPATH=. python -m compileall -q src/llm_baseline scripts/run_qwen_baseline.py
```

The focused tests cover three-way prompt parsing, malformed-response retries,
the frozen retain/exclude mapping, retention of uncertain/malformed/transport
outcomes, full Pilot reconstruction auditing on synthetic records, deterministic
label-blind Pilot sampling, run fingerprints, resume protection, and alignment
label mismatches.

Repository verification:

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 python -m pytest -q --tb=short
```

The Phase 2 run passed 97 tests. Three existing tests failed only because this
cloud checkout intentionally lacks the real criteria workbook and Train/
Validation CSVs: two criteria-matching tests and one TF-IDF supplementary test.
No production-like files were fabricated to suppress those failures.
