# Test-subset preflight

- Records: 5000 unique records from the frozen formal Test split.
- Sampling: hierarchical largest-remainder allocation; seed 42.
- PubMedBERT historical predictions: all 5000 records matched by stable ID; frozen threshold `0.016082723` reproduced every stored label.
- Qwen: Full raw fields, identical validated Full-Validation prompt/API configuration; no label is placed in the prompt.
- Criteria: all sampled reviews matched; unusable criteria were rejected by the shared join function.
- Missing title: 4; missing abstract: 533.
- Conservative context check: max checked UTF-8 bytes = 17188; configured safety limit = 900000.
- Planned Qwen API requests: 5,000. No API request has been made by this preparation script.
