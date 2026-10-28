# Supplementary AutoDL pre-deployment audit

Static audit date: 2026-10-28. No Test file was opened and no GPU run was started.

Protected-file SHA-256 baseline captured during this audit:

- `config/canonical_split.json`: `147BFD3427A73CD42FB0663A79C97725C02BB2B688ADCC916C0B007356DA82FD`
- `results/formal_training_strategy_20260910/lora_5e-5/run_summary.json`: `2765E6AB8E8B2BB689D07BB42F003BEF57E82D9D2054667F6F030DCC282102C2`
- `results/formal_training_strategy_20260910/lora_5e-5/config.json`: `7D934D0FB2A6D0461BB2B3FFC81CE92356ADAA45A4B6F8B2075DF026D28A67B7`
- `supplementary_202610/tfidf_logreg/validation_summary.csv`: `F79EF96268BF573F3E75A1B60260B112EEF2ED9C0D88EB4D69A6540E047D05AD`

## Confirmed statically

- Canonical split declares 18 Train, 4 Validation and 4 sealed Test reviews. Supplementary runners pass no Test option.
- The historical CLS+LoRA protocol is preserved for weighted robustness runs: CLS, r=8, alpha=16, dropout=0.05, query/value, head LR 1e-4, adapter LR 5e-5, batch 32, five epochs, length 512, seed-specific initialization, weighted BCE, and validation PR-AUC checkpoint selection.
- Mean+LoRA calls the existing `_masked_mean`: it includes special tokens and excludes padding via `attention_mask`.
- The new `class_weighting=weighted` default retains `negative/positive` Train-only `pos_weight`; `unweighted` constructs BCE without `pos_weight`. Comparison mode now rejects unweighted BCE and remains CLS/0.5/Test-disabled.
- Outputs are timestamped and seed-separated. Smoke invocations use independent temporary roots.
- The orchestration gate accepts only a `validation_manifest.json` with `status=passed`. Its safe resume mode skips one fully verified completed group and refuses ambiguous or partial artifact trees; it never attempts optimizer/RNG restoration.
- `scripts/create_experiment_manifest.py` is intentionally excluded because it reads `test.csv`; it is not part of any AutoDL supplementary command.
- The read-only result collector searches only `mean_lora`, `cls_lora_multiseed`, and `cls_lora_unweighted_bce`; it excludes smoke trees and rejects any `run_summary.json` which is not explicitly `test_evaluated: false`.

## Dynamic checks still required on AutoDL

Install the recorded compatible environment, then run `scripts/setup_autodl.sh` and `scripts/validate_autodl.sh`. Dynamic validation must establish adapter updates, frozen non-LoRA parameters, checkpoint reload equivalence, CUDA capacity, and all smoke branches before formal training.

## Environment evidence

Historical formal LoRA config recorded torch `2.6.0+cu124`, transformers `5.16.1`, peft `0.20.0`, accelerate `1.15.0`, numpy `2.1.3`, and scikit-learn `1.9.0` on RTX 4080 SUPER. Exact historical Python, pandas, openpyxl, tqdm and tokenizers versions were not recorded; the setup script logs them for review. The overlay never installs or changes torch, and `pip check` is mandatory after installing the non-torch overlay. Recommended minimum GPU memory is 16 GB; 24 GB is preferred to leave headroom for PubMedBERT LoRA at batch 32/length 512. Disk: reserve at least 30 GB for repository, model, environments, logs and six artifact trees.
