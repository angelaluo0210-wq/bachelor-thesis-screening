# Pre-AutoDL formal experiment runbook

`config/canonical_split.json` is the only canonical split declaration.  Its
source of truth is `src/split_dataset.py:SPLIT_REVIEWS`; the active article
files are `data/processed/splits/train.csv`, `validation.csv`, and `test.csv`.
`reports/proposed_review_split.csv` is historical and must not be used.

The formal strategy comparison always trains a fresh head-only baseline.  It
uses only train and validation files, locks CLS pooling and threshold 0.5 for
descriptive metrics, and selects checkpoints by validation PR-AUC.  Do not add
`--evaluate-test` to the comparison command.

Before preflight, upload the repository excluding `.venv`, existing `outputs/`,
and raw archival data unless it is separately needed. Upload the local
PubMedBERT snapshot to `/workspace/models/pubmedbert`; it must include
`config.json`, `pytorch_model.bin` or `model.safetensors`, `tokenizer_config.json`,
and `vocab.txt`.

From the repository root on AutoDL:

```bash
python -c "import torch, transformers, peft, accelerate; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(transformers.__version__, peft.__version__, accelerate.__version__); print(torch.cuda.get_device_name(0))"
python -m pip check
python -m pytest -q
python scripts/create_experiment_manifest.py --model-path /workspace/models/pubmedbert --output outputs/formal_experiment_manifest.json
python scripts/create_experiment_manifest.py --model-path /workspace/models/pubmedbert --verify outputs/formal_experiment_manifest.json
```

Run the three disposable GPU preflights independently:

```bash
python scripts/gpu_preflight.py --comparison-mode --training-strategy classifier_head_only --pooling cls --batch-size 32 --max-length 512 --model-name-or-path /workspace/models/pubmedbert --output-dir outputs/gpu_preflight
python scripts/gpu_preflight.py --comparison-mode --training-strategy partial_unfreezing --pooling cls --batch-size 32 --max-length 512 --model-name-or-path /workspace/models/pubmedbert --output-dir outputs/gpu_preflight
python scripts/gpu_preflight.py --comparison-mode --training-strategy lora --pooling cls --batch-size 32 --max-length 512 --model-name-or-path /workspace/models/pubmedbert --output-dir outputs/gpu_preflight
```

The following single command performs the fresh baseline and all four
adaptation-LR trials in separate run directories:

```bash
nohup python scripts/run_training_strategy_comparison.py --execute --train-baseline --model-name-or-path /workspace/models/pubmedbert --output-root outputs/training_strategy_comparison > outputs/training_strategy_comparison.nohup.log 2>&1 &
```

Monitor `outputs/training_strategy_comparison.nohup.log`, the newest
`comparison_manifest.json`, and each child `run.log`. Confirm CUDA, strategy,
trainable-parameter count, optimizer LR groups, finite train/validation loss,
validation PR-AUC, `test_evaluated: false`, and unique output directories.

After the researcher chooses a recall target, calibrate on one selected
validation-prediction artifact only, for example:

```bash
python scripts/calibrate_validation_threshold.py --validation-predictions outputs/training_strategy_comparison/<batch>/<strategy>/<trial>/<strategy>/seed_42/cls/<run>/validation_predictions.csv --target-recall <TARGET> --output outputs/selected_validation_threshold.json
```

This calibration command is intentionally not part of formal strategy training
and cannot be used with test predictions.
