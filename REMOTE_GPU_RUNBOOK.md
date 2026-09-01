# Remote Windows GPU experiment runbook

This package supports two formal stages with a fully frozen PubMedBERT
backbone:

- Stage 1: CLS classifier-head-only learning-rate tuning (`1e-4`, `5e-4`, `1e-3`).
- Stage 2: one manually selected shared LR, compared across `cls`, `mean`,
  `attention`, and `concat` pooling.

Run every command from `C:\thesis\project`. Neither runner passes
`--evaluate-test`; the package intentionally contains no `test.csv`.

## STEP 1 — Activate the environment and verify CUDA

PowerShell:

```powershell
Set-Location C:\thesis\project
C:\thesis\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

`requirements.txt` intentionally excludes PyTorch so it does not replace the
existing CUDA build.

## STEP 2 — Verify the local PubMedBERT model/tokenizer

The local directory must contain `config.json`, `pytorch_model.bin`,
`tokenizer_config.json`, and `vocab.txt`.

```powershell
python -c "from transformers import AutoModel, AutoTokenizer; p=r'C:\thesis\models\pubmedbert'; t=AutoTokenizer.from_pretrained(p, local_files_only=True); m=AutoModel.from_pretrained(p, local_files_only=True); print(type(t).__name__, type(m).__name__, m.config.hidden_size)"
```

Expected checkpoint identity:
`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`.

## STEP 3 — Stage 1 GPU preflight

This performs one real batch forward/backward only; it does not run an epoch or
open `test.csv`.

```powershell
python scripts\gpu_preflight.py --pooling cls --batch-size 32 --model-name-or-path C:\thesis\models\pubmedbert
```

If this reports a real CUDA OOM, repeat with 16 and use batch 16 for all Stage 1
runs and all later Stage 2 runs.

## STEP 4 — Run CLS LR tuning

```powershell
.\run_lr_tuning.bat --batch-size 32 --output-root C:\thesis\results\hyperparameter_tuning\cls --model-name-or-path C:\thesis\models\pubmedbert
```

The runner executes `1e-4`, `5e-4`, and `1e-3` sequentially, stops on failure,
and never selects a winner.

## STEP 5 — Review all three validation trajectories

For every LR, inspect:

- `training_history.csv` for every epoch and every validation metric;
- `run_summary.json`;
- `config.json` and `run.log`;
- both best-F1 and best-Recall checkpoint metadata.

Do not select an LR from a dry-run or smoke test.

## STEP 6 — Manually select the shared LR

IMPORTANT: no script automatically selects the LR from one metric. Retain the
complete validation results and apply the model-selection protocol defined for
the thesis. Record the resulting shared LR before Stage 2.

## STEP 7 — Preflight all four pooling strategies

Run all four at the same candidate batch size:

```powershell
python scripts\gpu_preflight.py --pooling cls --batch-size 32 --model-name-or-path C:\thesis\models\pubmedbert
python scripts\gpu_preflight.py --pooling mean --batch-size 32 --model-name-or-path C:\thesis\models\pubmedbert
python scripts\gpu_preflight.py --pooling attention --batch-size 32 --model-name-or-path C:\thesis\models\pubmedbert
python scripts\gpu_preflight.py --pooling concat --batch-size 32 --model-name-or-path C:\thesis\models\pubmedbert
```

If any one of the four has a real batch-32 CUDA OOM, Stage 2 must use batch 16
for all four. Never mix batch sizes across pooling strategies. If Stage 1 was
already reduced to 16, use 16 here from the start.

## STEP 8 — Run the four-pooling comparison

Replace `5e-4` below with the LR selected manually in STEP 6:

```powershell
.\run_pooling_comparison.bat --learning-rate 5e-4 --batch-size 32 --output-root C:\thesis\results\pooling_comparison --model-name-or-path C:\thesis\models\pubmedbert
```

The runner executes `cls`, `mean`, `attention`, and `concat` sequentially. It
uses identical data, model, LR, batch size, epochs, dropout, seed, max length,
threshold, optimizer, loss, pos_weight calculation, truncation, and validation
protocol. Only pooling and its necessary classifier-side parameterization vary.
It stops on failure and does not select a winning pooling strategy.

Expected Stage 2 layout:

```text
C:\thesis\results\pooling_comparison\
├── pooling_comparison_manifest.json
├── cls\run_<timestamp>\
├── mean\run_<timestamp>\
├── attention\run_<timestamp>\
└── concat\run_<timestamp>\
```

Every completed run contains `config.json`, `training_history.csv`,
`run_summary.json`, `run.log`, `best_f1_classifier_head.pt`,
`best_recall_classifier_head.pt`, and `validation_predictions.csv`.
The first comparison writes `pooling_comparison_manifest.json`; a repeated
comparison in the same root writes a timestamp-suffixed manifest instead of
overwriting the earlier one.

## STEP 9 — Back up results

Copy the complete Stage 1 and Stage 2 result roots back to the local computer.
At minimum retain all manifests, configs, histories, summaries, logs,
checkpoint files and validation predictions. The head-only checkpoints are
small, so both best-F1 and best-Recall versions should be backed up.

## Failure recovery

There is no optimizer-state resume mechanism. Timestamped run directories are
never overwritten. Retain an interrupted directory for diagnosis and rerun the
failed configuration; do not silently continue with a different batch size or
other shared setting. Both runners stop immediately when a child run fails.
