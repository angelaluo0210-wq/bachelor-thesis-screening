# Missing historical evidence

`missing` means an expected original artifact was not present at its recorded repository-relative path when this package was made. `not_recorded` means the historical run did not record that value. Neither marker is reconstructed from the current Git HEAD.

## P0: four nonselected formal training strategies

The formal comparison manifest records all five runs. For the following four nonselected runs, `validation_predictions.csv`, `best_pr_auc_classifier_head.pt`, and `run.log` are missing locally at the indicated original run directory. Their retained `config.json`, `metrics.json`, `training_history.csv`, `run_summary.json`, and per-review metrics remain in `results/formal_training_strategy_20260910/<result directory>/`. The reported AP and other saved metrics can be read, but their row-level scores cannot be independently recalculated from the currently available files.

| Experiment ID | Retained result directory | Recorded original run directory |
| --- | --- | --- |
| STRAT-HEAD | `results/formal_training_strategy_20260910/classifier_head_only/` | `outputs/training_strategy_comparison/20260910_150708_3f3f0eb0/classifier_head_only/head_lr_1e-4/classifier_head_only/seed_42/cls/run_20260910_150806/` |
| STRAT-PARTIAL-1E5 | `results/formal_training_strategy_20260910/partial_unfreezing_1e-5/` | `outputs/training_strategy_comparison/20260910_150708_3f3f0eb0/partial_unfreezing/backbone_lr_1e-5/partial_unfreezing/seed_42/cls/run_20260910_162015/` |
| STRAT-PARTIAL-2E5 | `results/formal_training_strategy_20260910/partial_unfreezing_2e-5/` | `outputs/training_strategy_comparison/20260910_150708_3f3f0eb0/partial_unfreezing/backbone_lr_2e-5/partial_unfreezing/seed_42/cls/run_20260910_174829/` |
| STRAT-LORA-1E4 | `results/formal_training_strategy_20260910/lora_1e-4/` | `outputs/training_strategy_comparison/20260910_150708_3f3f0eb0/lora/adapter_lr_1e-4/lora/seed_42/cls/run_20260910_215345/` |

The selected `STRAT-LORA-5E5` run has its Validation predictions, checkpoint, copied summary and metrics, but its original `run.log` is also missing. This makes its provenance **Partial**, not Missing. The original Pooling run logs are present locally and have sanitized public copies in this package.

## P1: code and environment gaps

- Except for the Qwen formal Test run, exact run-time Git SHAs are `not_recorded`. The current HEAD cannot stand in for a historical execution commit.
- The formal Qwen Test run recorded a dirty worktree at commit `b45968fdd3eefd644083f672c6534216e3776c56`; no historical diff was retained. The Qwen input-sensitivity preparation also recorded a dirty tree at `150c1334e5f1e35a127b246bcdbb24d971c0c7b1`, but its execution SHA/worktree state is `not_recorded`.
- Python and several dependency versions are absent from the per-run Pooling/formal/supplementary training configurations. See `environment_summary.md`; the AutoDL installation log is not an exact freeze for every later run.
- A complete original PubMedBERT final-Test execution log is not present. Its frozen predictions, threshold, checkpoint and metrics remain available.
- The Qwen provider's immutable model revision, raw response bodies for every call, and separate per-attempt retry logs were not retained. The 23,592 final parsed decisions and run summary remain available.
- The newly added Weighted BCE Validation comparison is indexed. A matched Unweighted BCE Test prediction does not yet exist; this requested Test extension remains pending GPU inference and is not represented as a historical result.

## Local-only and Git delivery gaps

- The original Pooling directory and split CSV/XLSX inputs are local and ignored by Git. This package supplies minimized Pooling copies and SHA-256 indexes, but no source data or model weights.
- The two TF-IDF `joblib` model pairs, the locally stored PubMedBERT base weights, and supplementary adapter checkpoints are not Git-tracked. Their local paths and SHA-256 values appear in `checkpoint_sha256.csv`.
- `results/weighted_bce_ablation/` and `results/qwen_input_length_sensitivity/` were local and untracked at packaging time. Their existing metrics and source paths are indexed without recomputation.
- The new `results/reproducibility/` files and root README are untracked until the user reviews and commits them. No `git add`, commit, push or merge was performed.

**Recovery priority:** First look for the four missing strategy run directories in historical storage or backups. If recovered, verify each file hash and update the provenance matrix. Do not recreate historical predictions or logs from current code and label them as original runs.
