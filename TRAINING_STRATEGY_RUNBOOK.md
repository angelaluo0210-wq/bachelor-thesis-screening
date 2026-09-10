# Training strategy comparison: implementation and verification

This phase fixes CLS, threshold 0.5 (descriptive metrics only), validation
Average Precision for checkpoint selection, and the existing review splits.
Do not change the data/criteria, tune thresholds, evaluate test, or reselect pooling.

## Locked strategies

| Strategy | Trainable parameters for PubMedBERT base | Learning rates |
|---|---:|---|
| `classifier_head_only` | Head weight + bias: 769 | Head `1e-4` |
| `partial_unfreezing` | Blocks 10/11 + head: 14,176,513 | Head `1e-4`; backbone `1e-5` or `2e-5` |
| `lora` | 24 query/value adapters + head: 295,681 | Head `1e-4`; adapter `5e-5` or `1e-4` |

LoRA uses PEFT, all 12 blocks, r=8, alpha=16, dropout=0.05, bias=none.
Base native dropout stays disabled; only LoRA dropout and head dropout train.
Partial unfreezing enables native dropout only in the upper two blocks.
All strategies use AdamW with weight_decay=0.01 including bias/LayerNorm,
no scheduler, no warmup, no gradient clipping, and the same weighted BCE.
Normal training completes all five epochs. Strict AP improvement replaces the
best checkpoint; ties retain the earlier epoch. No early stopping is added.

## Compatibility and checkpoints

The default CLI remains head-only; legacy `--learning-rate` sets the head LR.
`--head-learning-rate`, if supplied, overrides it. Old pooling calls and output
paths remain valid outside comparison mode. Input processing is unchanged.
The historical head-only numerical update is tested against the old construction.

All strategies retain the filename `best_pr_auc_classifier_head.pt` for tooling
compatibility; its version-2 contents are strategy-aware:

- A: original classifier/attention state keys, plus strategy and optimizer metadata.
- B: head plus strict state dictionaries for upper blocks 10/11.
- C: external head, PEFT adapter config and PEFT adapter state dictionary.

The loader first reconstructs the base and strategy, then restores all trained
weights. Available Hub commit revisions are recorded and used on reload.
For a local model directory, keep that directory immutable and available.
Base weights and tokenizer files are not embedded in these compact checkpoints.
Old head-only checkpoints without version/strategy fields remain loadable.
Loading now returns an eval-mode model; call `train()` explicitly to train.
These are inference checkpoints, not full optimizer/RNG resume checkpoints.

Smoke mode additionally compares every frozen parameter before/after training
and reloads the best checkpoint in a fresh model to compare validation predictions
(rtol=1e-5, atol=1e-6). Unit tests compare logits and restored parameter values.
Dry-run now includes a real optimizer step on its disposable model, with no
formal output checkpoint; this intentionally strengthens the old dry-run behavior.

## Dependencies (AutoDL)

`requirements.txt` adds only `peft==0.20.0`; it does not change the existing
Transformers constraint or install/replace CUDA PyTorch explicitly.
Local verification used Python 3.12.4, torch 2.13.0+cpu, Transformers 5.15.0,
PEFT 0.20.0, accelerate 1.12.0, psutil 7.2.2. This is NOT an AutoDL environment claim.
PEFT was installed with `--no-deps`, then only missing accelerate/psutil were
installed, also without dependency upgrades. `pip check` passed.
On AutoDL, inspect the existing versions and pip's installation plan before
installing the pinned PEFT dependency. Do not upgrade torch/Transformers blindly.
If dependencies are missing, install compatible versions without replacing the
existing CUDA stack, then rerun pip check and the tests below.

## Next commands: verification only

Local implementation verification: full pytest finished with **56 passed,
0 failed, 0 skipped**. A separate subprocess runner smoke completed A/B/C using
the synthetic model, 4 train + 4 validation records each, length 512, batch 2,
one epoch. All three reported frozen parameters unchanged and checkpoint
prediction round-trip passed. No CUDA/GPU or full-size PubMedBERT validation
is claimed by these CPU results.

Run from the repository root in the intended AutoDL Python environment.
First verify that the required PEFT dependency is installed:

```bash
python -c "import torch, transformers, peft, accelerate; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'accelerate', accelerate.__version__); print('CUDA', torch.cuda.is_available()); print([(torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory / 1024**3) for i in range(torch.cuda.device_count())])"
python -m pip check
python -m pytest -q
```

Pytest blocks opening the real repository test.csv before the OS open occurs.
Strategy tests use a local synthetic 12-layer BERT with hidden size 16, synthetic
criteria and train/validation rows. Existing criteria coverage tests read only
train/validation review keys and the workbook, not test articles.

GPU preflight (one disposable optimizer step per strategy, no checkpoints):

```bash
for strategy in classifier_head_only partial_unfreezing lora; do
  python scripts/gpu_preflight.py --comparison-mode --training-strategy "$strategy" --pooling cls --batch-size 32 --max-length 512 --seed 42 || break
done
```

Preflight logs GPU name, total VRAM, peak allocated/reserved memory, strategy,
microbatch and effective batch (both 32), length and parameter-update checks.
No gradient accumulation is currently implemented: batch 16 is effective batch 16,
NOT effective batch 32. If B/C OOM, stop; agree and implement accumulation before
using 16x2 or 8x4. Do not assume head-only preflight establishes B/C capacity.

Tiny real-model smoke (8 train + 8 validation requested, batch 2, one epoch per
strategy, CLS, length 512, seed 42, no test, no tuning):

```bash
python scripts/run_training_strategy_comparison.py --execute --smoke --output-root outputs/training_strategy_smoke
```

This smoke uses the selected PubMedBERT by default. A local immutable model and
tokenizer directory can be passed with `--model-name-or-path` to both commands.
Without CUDA, this smoke uses CPU; GPU preflight instead fails explicitly.

## Runner behavior for the later formal phase

Running the new runner without `--execute` only prints a plan and writes nothing.
It accepts no test or threshold flags, does not pass arbitrary trainer arguments,
and always passes `--comparison-mode`, which rejects test evaluation in the trainer.
The legacy single-model test flag remains outside comparison mode for compatibility.
No independent final-test evaluation entry point is added in this change.

Each invocation has a timestamp + unique suffix batch directory. Each strategy/LR
trial has an isolated seed/run directory with config, history, predictions,
per-review metrics, parameter_report, checkpoint and log. The batch manifest records
completed run paths, strategy, LR groups, seed, parameter count, best epoch and AP.
Failure stops the matrix and records the active trial/error; no failed trial is skipped.
Existing pooling output directories are not used.

The later matrix has two backbone-LR trials and two adapter-LR trials. A may be
reused with a verified `--baseline-summary`, or explicitly trained; reuse checks
the local checkpoint, basic config, seed, class weight and selection criterion.
Historical data/criteria/base-model identity must still be verified separately;
matching paths or row counts alone cannot establish identical contents.
Formal execution commands are deliberately omitted here. Multi-seed training,
unseen-article sensitivity analysis and threshold tuning are not run in this phase.
