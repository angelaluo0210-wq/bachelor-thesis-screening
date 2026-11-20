# Historical environment evidence

This page reports versions found in historical configuration and run records. `not_recorded` in `experiment_provenance.csv` means that a per-run record did not establish that version. The current computer environment was not used to fill historical gaps.

| Evidence and runs | Python | PyTorch | Transformers | PEFT | Accelerate | NumPy | Pandas | scikit-learn | CUDA | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pooling run `config.json` (4 runs) | not_recorded | 2.6.0+cu124 | 5.16.1 | not_recorded | not_recorded | not_recorded | not_recorded | not_recorded | 12.4 | NVIDIA GeForce RTX 4080 |
| Formal strategy `config.json` (5 retained summaries), selected LoRA run, and supplementary CLS/mean/BCE run configurations | not_recorded | 2.6.0+cu124 | 5.16.1 | 0.20.0 | 1.15.0 | 2.1.3 | not_recorded | 1.9.0 | 12.4 | NVIDIA GeForce RTX 4080 SUPER |
| Qwen final Test `run_manifest.json` | 3.12.4 | not_recorded | 5.16.1 | not_recorded | not_recorded | not_recorded | not_recorded | not_recorded | not_recorded | not_recorded |
| Qwen Validation Full/Truncated preparation `experiment_config.json` | 3.12.4 (preparation only) | not_recorded | 5.16.1 (preparation only) | not_recorded | not_recorded | not_recorded | not_recorded | not_recorded | not_recorded | not_recorded |
| AutoDL preparation log `supplementary_202610/autodl_setup/setup_20261028T001939Z.log` | 3.12.3 | 2.5.1+cu124 at setup | 5.16.1 after install | 0.20.0 after install | 1.15.0 after install | 2.1.3 | 2.3.3 after install | 1.9.0 after install | 12.4 | not_recorded in this log |

The AutoDL setup log is **environment preparation evidence**, not a per-run freeze. Its PyTorch 2.5.1 value must not replace the PyTorch 2.6.0 value recorded in later training configurations. The setup Python/Pandas values likewise do not establish the exact interpreter or Pandas version for every subsequent run.

The two TF-IDF baselines, original data split, frozen-prediction analyses, and post-hoc comparisons have no complete per-run dependency inventory. `requirements.txt` contains version ranges; `requirements-autodl-supplementary.txt` pins only part of the supplementary stack. No complete historical `pip freeze` or `environment.yml` was found.

Seeds are recorded per row in the provenance matrix. The formal Pooling and training-strategy comparisons use seed 42; supplementary CLS/mean runs use the seeds shown in their configurations; Qwen sampling uses seed 42 where documented. A Qwen API call is not a seeded local model run. `not_recorded` is retained where a run did not document a seed.

Code provenance is separate from environment provenance. The formal Qwen Test run records commit `b45968fdd3eefd644083f672c6534216e3776c56` with a dirty worktree; the changed files were not captured as a patch. The Qwen Validation/Full Test subset preparation records commit `150c1334e5f1e35a127b246bcdbb24d971c0c7b1` and a dirty worktree, but the actual execution commit is not recorded. Historical commits for the remaining runs are not established by their filenames or by the current `work` HEAD.
