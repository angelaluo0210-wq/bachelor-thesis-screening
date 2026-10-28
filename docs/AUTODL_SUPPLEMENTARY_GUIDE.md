# AutoDL 补充实验部署指南

> 补充：六次正式训练均完成后，可运行下列只读汇总命令；它只接收 `test_evaluated: false` 的训练摘要，不会读取 Test。
>
> ```bash
> python scripts/summarize_supplementary_results.py \
>   --output-csv supplementary_202610/validation_summary.csv \
>   --output-json supplementary_202610/validation_summary.json
> ```
>
> 汇总器拒绝覆盖既有输出。归档时下载整个 `supplementary_202610/`，并保留上述汇总、setup/validation/orchestration 日志、配置、manifest、checkpoint 和 validation predictions。

本指南对应仓库内实际存在的 `setup_autodl.sh`、`validate_autodl.sh` 和 `run_autodl_supplementary.sh`。脚本默认不会训练；只有显式 `--execute` 才会启动正式训练。

## 上传内容

上传代码、`config/canonical_split.json`、`data/processed/splits/train.csv`、`validation.csv`、eligibility criteria Excel、`model/pubmedbert/`（`config.json`、tokenizer 配置、vocab 和唯一权重文件）、`requirements-autodl-supplementary.txt`，以及历史 seed=42 的配置/metrics 作为溯源证据。不要上传 `.venv/`、`__pycache__/`、raw archive、历史 `outputs/`，也不需要上传 `test.csv`。

建议目录映射：项目为 `/workspace/bachelor-thesis-screening`，模型为 `$PROJECT_ROOT/model/pubmedbert`。如使用其他位置，设置 `PROJECT_ROOT`、`MODEL_DIR`、`CRITERIA_PATH`、`PYTHON_BIN` 环境变量；不要假设 `/workspace/models/pubmedbert` 已存在。

## 实例与预检

建议 24 GB GPU 显存；16 GB 可尝试但必须先通过 smoke/preflight。预留至少 30 GB 磁盘。进入项目根目录后：

```bash
chmod +x scripts/setup_autodl.sh scripts/validate_autodl.sh scripts/run_autodl_supplementary.sh
PROJECT_ROOT="$PWD" MODEL_DIR="$PWD/model/pubmedbert" scripts/setup_autodl.sh
PROJECT_ROOT="$PWD" MODEL_DIR="$PWD/model/pubmedbert" scripts/validate_autodl.sh
```

`setup` 检查 GPU、CUDA、磁盘、模型、Train/Validation、criteria，安装非 torch 的受控依赖并记录版本与 SHA-256。它不会下载模型或删除结果。`validate` 执行模型测试和三条真实 forward/backward smoke 分支；失败会保留日志和 smoke artifacts，且不允许进入正式训练。

## 计划、训练与恢复

先查看六次训练计划：

```bash
scripts/run_autodl_supplementary.sh
```

验证通过后才可执行：

```bash
PROJECT_ROOT="$PWD" MODEL_DIR="$PWD/model/pubmedbert" scripts/run_autodl_supplementary.sh --execute
```

建议用 tmux：`tmux new -s supplementary`，执行命令后按 `Ctrl-b d` 脱离；用 `tmux attach -t supplementary` 回连。日志位于 `supplementary_202610/autodl_orchestration/`，每个训练目录内还有 `run.log`、config、history、checkpoint、validation predictions 和 metrics。

当前训练器不保存 optimizer/RNG 状态，所以不支持精确断点续训。`--resume` 仅表示允许人工核验成功 manifest 后跳过完整 run；对状态不明确或部分产物的目录必须人工检查，绝不覆盖或删除。

## 下载、备份与停止

完成后下载整个 `supplementary_202610/`，而非只下载汇总表；必须保留 configs、manifests、logs、histories、checkpoints 和 validation predictions。用 `tail -f supplementary_202610/autodl_orchestration/*.log` 查看状态。需要停止时，在 tmux 内使用 `Ctrl-c`，检查已保留日志后再关闭实例；确认没有运行进程后关机，避免持续计费。
