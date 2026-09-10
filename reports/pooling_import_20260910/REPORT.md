# Pooling comparison 整理报告

四种正式 GPU 实验已经完整导入，未删除、覆盖或改写任何原始实验文件。桌面原件仍保留。实验结果可直接用于论文分析，CLS summary 可由现有下一阶段结果读取函数加载。

## 来源与去向

来源：`C:/Users/86150/Desktop/pooling实验结果/第一次实验结果-比较4种poolin/`。

目标：`C:/Users/86150/Desktop/论文代码/outputs/pooling_comparison/formal/20260905_122936/`。

| Pooling | 来源 pooling_comparison 下的运行目录 | 目标批次内运行目录 |
|---|---|---|
| CLS | cls/run_20260905_123026 | cls/run_20260905_123026 |
| Mean | mean/run_20260905_132403 | mean/run_20260905_132403 |
| Attention | attention/run_20260905_141749 | attention/run_20260905_141749 |
| Concat | concat/run_20260905_151150 | concat/run_20260905_151150 |

身份依据：`pooling_comparison_manifest_20260905_122936.json` 为唯一 completed manifest，创建于 2026-09-05 04:29:36 UTC，完成于 08:05:14 UTC（北京时间 12:29–16:05）。四份 config、checkpoint、5 轮 history、完成日志和 metrics 一致。config 明确 smoke/dry-run/test evaluation 为 false，无样本截断，GPU 为 RTX 4080。训练为 123,291 行 / 18 reviews（1,945 阳性），validation 为 22,397 行 / 4 reviews（414 阳性）。

## 旧结果与重复结果

- 项目原 `outputs/classifier_head/` 的七份运行全部是 2026-08-20 smoke。四种 pooling 子目录各一份，其余三份直接位于 classifier_head 下。均原位安全保留，新增 README 标明身份，避免破坏旧路径引用。
- `outputs/smoke_reporting/` 和 `outputs/strategy_verification/` 为其他 smoke / CPU 测试产物，也完整保留。
- 下载包三份 failed manifest 均在 CLS 阶段失败，runs 为空；放入正式批次的 `archive/failed_attempts/`，不计入正式结果。
- 四份 Jupyter 自动保存文件经 SHA-256 证明分别与 final_comparison、CLS config、CLS run.log、CLS summary 完全相同；放入 `archive/notebook_autosaves/`，未删除。
- 正式结果文件与项目原有 outputs 无相同 SHA-256 文件，因此未发现已导入的同一正式批次。
- 完整批次日志放入 `logs/pooling_comparison.log`，原始 completed manifest 和 final comparison 留在批次根目录。

## 代码和路径检查

- 未修改 `src/`、现有训练/运行脚本、配置、数据集或 split；未运行正式 GPU training。
- 单模型训练默认写入 `outputs/classifier_head/<pooling>/<run或smoke时间戳>/`；pooling runner 默认根为 `outputs/pooling_comparison/`，支持 `--output-root`，无需改默认值。
- 下一阶段 training-strategy runner 使用独立输出根；其 `read_result` 已支持 checkpoint 同目录回退，实际读取导入 CLS summary 成功，无需路径代码修补。
- 原始 JSON/checkpoint 中的远程绝对路径原样保留；新增 `LOCAL_INDEX.json` 提供可移植的相对路径。
- 新增本报告、目录说明和 `organize_and_verify.py` 整理/审计工具。整理前已存在的未提交代码改动未动。
- 新实验应显式使用全新的批次目录，避免复用本次已完成目录并重写汇总文件。

## 验证

- 全部 42 个下载文件均已复制、逐文件校验 SHA-256；源文件集合未变、原件哈希未变。
- 整理前已有 outputs 的 229 个文件均通过原哈希检查，无丢失或覆盖；详见 `existing_outputs_before.json`。
- 四种 pooling 的全部 8 个关键文件均非空；每份 history 都完整含 1–5 epoch，最佳 epoch 与最大 AP 相符。
- config、summary、metrics、history、checkpoint、final comparison 交叉核对一致。
- 四份 checkpoint 均以 `torch.load(weights_only=True, map_location='cpu')` 读取，分类头及 Attention 权重严格加载成功，参数量分别为 769 / 769 / 1537 / 2305。
- 从每种 pooling 的全部 22,397 条已保存预测重算 AP、ROC-AUC、混淆矩阵，与原记录一致；验证 threshold=0.5，per-review 表和 AP 一致。
- validation 的 review、title、标识符和标签逐行与本地 validation split 一致。没有打开真实 test.csv。
- Mean 某个 review 的未定义 precision 等 NaN 按原样保留，不替换为零。
- 现有 reporting 与 training-strategy CPU 回归测试在项目 `.venv` 环境中 **28 passed（31.62 秒）**，详见 `tests.txt`。首次误用系统 Python 时为 23 passed / 5 failed，五项均因缺少 PEFT；改用已有 PEFT 0.20.0 的项目环境后全部通过，未安装依赖、未改测试或训练代码。

## Validation 指标

下表为各自最佳 validation AP checkpoint，阈值固定 0.5；AP 在项目中命名为 PR-AUC，非梯形积分。

| Pooling | Best epoch | AP / PR-AUC | ROC-AUC | Recall | FNR | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CLS | 1 | 0.094375 | 0.676696 | 0.584541 | 0.415459 | 0.029206 | 0.055632 |
| Mean | 5 | 0.039143 | 0.657150 | 0.676329 | 0.323671 | 0.023995 | 0.046346 |
| Attention | 3 | 0.038720 | 0.636499 | 0.415459 | 0.584541 | 0.036658 | 0.067372 |
| Concat | 1 | 0.090941 | 0.723496 | 0.572464 | 0.427536 | 0.042865 | 0.079758 |

## 下一阶段

可以继续已有预测的论文分析，并将 CLS 作为现有策略对比流程的基线输入；CLS 的 AP 在此批次中最高，不代表已经验证统计显著性。

已用现有 runner 生成 `next_stage_plan.json`，其中 `execute=false`、`test_disabled=true`，复用当前 CLS summary，仅列出后续 partial-unfreezing / LoRA 的四项计划，没有执行它们。

如要开始下一阶段正式 GPU 训练，仍需在 GPU 环境准备并确认原 `models/pubmedbert` 基础模型/分词器。历史包没有基础模型、完整训练输入、criteria 的内容哈希，现有行数或路径一致不能证明这些内容完全相同；本次也未做完整 backbone 推理。这个复现边界与目录整理无关，不需重跑已经完成的 pooling comparison。checkpoint 不含 optimizer/RNG，不能当作完整断点续训包。
