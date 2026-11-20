# Parameter-Efficient Adaptation 资源评价说明

本说明仅整理五组正式 Training Strategy Comparison 的已有记录，不进行训练或资源测试。逐组核对了 [comparison manifest](../formal_training_strategy_20260910/comparison_manifest.json)、`config.json`、`parameter_report.json`、`run_summary.json` 和 `training_history.csv`。完整精度的参数比例和每组来源路径见 [resource_summary.csv](resource_summary.csv)。

| Training Strategy | 策略学习率 | 可训练参数 | 占该模型总参数比例 | 训练总耗时 | GPU 峰值显存 |
| --- | ---: | ---: | ---: | --- | --- |
| Classifier Head Only | Head `1e-4` | 769 | 0.0007% | `not_recorded` | `not_recorded` |
| Partial Unfreezing | Backbone `1e-5` | 14,176,513 | 12.9486% | `not_recorded` | `not_recorded` |
| Partial Unfreezing | Backbone `2e-5` | 14,176,513 | 12.9486% | `not_recorded` | `not_recorded` |
| LoRA | Adapter `5e-5` | 295,681 | 0.2693% | `not_recorded` | `not_recorded` |
| LoRA | Adapter `1e-4` | 295,681 | 0.2693% | `not_recorded` | `not_recorded` |

表中的学习率是各策略发生更新的主干或 adapter 学习率；五组的分类头学习率均为 `1e-4`。参数比例直接取自原始 `parameter_report.json`，且参数数量与各组 `config.json`、`run_summary.json` 及 manifest 一致。LoRA 模型总参数为 109,777,921（包含新增 adapter 参数）；Head-only 和 Partial Unfreezing 的模型总参数为 109,483,009，因此比例按各自模型的总参数计算。

五组均记录 seed 42、CLS pooling、batch size 32、5 epochs、最大输入长度 512、相同的 Train/Validation 行数，以及相同的加权 BCE 正类权重。参数数量上，LoRA 只更新 295,681 个参数，约为 Partial Unfreezing 的 2.09%（后者约为 LoRA 的 47.95 倍）；Head-only 只更新 769 个参数。这个比较支持“LoRA 相对 Partial Unfreezing 显著减少可训练参数”的表述，但 Head-only 的可训练参数更少。

所检查的 summary 和五个 epoch 的 history 均无训练总耗时或 GPU 峰值显存字段；manifest 只有批次创建时间，没有可用于计算各运行耗时的明确结束时间。本地所定位的五组正式运行目录也没有 `run.log`。因此，现有材料**不能**量化或直接比较实际训练速度和显存消耗。可训练参数变少不等于训练用时或峰值显存按同一比例下降：这些资源还取决于前向计算、反向传播、激活存储及硬件运行条件。论文应将“parameter-efficient”限定为**可训练参数数量**层面的已测量事实，不宣称未经记录的速度或显存优势。
