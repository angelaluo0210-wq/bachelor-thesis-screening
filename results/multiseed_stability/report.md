# Multi-Seed Stability Analysis（仅基于已有 Validation 结果）

## 数据来源与核验

六次运行的完整来源路径、原始精度 AP、配置、逐条预测、训练历史、运行摘要及 checkpoint 见 [seed_ap_values.csv](seed_ap_values.csv)。CLS–LoRA seed 42 是正式训练策略比较中的获选运行；CLS–LoRA seed 123、2026 来自 `supplementary_202610/cls_lora_multiseed/`；三个 Mean–LoRA 运行来自 `supplementary_202610/mean_lora/`。已有 `supplementary_202610/validation_summary.csv` 汇集了五个补充运行的 AP，但没有计算两组均值和样本标准差，因此本目录补充描述性统计。

六份 `metrics.json` 均记录 `selection_metric = validation_pr_auc`。逐份核对 `training_history.csv`、`run_summary.json` 和 checkpoint 路径后，最高 Validation AP 对应的最佳 epoch 分别为：CLS seed 42/123/2026 的 4/1/5，以及 Mean seed 42/123/2026 的 3/4/1。六个最佳 checkpoint 文件均存在。每份 Validation 预测均有 22,397 条记录，其中正例 414 条；六份预测的文献标识、顺序和真实标签逐行一致。用各自已保存的预测概率计算 `average_precision_score`，所得值与 `metrics.json` 完全相同。论文 Table 4.8 报告的六个四位小数 AP 均与原始结果一致。

| Seed | CLS–LoRA AP | Mean–LoRA AP | CLS − Mean |
| ---: | ---: | ---: | ---: |
| 42 | 0.1901 | 0.0859 | 0.1042 |
| 123 | 0.1794 | 0.0791 | 0.1003 |
| 2026 | 0.1250 | 0.0596 | 0.0654 |

统计计算使用 [seed_ap_values.csv](seed_ap_values.csv) 中未经四舍五入的原始 AP，表格仅在展示时保留四位小数。

## 描述性统计

| Pooling | Mean AP | 样本 SD（ddof=1） | Min AP | Max AP | Range | 相对 SD（SD / Mean） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CLS–LoRA | 0.1648 | 0.0349 | 0.1250 | 0.1901 | 0.0651 | 21.17% |
| Mean–LoRA | 0.0749 | 0.0136 | 0.0596 | 0.0859 | 0.0262 | 18.18% |

完整统计值见 [stability_summary.csv](stability_summary.csv)，折线图见 [stability_comparison.svg](stability_comparison.svg)。

在三个已观察到的 seed 下，CLS–LoRA 的 AP 均高于 Mean–LoRA；平均 AP 高 0.0899。CLS–LoRA 的绝对波动更大：样本 SD 为 0.0349，AP 范围为 0.0651；Mean–LoRA 分别为 0.0136 和 0.0262。按各组均值归一化，CLS–LoRA 的相对 SD 为 21.17%，Mean–LoRA 为 18.18%。两组均在 seed 2026 取得最低 AP，表明这三次运行的结果对随机种子有可见变化。本分析只有每组 3 次运行，不做显著性检验，也不据此推断 LoRA 相对 Head-only 或 Partial Unfreezing 的稳定性。

## 训练条件与限制

六份配置均记录同一训练策略 LoRA、Train 123,291 条、Validation 22,397 条、AdamW（weight decay 0.01）、分类头学习率 `1e-4`、adapter 学习率 `5e-5`、batch size 32、5 epochs、最大输入长度 512、LoRA rank 8、alpha 16、LoRA dropout 0.05、目标模块 `query`/`value`、分类头 dropout 0.1，以及 `weighted_BCEWithLogitsLoss`，正类权重均为 `62.38868894601542`。六次运行均未评价 Test。

记录中有两处需要保留的差异。CLS 运行设置 `comparison_mode=true`，Mean 运行设置 `false`；现有训练代码中该模式用于约束 CLS、阈值及正式比较参数，并对基础模型结构增加检查。seed 42 的 CLS 配置没有单列 `class_weighting` 字段，但已记录加权 BCE 损失和相同正类权重。另一个差异是正式 CLS seed 42 使用较早的服务器路径，其余运行使用 AutoDL 路径；六份 Validation 预测已经逐行对齐，Train 行数和正类权重相同，但历史 Train 文件的字节级一致性无法仅从这些运行记录独立确认。因此，本报告只描述已保存运行的观测差异，不把全部 AP 变化解释为随机种子的因果效应。

正式 CLS seed 42 的已整理材料包含配置、指标、训练历史和运行摘要，原始逐条预测及 checkpoint 也存在；在所定位的目录中未找到该运行的独立 `run.log`。其余五次运行均有 `run.log`。本次未修改任何原始文件，也未重新训练、推理、校准阈值或运行 Test。
