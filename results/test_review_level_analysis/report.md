# Test Review-level Performance Analysis

本报告仅整理 PubMedBERT + LoRA 的正式 Test 结果。模型 checkpoint 见 [正式 Test 配置](../../outputs/final_test/final_test_config.json)；分类阈值 `0.016082723` 由 [Validation 阈值记录](../../results/formal_training_strategy_20260910/thresholds/selected_validation_threshold.json)选定，并在全部四个未见过的 Test reviews 上保持固定。原始材料为[逐条预测](../../outputs/final_test/final_test_predictions.csv)、[各 review 指标](../../outputs/final_test/per_review_test_metrics.csv)和[总体指标](../../outputs/final_test/final_test_metrics.json)。评价入口为 `scripts/evaluate_final_test.py`。本次没有重新推理或更改原始结果。

## 核验结果与各 review 表现

对 23,592 条已保存预测逐行读取真实标签、预测概率和二分类决策，独立计算混淆矩阵、阳性比例、Recall、Precision、AP 与 Workload Reduction。全部预测决策均符合固定阈值；四个 review 的结果与原有 `per_review_test_metrics.csv` 一致，汇总结果与 `final_test_metrics.json` 一致。AP 用 `average_precision_score` 从各 review 的原始概率计算。未发现重复的复合文献标识。完整精度结果见 [per_review_verified.csv](per_review_verified.csv)。

| Test review | Records | Positive | Prevalence | TP | FP | TN | FN | Recall | Precision | AP | Workload Reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Leenaars_2019 | 5,811 | 17 | 0.29% | 17 | 4,367 | 1,427 | 0 | 100.00% | 0.39% | 0.7581 | 24.56% |
| Muthu_2021 | 2,718 | 335 | 12.33% | 335 | 2,373 | 10 | 0 | 100.00% | 12.37% | 0.2525 | 0.37% |
| Radjenovic_2013 | 5,935 | 48 | 0.81% | 48 | 5,572 | 315 | 0 | 100.00% | 0.85% | 0.1128 | 5.31% |
| van_Dis_2020 | 9,128 | 72 | 0.79% | 70 | 5,666 | 3,390 | 2 | 97.22% | 1.22% | 0.0521 | 37.16% |

## Macro/Micro 汇总与计算方法

Micro 指标先合并各 review 的整数混淆矩阵，再计算指标。合并结果为 TP 470、FP 17,978、TN 5,142、FN 2，总计 23,592 条、正例 472 条。Macro 指标先分别计算四个 review 的指标，再取不加权算术平均。对任一 review，`Recall = TP / (TP + FN)`，`Precision = TP / (TP + FP)`，`Workload Reduction = (TN + FN) / Records`。六个汇总值已存在于原始 `final_test_metrics.json` 的 `micro_global` 与 `macro_review_level` 字段；[macro_micro_summary.csv](macro_micro_summary.csv)只是便于查阅的表格形式，与原始值完全一致。

| 指标 | Micro | Macro |
| --- | ---: | ---: |
| Recall | 99.58% | 99.31% |
| Precision | 2.55% | 3.71% |
| Workload Reduction | 21.80% | 16.85% |

## 从结果直接观察到的事实

- Leenaars、Muthu 和 Radjenovic 的 Recall 均为 100.00%，且没有 FN；van_Dis 的 Recall 为 97.22%，全部 2 个 FN 均出自该 review。
- Muthu 的 335 个相关文献全部保留，但只预测排除 10/2,718 条，Workload Reduction 为 0.37%。Radjenovic 的 Workload Reduction 也只有 5.31%。van_Dis 预测排除 3,392/9,128 条，Workload Reduction 最高，为 37.16%，同时出现 2 个 FN。
- 四个 review 的阳性比例从 Leenaars 的 0.29% 到 Muthu 的 12.33%；Precision 从 0.39% 到 12.37%。AP 也存在明显差异：Leenaars 为 0.7581，van_Dis 为 0.0521。

## 基于现有结果的解释

- Muthu 的 Precision（12.37%）接近其阳性比例（12.33%），与模型几乎保留全部文献的事实一致。因此，100% Recall 在该 review 上没有带来实际可观的人工工作量减少。Radjenovic 同样以较低的排除比例取得 100% Recall。
- 阳性比例影响 Precision 的解读，但不能单独解释各 review 的差异。Precision 同时取决于 FP：Leenaars 只有 17 个正例，在固定阈值下产生 4,367 个 FP，因而虽然 AP 达 0.7581，阈值处的 Precision 仍只有 0.39%。AP 衡量整体排序，不能替代该固定阈值下的筛选效果。
- Micro Recall 为 99.58%，高于 Macro Recall 99.31%。Muthu 包含全部 472 个正例中的 335 个，且无 FN，对按正例数加权的 Micro Recall 影响较大；Macro Recall 则让出现 2 个 FN 的 van_Dis 占四分之一权重。Micro Workload Reduction 为 21.80%，高于 Macro 的 16.85%；记录最多、排除比例较高的 van_Dis 对按文献数加权的 Micro 值影响较大。这说明整体 Test 指标不能充分代表每个 review 的实际筛选效果。

本报告仅比较这四个 held-out reviews 在同一个固定 Validation 阈值下的已观察结果。没有分析新的分数分布、调整阈值或改变 review 划分；这些数值不能作为不同 review-level splits 稳健性的证据。
