# Post-hoc Sensitivity Analysis：缺失 Abstract 默认保留

本分析仅使用 PubMedBERT + LoRA 的[正式 Test 逐条预测](../../outputs/final_test/final_test_predictions.csv)和[原始 Test 数据](../../data/processed/splits/test.csv)，不重新运行模型。原始策略沿用 [Test 配置](../../outputs/final_test/final_test_config.json)中由 Validation 选定并固定的阈值 `0.016082723`；事后规则仅将**缺失 Abstract 且原本预测为 Exclude** 的文献改为 Include，交由人工复核。其他文献的预测保持不变。调整后的结果不能替代[原始正式 Test 指标](../../outputs/final_test/final_test_metrics.json)。

## 数据核验

- Test 数据与预测均有 23,592 条。原始 Test 文件的 SHA-256 与正式 Test 配置记录一致；逐行核对 review、title、PMID、DOI、OpenAlex ID 和真实标签，没有错位。保存的二分类结果均符合固定阈值。
- 按正式代码的缺失值口径（Pandas `read_csv` 默认缺失值识别，加空白字符串判断），缺失 Abstract 共 2,517 条，其中真实正例 12 条、负例 2,505 条；与正式 Test 配置一致。直接只数空白字符串会得到 2,516 条，因为另有 1 条 Abstract 为 `NA`，Pandas 将其识别为缺失。该条原本已预测为 Include，不影响本次改判数。
- 原始 Test 的 2 个 False Negatives 均缺失 Abstract，与已有[漏检背景记录](../../results/final_error_analysis/missing_abstract_context.json)一致。原始混淆矩阵和指标也已从逐条预测复算，并与正式结果一致。

## 两种筛选策略

Workload Reduction 定义为预测排除数除以 Test 总数，即 `(TN + FN) / 23,592`。以下百分比仅在展示时四舍五入；[comparison_metrics.csv](comparison_metrics.csv)保留完整计算精度。

| 策略 | TP | FP | TN | FN | Recall | Precision | Workload Reduction | 预测保留 | 预测排除 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A. 原始正式 Test 策略 | 470 | 17,978 | 5,142 | 2 | 99.58% | 2.55% | 21.80% | 18,448 | 5,144 |
| B. 缺失 Abstract 默认保留 | 472 | 18,468 | 4,652 | 0 | 100.00% | 2.49% | 19.72% | 18,940 | 4,652 |

原本预测为 Exclude 的缺失摘要文献共有 **492 条**，现全部改为 Include，详见 [changed_predictions.csv](changed_predictions.csv)。其中真实正例 **2 条**、负例 **490 条**；其余 2,025 条缺失摘要文献原本已被保留。新规则因此多增加 **492 条人工筛选工作**，消除了这份 Test 中现有的 **2 个 FN**。Workload Reduction 从 21.80% 降至 19.72%，下降 **2.09 个百分点**；Precision 从 2.55% 略降至 2.49%。

## 解释边界

这是一项基于已知 Test 标签的 **Post-hoc Sensitivity Analysis**，用于量化保守规则在现有 Test 上的 Recall–Workload Reduction 权衡。它说明该规则可以避免这份 Test 中观察到的两次漏检，同时需要额外人工复核 492 篇文献；不能据此断言缺失摘要导致了模型漏检，也不能保证该规则在新数据上实现 100% Recall。正式 Test 结果仍应报告策略 A，策略 B 只能作为事后敏感性分析单独呈现。
