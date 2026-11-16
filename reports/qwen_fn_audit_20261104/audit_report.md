# Qwen Full-Test 假阴性审计（部分完成，2026-11-04）

## 核实结果与尚未完成的工作

正式冻结结果中确有 **236 条**正标签且按冻结策略排除的记录，分别为 Leenaars_2019 2、Muthu_2021 207、Radjenovic_2013 4、van_Dis_2020 23。全部 23,592 条预测均逐条与 Test 行的 record ID、review、标签和源行索引核对；ID 与索引均唯一，没有不匹配。Test 与准则工作簿哈希均与运行清单一致。预测值未重新生成。

**审计未完成：**236 条已全部提取，但仅对其中 11 条进行了探索性逐条判读：4 条 A（表面明确冲突）、7 条 B（信息不足或有歧义），另外 225 条尚未判读。Muthu 的 207 条中仅判读 9 条（3 A、6 B），198 条未判读。C、D 在目前已判读样本中计数为零，不能解读为总体为零；未判读不是 D。样本为方便选取，不具有代表性；4/236 不是冲突率或错误率，也不是对全部冲突数量的估计。所有 A/B 均非独立裁决或 Qwen 理由，尚无经独立核实的标签错误数。

两条 FN 缺摘要，均为 van_Dis_2020，分别为源行 14480 和 14486；两条尚未逐条判定缺摘要是否阻止可靠判断，故该数量不能报告为已证实的“缺摘要导致无法评估”数量。Muthu 的 207 条均有非空摘要。准则映射中四个 review 均有非空纳入和排除文本。

## 方法与字段

冻结结果：`supplementary_202610/llm_baseline/qwen_final_test_20261029/predictions.jsonl`；运行清单与最终完成记录同目录的 `run_manifest.json`、`final_test_execution_summary.json`。源数据：`data/processed/splits/test.csv`，准则：`data/eligibility_criteria(排纳标准).xlsx` 的 `Eligibility criteria` sheet。`config/canonical_split.json` 确定 Test 的四个 review。

- Test `review_key`、`title`、`abstract`、`label_included` 分别对应 review、原始题名、原始摘要和 gold label；预测 `record_id` 为稳定 ID、`source_row_index` 为零基 Test 行、`review_name` 和 `true_label` 分别交叉校验 review 和 gold，`status` 和 `decision` 为运行输出。
- workbook 的 `Review key`、`Inclusion criteria`、`Exclusion criteria` 对应 review 和两段准则；若两段均为空，项目代码才回退到 `Eligibility criteria`。本次四个 review 无需回退。导出 CSV 保留原始题名、摘要和工作簿单元格文字；空摘要与空准则单独标记。
- `src/llm_baseline/evaluation.py` 的 `retain_unresolved_v1` 仅将 `completed/exclude` 映射为排除，uncertain、格式错误及传输失败均保留。故 FN 筛选是 `label_included=1 AND status=completed AND decision=exclude`；原 JSONL 没有名为 `final_decision` 的字段，该字段在任务中的含义由此策略计算。
- 正式清单记录 `qwen-zero-shot-screening-v2` 和 prompt SHA-256 `92d5ec3e57208464fdfafece65202a68d3363b790a64a9821cdfdc335317625a`；`src/llm_baseline/prompt.py` 当前版本名一致，并要求高召回初筛、不推断未陈述事实。运行清单注明当时工作区 dirty，因此仅凭当前源文件不能独立证明其字节与当时版本完全一致。运行重建审计 `reconstruction_audit.json` 报告 23,592 条 token 序列匹配及字段 roundtrip 零失败；原始完整摘要与模型实际 512 token 输入不必相同，本审计使用源文件全文，不能断言每段证据被模型看到。
- Test SHA-256 `23728ed5139430ee3531b4432a2acadcd49aff8c3d4180fa50f53b082e7ab658`；工作簿 SHA-256 `7a682824979cf65fec650ba31f6375637a2fe0fb24e7d979b66d5cb55b369b7c`。`results/qwen_final_test/per_review_qwen_metrics.csv` 与逐条计数一致。

## Muthu 解释

Muthu 的 FN 占全部 FN 的 87.71%（207/236），占其历史正标签的 61.79%（207/335）：这是已确认的 Qwen 与历史标签分歧集中现象，而不是已确认的系统性标注错误。工作簿同时要求脊柱手术相关、1:1 两臂平行 RCT、二分类主要或次要结局，排除无统计显著主要或次要结局的研究。两个已查看正标签条目明确是荟萃分析，另一个明确三组尸体标本试验；这些是值得追查的表面设计冲突，不能直接判断标签错误。另有方案论文与非手术研究等，是否不合格依赖纳入阶段和实际准则；未报告显著不等于明确没有显著结果，单项非显著不等于全部主要和次要结局均非显著，连续主要结局也不能证明没有二分类次要结局。

项目中找到了原始整理记录 `数据(raw)/review整理/Muthu_2021_final.csv`，但未找到足以解释各标签来源与筛选阶段的原始 review 全文或逐条筛选决定；因此不能确定这些工作簿准则与历史正标签的定义和阶段完全一致。需取得原始 review 论文及补充材料、检索集合及包含/排除列表、数据集发布说明和正标签的生成规则（初筛保留、全文纳入、最终分析或其他），再针对冲突案例查原文和人工双人裁决。

## 后续建议

完成其余 225 条独立逐条判读和双人分歧裁决，并溯源历史标签及 Muthu 的原始流程。之后如有必要，在预先登记并明确标为探索性的敏感性分析中使用单独保存的裁决标签复算冻结预测指标；不改变原 Test、原标签或原始报告，不重新请求 Qwen，不把探索性结果替换正式结果。Python 虚拟环境入口指向不存在的 Python 安装；本次用 PowerShell 只读解析并核验，未进行训练、模型推理或 API 调用。

`qwen_fn_audit.csv`、`summary_by_review.csv`、`summary_by_category.csv` 和 `illustrative_cases.md` 是本次部分审计产物；`extract.ps1` 和 `annotate.ps1` 是生成和标注脚本，重运行提取脚本会拒绝覆盖已有文件。
