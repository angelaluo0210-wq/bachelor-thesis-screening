# Thesis Revision Impact Matrix

This is a comparison against the LaTeX source only. It does not modify any thesis file.

| Chapter / section | Current content | New evidence | Recommendation | Priority |
|---|---|---|---|---|
| Abstract (in `thesis/thesis.tex`) | PubMedBERT-only headline needs checking against final scope | Complete Qwen Test and baselines now exist | UPDATE: describe primary PubMedBERT result accurately; add Qwen only if the abstract scope expands | High |
| Ch. 1 RQs and contributions | Clear staged PubMedBERT protocol; no Qwen comparison RQ | Qwen full Test is a new, same-population comparison but different decision policy | ADD: either a tightly bounded exploratory comparison objective or explicitly exclude it from RQs | High |
| Ch. 2 LLM background | Reviews LLM literature but no local Qwen evidence | Qwen3.8-Flash zero-shot run, policy, failures, and limitations | UPDATE: connect comparison design to hard-decision / unresolved handling | Medium |
| Ch. 3 data and methodology | Primary split/model/threshold protocol documented | TF-IDF input differs; Qwen prompt/policy; subset is label-stratified and precision matching uses Test labels | ADD: an explicit “supplementary analyses” subsection and comparability caveats | High |
| Ch. 3 evaluation metrics | Correctly distinguishes AP and threshold metrics | New Qwen has no score/AP/ROC; valid-response-only is supplementary | UPDATE: define end-to-end denominator and N/A metrics | High |
| Ch. 4 pooling/training results | Formal comparisons correctly report selection | Multi-seed and loss ablation now qualify the statement that every formal run used one seed | UPDATE: add a supplementary robustness table or revise limitation language | High |
| Ch. 4 final Test | 470/472, .016082723, 21.80% claims match artifacts | No conflict detected | KEEP | High |
| Ch. 4 baselines | No TF-IDF results | Two completed Validation-only classical baselines | ADD: clearly labelled Validation-only table; do not imply Test comparison | Medium |
| Ch. 4 LLM comparison | No Qwen results | Full Test and subset results now available | ADD: primary Qwen end-to-end table, then separately label 5k subset/precision match post-hoc | High |
| Ch. 4 figures/tables | PR curve and pipeline support PubMedBERT | New comparative results are tabular; precision-recall SVG is exploratory | ADD/VERIFY: only add subset curve with an explicit post-hoc caption | Medium |
| Ch. 5 discussion | States no repeated runs | New seed checks exist: selected CLS-LoRA AP .1901/.1794/.1250 across seeds 42/123/2026 | UPDATE: replace absolute absence claim with limited, variable Validation-only seed evidence | High |
| Ch. 5 utility/limitations | Correctly warns against autonomous exclusion | Qwen achieves higher exclusion volume but misses half positives; precision match has Test-label bias | ADD: direct trade-off and post-hoc threshold-selection warning | High |
| Ch. 6 conclusion | PubMedBERT conclusion is artifact-consistent | Supplementary experiments do not change official frozen Test claim | KEEP primary conclusion; ADD narrowly worded comparison paragraph if results chapter adds it | High |
| Appendix | Need traceable configurations | Qwen manifests, prompts, policies, and inventory now exist | ADD: artifact paths/configuration table, without secrets or raw responses | Medium |

Priority definitions: High = scientific claim/validity; Medium = completeness/traceability. The primary result must remain the Validation-selected, frozen-threshold PubMedBERT Test. The .85556966 threshold must never be introduced as a primary threshold.
