"""Offline paired evaluation after all 5,000 Qwen Full Test predictions exist."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import pandas as pd
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.llm_baseline.evaluation import apply_screening_policy, evaluate_policy_records, wilson_interval


def main() -> int:
    out=ROOT/'outputs'/'pubmedbert_vs_qwen_full_test_subset'
    pub=pd.read_csv(out/'pubmedbert_predictions_subset.csv',low_memory=False)
    path=out/'predictions_qwen_full'/'predictions.jsonl'
    qwen=pd.DataFrame(json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x)
    if len(pub)!=5000 or len(qwen)!=5000 or pub.record_id.duplicated().any() or qwen.record_id.duplicated().any() or set(pub.record_id)!=set(qwen.record_id): raise ValueError('Incomplete or misaligned paired predictions')
    qwen=qwen.rename(columns={'review_name':'review_key','gold_label':'gold_label_qwen'})
    pair=pub.merge(qwen[['record_id','review_key','gold_label_qwen','status','decision','usage']],on='record_id',validate='one_to_one')
    pair=pair.rename(columns={'review_key_x':'review_key', 'review_key_y':'review_key_qwen'})
    if not (pair.review_key == pair.review_key_qwen).all(): raise ValueError('Review-key mismatch')
    if not (pair.gold_label.astype(int)==pair.gold_label_qwen.astype(int)).all(): raise ValueError('Gold-label mismatch')
    pair['pubmedbert_retained']=pair.predicted_label.astype(int).eq(1)
    scored=apply_screening_policy(pair[['review_key','gold_label','status','decision']].rename(columns={'gold_label':'label_included'}))
    pair['qwen_retained']=scored.retained.to_numpy()
    def decision_series(retained):
        return pd.Series(["include" if bool(value) else "exclude" for value in retained], index=retained.index)
    def summary(retained):
        frame=pd.DataFrame({'review_key':pair.review_key,'label_included':pair.gold_label.astype(int),'status':'completed','decision':decision_series(retained)})
        return evaluate_policy_records(frame)['micro_global']
    positives=pair[pair.gold_label.eq(1)].copy(); a=positives.pubmedbert_retained; b=positives.qwen_retained
    only_pub=int((a&~b).sum()); only_qwen=int((~a&b).sum()); discordant=only_pub+only_qwen
    pubmedbert_summary=summary(pair.pubmedbert_retained)
    # The generic policy evaluator uses synthetic include/exclude decisions.  Correct
    # the presentation metadata so the saved report preserves each system's actual
    # frozen decision rule rather than suggesting a new 0.5 probability threshold.
    pubmedbert_summary['threshold']=0.016082723
    pubmedbert_summary['score_type']='frozen_historical_probability_threshold'
    qwen_summary=summary(pair.qwen_retained)
    qwen_summary['threshold']=None
    qwen_summary['score_type']='categorical_policy_exclude_is_auto_exclude'
    result={'pubmedbert':pubmedbert_summary,'qwen_full':qwen_summary,'positive_records':len(positives),'natural_test_positive_rate_note':0.020006781959986435,
            'recall_ci_wilson':{'pubmedbert':wilson_interval(int(a.sum()),len(a)),'qwen_full':wilson_interval(int(b.sum()),len(b))},
            'positive_retention_pairing':{'both_correctly_retained':int((a&b).sum()),'only_pubmedbert_correctly_retained':only_pub,'only_qwen_correctly_retained':only_qwen,'both_false_negative':int((~a&~b).sum()),'mcnemar_exact_two_sided_p':None if discordant==0 else binomtest(only_pub,discordant,0.5).pvalue},
            'per_review':[],'interpretation_note':'This is a stratified Test subset. Report its observed positive count and compare paired recalls; do not infer full-Test prevalence or causality from the subset alone.'}
    for review,g in pair.groupby('review_key',sort=True):
        r={'review_key':review,'records':len(g),'positives':int(g.gold_label.sum())}
        for name,col in [('pubmedbert','pubmedbert_retained'),('qwen_full','qwen_retained')]:
            f=pd.DataFrame({'review_key':g.review_key,'label_included':g.gold_label.astype(int),'status':'completed','decision':decision_series(g[col])})
            metric=evaluate_policy_records(f)['micro_global']
            if name == 'pubmedbert':
                metric['threshold']=0.016082723
                metric['score_type']='frozen_historical_probability_threshold'
            else:
                metric['threshold']=None
                metric['score_type']='categorical_policy_exclude_is_auto_exclude'
            r[name]=metric
        result['per_review'].append(r)
    prompt=sum((v or {}).get('prompt_tokens',0) for v in qwen.usage); completion=sum((v or {}).get('completion_tokens',0) for v in qwen.usage)
    result['qwen_usage']={'prompt_tokens':int(prompt),'completion_tokens':int(completion),'estimated_cost_cny_cache_miss':prompt*.8/1e6+completion*2.7/1e6}
    (out/'comparison_evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'records':len(pair),'positive_records':len(positives)},ensure_ascii=False)); return 0
if __name__=='__main__': raise SystemExit(main())
