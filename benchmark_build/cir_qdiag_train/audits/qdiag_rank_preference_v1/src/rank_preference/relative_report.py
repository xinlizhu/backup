from collections import Counter,defaultdict
from pathlib import Path

from rank_preference.common import digest,read_json,read_jsonl,write_json,write_jsonl
from rank_preference.report import export_html,resolve_outcome,write_csv
from rank_preference.schema import RelativePairJudgment


def preference_tables(rows,simulated=False):
    groups=defaultdict(list)
    for row in rows:groups[row['epoch'],row['rank']].append(row)
    table=[];diagnostics=[]
    for (epoch,rank),group in sorted(groups.items()):
        counts=Counter(x['outcome'] for x in group)
        judged=counts['teacher_preferred']+counts['student_preferred']
        complete=counts['pending']==0 and counts['failed']==0
        show=judged>0 and complete and not simulated
        common={'epoch':epoch,'student_checkpoint':Path(group[0]['student_checkpoint']).name,'rank':rank}
        table.append({**common,
            'teacher_preference_percent':100*counts['teacher_preferred']/judged if show else None,
            'student_preference_percent':100*counts['student_preferred']/judged if show else None})
        diagnostics.append({**common,'query_count':len(group),'judged_comparisons':judged,
            'teacher_preferred':counts['teacher_preferred'],'student_preferred':counts['student_preferred'],
            'same_image_skipped':counts['same_image_skipped'],'pending':counts['pending'],'failed':counts['failed'],
            'low_confidence_choices':sum(x.get('judge_confidence')=='low' for x in group),
            'group_complete':complete,'simulated':simulated})
    return table,diagnostics


def summarize_relative(config):
    out=Path(config['runtime']['output_dir'])
    meta=read_json(out/'prepared_metadata.json')
    comparisons=read_jsonl(out/'comparisons.private.jsonl')
    cases={x['case_id']:x for x in read_jsonl(out/'judge_cases.private.jsonl')}
    results={p.stem:read_json(p) for p in (out/'judgments').glob('*.json')}
    if set(results)-set(cases):raise ValueError('Unexpected judgment IDs in output')
    simulated=any(r.get('simulated',False) for r in results.values())
    identity_path=out/'judgment_identity.json'
    if identity_path.exists():
        identity=read_json(identity_path)
        if identity['judging'].get('mode')!='relative_preference':
            raise ValueError('Refusing to summarize old absolute judgments as relative preferences')
        signature=digest(identity);simulated=identity['mock']
        if any(r['signature']!=signature or r['simulated']!=simulated for r in results.values()):
            raise ValueError('Judgment identity mismatch')
    elif results:raise ValueError('Judgment identity is missing')
    for case_id,result in results.items():
        if result['status']=='complete':
            answer=RelativePairJudgment.model_validate(result['judgment'])
            if answer.case_id!=case_id:raise ValueError('Judgment case_id mismatch')
    rows=[]
    for c in comparisons:
        r=results.get(c['case_id']) if c['case_id'] else None
        j=r['judgment'] if r and r['status']=='complete' else {}
        rows.append({**c,**resolve_outcome(c,r),'simulated':simulated,
            'judge_reason':j.get('reason',''),'judge_confidence':j.get('confidence','')})
    table,diagnostics=preference_tables(rows,simulated)
    summary={'complete':all(x['group_complete'] for x in diagnostics),'simulated':simulated,
        'judging_mode':'relative_preference','scope':meta['candidate_scope'],
        'query_count':meta['query_count'],'planned_comparisons':len(rows),
        'same_image_skipped':meta['same_image_skipped'],'gpt_case_count':len(cases),
        'completed_gpt_cases':sum(r['status']=='complete' for r in results.values()),
        'failed_gpt_cases':sum(r['status']=='failed' for r in results.values()),
        'pending_gpt_cases':len(cases)-len(results),'swapped_order_repeats':0,
        'duplicate_unordered_pairs_reused':meta['duplicate_unordered_pairs_reused'],
        'denominator':'All completed different-image comparisons per epoch and rank; includes low-confidence forced choices. Same-image skips excluded. Percentages shown only for complete groups.',
        'table':table,'diagnostics':diagnostics,
        'limits':['Forced A/B relative preference is not absolute instruction correctness',
                  'Low-confidence choices are retained, not filtered to favor either model',
                  'Same-image comparisons are skipped and do not enter preference denominators',
                  'GPT blind judgments are not human-adjudicated ground truth',
                  'Epoch 1 uses initial.pt; epoch 4 uses checkpoint_epoch_003.pt',
                  'Different query samples across epochs do not establish within-query improvement',
                  'Similarity scores and model identity were hidden from GPT']}
    write_json(out/'summary.json',summary)
    write_csv(out/'summary_by_epoch_rank.csv',table)
    write_csv(out/'summary_diagnostics_by_epoch_rank.csv',diagnostics)
    write_csv(out/'comparison_results.csv',rows)
    write_jsonl(out/'final.jsonl',[results[k] for k in sorted(results)])
    export_html(out,rows,results,summary)
    return summary
