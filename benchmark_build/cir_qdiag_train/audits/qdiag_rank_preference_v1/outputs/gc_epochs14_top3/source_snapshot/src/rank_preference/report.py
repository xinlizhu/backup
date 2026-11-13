from collections import Counter,defaultdict
import csv
import html
import json
import os
from pathlib import Path
from urllib.parse import quote
from rank_preference.common import read_json,read_jsonl,write_json,write_jsonl


def resolve_outcome(comparison,result):
    if comparison['same_image_skipped']:
        return {'outcome':'same_image_skipped','teacher_status':'not_assessed','student_status':'not_assessed'}
    if result is None:return {'outcome':'pending','teacher_status':'not_assessed','student_status':'not_assessed'}
    if result['status']!='complete':return {'outcome':'failed','teacher_status':'not_assessed','student_status':'not_assessed'}
    j=result['judgment'];decision=j['decision']
    if decision in ['A_only','B_only']:
        outcome='teacher_only' if decision[0]==comparison['teacher_side'] else 'student_only'
    else:outcome=decision
    return {'outcome':outcome,'teacher_status':j[comparison['teacher_side']+'_status'],
            'student_status':j[comparison['student_side']+'_status']}


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def summarize(config):
    out=Path(config['runtime']['output_dir'])
    meta=read_json(out/'prepared_metadata.json')
    comparisons=read_jsonl(out/'comparisons.private.jsonl')
    cases={x['case_id']:x for x in read_jsonl(out/'judge_cases.private.jsonl')}
    results={p.stem:read_json(p) for p in (out/'judgments').glob('*.json')}
    if set(results)-set(cases):raise ValueError('Unexpected judgment IDs in output')
    simulated=any(r.get('simulated',False) for r in results.values())
    ident=out/'judgment_identity.json'
    if ident.exists():simulated=read_json(ident)['mock']
    rows=[]
    for c in comparisons:
        r=results.get(c['case_id']) if c['case_id'] else None
        row={**c,**resolve_outcome(c,r),'simulated':simulated,
             'judge_reason':r['judgment']['reason'] if r and r['status']=='complete' else ''}
        rows.append(row)
    groups=defaultdict(list)
    for row in rows:groups[row['epoch'],row['rank']].append(row)
    table=[]
    for (epoch,rank),group in sorted(groups.items()):
        counts=Counter(x['outcome'] for x in group)
        decisive=counts['teacher_only']+counts['student_only']
        table.append({'epoch':epoch,'student_checkpoint':Path(group[0]['student_checkpoint']).name,'rank':rank,
            'query_count':len(group),'same_image_skipped':counts['same_image_skipped'],
            'teacher_only':counts['teacher_only'],'student_only':counts['student_only'],
            'both_satisfy':counts['both'],'neither_satisfies':counts['neither'],'uncertain':counts['uncertain'],
            'pending':counts['pending'],'failed':counts['failed'],
            'teacher_supported_on_judged_subset':sum(x['teacher_status']=='supported' for x in group),
            'student_supported_on_judged_subset':sum(x['student_status']=='supported' for x in group),
            'decisive_comparisons':decisive,
            'teacher_share_of_decisive_percent':100*counts['teacher_only']/decisive if decisive and not simulated else None,
            'teacher_score_tied_at_rank':sum(x['teacher_tie_size']>1 for x in group),
            'student_score_tied_at_rank':sum(x['student_tie_size']>1 for x in group),
            'simulated':simulated})
    complete=all(x['outcome'] not in ['pending','failed'] for x in rows)
    summary={'complete':complete,'simulated':simulated,'scope':meta['candidate_scope'],
        'query_count':meta['query_count'],'planned_comparisons':len(rows),'same_image_skipped':meta['same_image_skipped'],
        'gpt_case_count':len(cases),'completed_gpt_cases':sum(r['status']=='complete' for r in results.values()),
        'failed_gpt_cases':sum(r['status']=='failed' for r in results.values()),
        'pending_gpt_cases':len(cases)-len(results),'swapped_order_repeats':0,'table':table,
        'limits':['Same-image comparisons are skipped and never automatically labeled correct',
                  'Instruction satisfaction counts apply only to GPT-judged differing-image cases',
                  'GPT blind judgments are not human-adjudicated ground truth',
                  'Epoch 1 uses initial.pt; epoch 4 uses checkpoint_epoch_003.pt',
                  'Different query samples across epochs do not establish within-query improvement',
                  'Similarity scores and model identity were hidden from GPT'],
        'duplicate_unordered_pairs_reused':meta['duplicate_unordered_pairs_reused']}
    write_json(out/'summary.json',summary)
    write_csv(out/'summary_by_epoch_rank.csv',table)
    write_csv(out/'comparison_results.csv',rows)
    write_jsonl(out/'final.jsonl',[results[k] for k in sorted(results)])
    export_html(out,rows,results,summary)
    return summary


def export_html(out,rows,results,summary):
    esc=lambda x:html.escape(str(x))
    css='<style>body{font:16px system-ui;max-width:1600px;margin:24px}img{height:230px;max-width:100%;object-fit:contain}.grid{display:grid;grid-template-columns:repeat(3,minmax(250px,1fr));gap:14px}.case{border:1px solid #bbb;padding:18px;margin:24px 0}pre{white-space:pre-wrap}td,th{border:1px solid #ccc;padding:8px}table{border-collapse:collapse}</style>'
    fields=['epoch','rank','query_count','same_image_skipped','teacher_only','student_only','both_satisfy','neither_satisfies','uncertain','pending','failed']
    parts=['<!doctype html><meta charset="utf-8">',css,'<h1>教师／学生 Top-1、2、3 指令符合度盲审</h1>',
           '<p>同图跳过，不能视为都正确。GPT 不接收模型身份、分数、排名或 target 标签；本页身份信息仅供事后查看。</p>']
    if summary['simulated']:parts+=['<h2>MOCK：仅测试，非实验结果</h2>']
    parts+=['<table><tr>'+''.join('<th>'+f+'</th>' for f in fields)+'</tr>']
    for row in summary['table']:parts+=['<tr>'+''.join('<td>'+esc(row[f])+'</td>' for f in fields)+'</tr>']
    parts+=['</table>']
    for row in rows:
        parts+=['<div class="case"><h2>'+esc(f"Epoch {row['epoch']} · {row['audit_query_id']} · Rank {row['rank']}")+'</h2>',
                '<p>'+esc(row['source_instruction'])+'</p><div class="grid">']
        for label,path in [('Reference',row['reference_path']),('Teacher',row['teacher_image_path']),('Student',row['student_image_path'])]:
            src=quote(os.path.relpath(path,out),safe='/')
            parts+=['<div>'+label+'<br><img loading="lazy" src="'+src+'"></div>']
        parts+=['</div><p><b>'+esc(row['outcome'])+'</b> '+esc(row['judge_reason'])+'</p>']
        result=results.get(row['case_id']) if row['case_id'] else None
        parts+=['<details><summary>分数、随机顺序、逐条件判断及来源</summary><pre>'+esc(json.dumps({'comparison':row,'result':result},ensure_ascii=False,indent=2))+'</pre></details></div>']
    (out/'audit.html').write_text(''.join(parts))
