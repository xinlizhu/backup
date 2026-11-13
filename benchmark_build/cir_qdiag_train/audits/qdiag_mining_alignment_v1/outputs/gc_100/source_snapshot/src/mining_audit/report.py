from collections import Counter,defaultdict
import csv
import html
import json
import os
from pathlib import Path
from urllib.parse import quote
import numpy as np
from mining_audit.common import read_json,read_jsonl,write_json,write_jsonl

CONFLICT_TYPES={'substitution_contrast','deletion_relaxation'}


def csv_write(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def uncertainty(result):
    if result.get('status')!='complete':return True
    review=result['review']
    return any(review[k]['status']=='uncertain' for k in ['source_on_positive','source_on_candidate']) or any(v['decision']=='uncertain' for v in review['verdicts'])


def summarize(config):
    out=Path(config['runtime']['output_dir'])
    queries=read_jsonl(out/'queries.private.jsonl')
    cases={x['case_id']:x for x in read_jsonl(out/'cases.private.jsonl')}
    members=read_jsonl(out/'memberships.private.jsonl')
    results={}
    for p in (out/'results').glob('*.json'):
        value=read_json(p)
        if value['case_id'] not in cases:raise ValueError('Unexpected case result')
        results[value['case_id']]=value
    simulated=any(x.get('simulated',False) for x in results.values())
    identity=out/'annotation_identity.json'
    if identity.exists():simulated=read_json(identity).get('mock',False)
    counts=Counter(x['status'] for x in results.values())
    complete=counts['complete']==len(cases)
    rows=[];grouped=defaultdict(list);accepted_rows=[]
    for member in members:
        result=results.get(member['case_id'],{'status':'pending'})
        accepted=result.get('accepted_witnesses',[])
        types={w['witness_type'] for w in accepted}
        operations={w['operation'] for w in accepted}
        row={**member,'status':result['status'],'simulated':simulated,
            'confirmed_conflict':bool(types&CONFLICT_TYPES),'addition_disambiguation':'addition_disambiguation' in types,
            'accepted_operations':'|'.join(sorted(operations)), 'uncertain_or_unavailable':uncertainty(result)}
        rows.append(row);grouped[member['origin'],member['audit_query_id']].append(row)
        for witness in accepted:
            accepted_rows.append({**member,'witness_id':witness['witness_id'],'operation':witness['operation'],
                'semantic_target':witness['semantic_target'],'witness_type':witness['witness_type'],
                'source_instruction':cases[member['case_id']]['source_instruction'],
                'before_span':witness['before_span'],'after_span':witness['after_span'],
                'edited_instruction':witness['edited_instruction'],'review_reason':witness['independent_review']['reason'],
                'simulated':simulated})
    query_rows=[]
    for (origin,qid),items in sorted(grouped.items()):
        known=[x for x in items if x['status']=='complete']
        n=sum(x['confirmed_conflict'] for x in items)
        pending=sum(x['status']!='complete' for x in items)
        query_rows.append({'origin':origin,'audit_query_id':qid,'epoch':items[0]['epoch'],'candidate_count':len(items),
            'complete_pairs':len(known),'pending_or_failed_pairs':pending,'confirmed_conflict_candidates':n,
            'query_has_confirmed_conflict':bool(n),'query_may_gain_confirmed_conflict':bool(n or pending),
            'addition_candidates':sum(x['addition_disambiguation'] for x in items),
            'uncertain_or_unavailable_pairs':sum(x['uncertain_or_unavailable'] for x in items)})
    rng=np.random.default_rng(config['runtime']['bootstrap_seed'])
    replicates=config['runtime']['bootstrap_replicates']
    sample_indices=np.concatenate([rng.integers(e*25,(e+1)*25,size=(replicates,25)) for e in range(4)],axis=1)
    per_arm={};bootstrap={}
    for origin in sorted({m['origin'] for m in members}):
        arms=[x for x in rows if x['origin']==origin]
        by_q={x['audit_query_id']:x for x in query_rows if x['origin']==origin}
        ordered=[by_q[q['audit_query_id']] for q in queries]
        assert len(ordered)==100 and all(x['candidate_count']==31 for x in ordered)
        coverage=np.array([x['query_has_confirmed_conflict'] for x in ordered],dtype=float)
        frequency=np.array([x['confirmed_conflict_candidates']/31 for x in ordered])
        accepted=[x for x in arms if x['confirmed_conflict']]
        arm_complete=all(x['status']=='complete' for x in arms)
        stats={'queries':100,'candidate_pairs':len(arms),'completed_pairs':sum(x['status']=='complete' for x in arms),
            'confirmed_queries':int(coverage.sum()),'confirmed_candidate_pairs':len(accepted),
            'uncertain_or_unavailable_pairs':sum(x['uncertain_or_unavailable'] for x in arms),
            'compatible_addition_pairs':sum(x['addition_disambiguation'] for x in arms),
            'confirmed_query_coverage_percent':None if simulated else float(100*coverage.mean()),
            'confirmed_candidate_frequency_percent':None if simulated else float(100*frequency.mean()),
            'interpretation':'Observed independently confirmed SUB/DELETE witnesses; no-witness results do not prove absence. Partial runs give confirmed lower bounds, not final frequencies.',
            'complete':arm_complete}
        for field in ['teacher_margin_saved','teacher_margin_recomputed_fp32']:
            margins=np.array([x[field] for x in accepted if x[field] is not None])
            stats[field]={'n':len(margins),'positive_count':int((margins>0).sum()),'tie_count':int((margins==0).sum()),
                'negative_count':int((margins<0).sum()),
                'positive_rate_percent':float(100*(margins>0).mean()) if len(margins) and not simulated else None,
                'mean':float(margins.mean()) if len(margins) and not simulated else None}
        if arm_complete and not simulated:
            bootstrap[origin]={'query':coverage[sample_indices].mean(1)*100,'candidate':frequency[sample_indices].mean(1)*100}
            stats['query_coverage_95CI_percent']=np.quantile(bootstrap[origin]['query'],[.025,.975]).tolist()
            stats['candidate_frequency_95CI_percent']=np.quantile(bootstrap[origin]['candidate'],[.025,.975]).tolist()
        stats['accepted_operation_counts_nonexclusive']={op:sum(op in x['accepted_operations'].split('|') for x in arms) for op in ['add','delete','substitute']}
        per_arm[origin]=stats
    summary={'complete':complete,'simulated':simulated,'source':'Recorded GC head-ablation candidates; not the original RIM history',
        'unique_pairs':len(cases),'completed_unique_pairs':counts['complete'],'failed_unique_pairs':counts['failed'],
        'pending_unique_pairs':len(cases)-sum(counts.values()),'arms':per_arm,
        'CI_scope':'Sampling variability of confirmed witness detection; not annotation accuracy or causal learning',
        'model_review':'Fresh independent calls; generation and review may use the same model. Human validation is still not recorded.',
        'manual_review_status':'not_performed'}
    if 'mined' in bootstrap and 'random' in bootstrap:
        summary['paired_mined_minus_random_95CI_pp']={k:np.quantile(bootstrap['mined'][k]-bootstrap['random'][k],[.025,.975]).tolist() for k in ['query','candidate']}
    write_json(out/'summary.json',summary)
    csv_write(out/'candidate_audit.csv',rows)
    csv_write(out/'query_audit.csv',query_rows)
    cells=[]
    for origin in per_arm:
        for operation in ['add','delete','substitute']:
            for axis in ['object','attribute','predicate']:
                subset=[x for x in accepted_rows if x['origin']==origin and x['operation']==operation and x['semantic_target']==axis]
                cells.append({'origin':origin,'operation':operation,'semantic_target':axis,
                    'confirmed_distinct_pairs':len({x['case_id'] for x in subset}),
                    'confirmed_distinct_queries':len({x['audit_query_id'] for x in subset}),
                    'total_pairs':per_arm[origin]['candidate_pairs'],'total_queries':100,
                    'simulated':simulated,'note':'Nonexclusive categories; do not sum across cells'})
    csv_write(out/'operation_axis_counts.csv',cells)
    if accepted_rows:csv_write(out/'accepted_witnesses.csv',accepted_rows)
    else:(out/'accepted_witnesses.csv').write_text('case_id,operation,semantic_target,witness_type,edited_instruction\n')
    write_jsonl(out/'final.jsonl',[results[k] for k in sorted(results)])
    export_html(out,queries,cases,members,results,summary)
    return summary


def export_html(out,queries,cases,members,results,summary):
    """Local linked image review pages: no resized or generated scientific images."""
    pages=out/'audit_queries';pages.mkdir(exist_ok=True)
    esc=lambda x:html.escape(str(x))
    css='<style>body{font:16px system-ui;margin:24px;max-width:1600px}img{max-width:100%;height:200px;object-fit:contain}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:16px}.card{border:1px solid #ccc;padding:12px}pre{white-space:pre-wrap;font:13px monospace}h2{margin-top:32px}</style>'
    def img(path):
        return '<img loading="lazy" src="'+quote(os.path.relpath(path,pages),safe='/')+'">'
    by_query=defaultdict(list)
    for m in members:by_query[m['audit_query_id']].append(m)
    links=[]
    for q in queries:
        parts=['<!doctype html><meta charset="utf-8">',css,'<a href="../audit.html">返回汇总</a>',
            '<h1>'+esc(q['audit_query_id'])+'</h1>','<p>'+esc(q['source_instruction'])+'</p>',
            '<div class="grid"><div class="card">Reference<br>'+img(q['reference_path'])+'</div>',
            '<div class="card">Dataset target<br>'+img(q['target_path'])+'</div></div>',
            '<p>模型标注时不提供来源、排名和教师分数；本页仅供事后复核。'+(' MOCK：模拟联调，非实验结论。' if summary['simulated'] else '')+'</p>']
        for origin in ['mined','random']:
            group=[m for m in by_query[q['audit_query_id']] if m['origin']==origin]
            if not group:continue
            parts+=['<h2>'+esc(origin)+'</h2><div class="grid">']
            for m in group:
                case=cases[m['case_id']];r=results.get(m['case_id'],{'status':'pending'})
                parts+=['<div class="card">'+img(case['images'][2]),'<p>'+esc(m['case_id'])+' — '+esc(r['status'])+'</p>']
                for w in r.get('accepted_witnesses',[]):
                    parts+=['<p><b>'+esc(w['operation']+' / '+w['semantic_target'])+'</b>: '+esc(w['edited_instruction'])+'</p>']
                parts+=['<details><summary>判定及来源</summary><pre>'+esc(json.dumps({'membership':m,'result':r},ensure_ascii=False,indent=2))+'</pre></details></div>']
            parts+=['</div>']
        (pages/(q['audit_query_id']+'.html')).write_text(''.join(parts))
        links.append('<li><a href="audit_queries/'+q['audit_query_id']+'.html">'+esc(q['audit_query_id'])+'</a> '+esc(q['source_instruction'])+'</li>')
    title='MOCK — 非实验结果' if summary['simulated'] else 'GC 候选与 QDiag 单编辑语义对照审计'
    (out/'audit.html').write_text('<!doctype html><meta charset="utf-8">'+css+'<h1>'+title+'</h1><pre>'+esc(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre><ol>'+''.join(links)+'</ol>')
