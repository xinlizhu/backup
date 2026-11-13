"""Exploratory single-target visual audit; separate from QDiag's multi-positive validator."""
from collections import Counter,defaultdict
import concurrent.futures
import csv
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import threading
import time
from typing import Literal

from pydantic import BaseModel,ConfigDict,Field
import yaml
from cir_qdiag.models.base import FatalProviderError
from cir_qdiag.models.client import StructuredVLMClient
from cir_qdiag.models.openai_chat import OpenAIChatBackend

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parents[2]/'cir_qdiag_openai'
CELLS=list(itertools.product(['add','substitute','delete'],['object','attribute','predicate']))


class Judgment(BaseModel):
    model_config=ConfigDict(extra='forbid')
    case_id: str=Field(min_length=1)
    decision: Literal['accept','reject','uncertain']
    reason: str=Field(min_length=1)


def read_json(path):return json.loads(Path(path).read_text())
def read_jsonl(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+f'.tmp.{threading.get_ident()}')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def write_jsonl(path,rows):
    Path(path).write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def redact(value):
    key=os.environ.get('OPENAI_API_KEY','');value=str(value)
    return value.replace(key,'[REDACTED]') if key else value
def load_config(path,output=None):
    c=yaml.safe_load(Path(path).read_text())
    c['source_audit_dir']=str((ROOT/c['source_audit_dir']).resolve())
    c['runtime']['output_dir']=str(Path(output).resolve() if output else (ROOT/c['runtime']['output_dir']).resolve())
    if c['runtime']['max_workers']<1:raise ValueError('max_workers must be positive')
    if len(c['models'])!=len(set(c['models'])):raise ValueError('Duplicate model names')
    return c


def choose_target(group,seed):
    by_id={im['image_id']:im for im in group['positive_images']}
    ids=group['metadata']['validated_positive_image_ids']
    if not ids or any(k not in by_id for k in ids):raise ValueError('Missing original reviewed positives')
    chosen=min(ids,key=lambda k:digest([seed,'single_target',group['group_id'],k]))
    return by_id[chosen]


def build_cases(sample,groups,seed):
    cases=[]
    for edit in sample:
        group=groups[edit['group_id']];refs=group['references'];target=choose_target(group,seed)
        case_id='case_'+digest([edit['negative_id'],target['image_id']])[:24]
        cases.append({'case_id':case_id,'negative_id':edit['negative_id'],'group_id':edit['group_id'],
            'instruction_id':edit['instruction_id'],'edit_operation':edit['edit_operation'],
            'semantic_target':edit['semantic_target'],'reference_image_ids':[r['image_id'] for r in refs],
            'target_image_id':target['image_id'],'images':[r['image_path'] for r in refs]+[target['image_path']],
            'modification_text':edit['source_text'],'edited_text':edit['text'],
            'image_layout':{'reference_images':list(range(1,len(refs)+1)),'target_image':len(refs)+1}})
    if len({c['case_id'] for c in cases})!=len(cases):raise ValueError('Duplicate cases')
    random.Random(seed).shuffle(cases)
    return cases


def payload_for(case):
    return {k:case[k] for k in ['case_id','modification_text','edited_text','image_layout']}


def prepare(config):
    out=Path(config['runtime']['output_dir']);out.mkdir(parents=True,exist_ok=True)
    src=Path(config['source_audit_dir']);meta=read_json(src/'preparation.json')
    for name,expected in meta['artifacts'].items():
        if sha(src/name)!=expected:raise ValueError('Source artifact changed: '+name)
    identity={'source_signature':meta['signature'],'source_sample_sha256':sha(src/'sampled_edits.private.jsonl'),
        'source_groups_sha256':sha(src/'review_groups.private.jsonl'),'seed':config['sampling_seed'],
        'protocol':'single_target_simple_v1','prompt_sha256':sha(ROOT/'prompts/review.txt'),
        'target_rule':'minimum sha256(seed,single_target,group_id,image_id), among originally reviewed positives; shared across edits and models'}
    marker=out/'preparation.json'
    if marker.exists():
        old=read_json(marker)
        if old['identity']!=identity:raise ValueError('Protocol/input changed; use a new output directory')
        for name,expected in old['artifacts'].items():
            if sha(out/name)!=expected:raise ValueError('Prepared artifact changed: '+name)
        for p,st in read_json(out/'image_inventory.json').items():
            actual=Path(p).stat()
            if [actual.st_size,actual.st_mtime_ns]!=st:raise ValueError('Image changed: '+p)
        return old
    sample=read_jsonl(src/'sampled_edits.private.jsonl')
    groups={g['group_id']:g for g in read_jsonl(src/'review_groups.private.jsonl')}
    counts=Counter((n['edit_operation'],n['semantic_target']) for n in sample)
    if len(sample)!=450 or set(counts)!=set(CELLS) or set(counts.values())!={50}:
        raise ValueError('Expected the frozen 450-edit, nine-cell sample')
    cases=build_cases(sample,groups,config['sampling_seed'])
    paths={p for c in cases for p in c['images']}
    inventory={p:[Path(p).stat().st_size,Path(p).stat().st_mtime_ns] for p in paths}
    write_jsonl(out/'cases.private.jsonl',cases);write_json(out/'image_inventory.json',inventory)
    (out/'review_prompt.txt').write_bytes((ROOT/'prompts/review.txt').read_bytes())
    names=['cases.private.jsonl','image_inventory.json','review_prompt.txt']
    result={'identity':identity,'signature':digest(identity),'sampled_edits':len(cases),'per_cell':50,
        'groups':len({c['group_id'] for c in cases}),'reference_count_by_edit':dict(Counter(len(c['images'])-1 for c in cases)),
        'models':config['models'],'planned_calls':len(cases)*len(config['models']),
        'limits':['This is a new single-target semantic check, not a replication of the original all-supplied-positives rule.',
            'No semantic-axis labels, span anchors, old decisions, or other candidate edits are shown to the judge.',
            'Necessary reference images are retained; a two-reference query receives three images total.',
            'The target is an annotated positive; this protocol focuses on the edited condition rather than revalidating the source label.',
            'The sample was selected from original accepted edits. Rates are conditional acceptance, not annotation accuracy.',
            'Model responses depend on the supplied API deployment; matching model names alone do not verify upstream model identity.'],
        'artifacts':{name:sha(out/name) for name in names}}
    write_json(marker,result);return result


def backend_for(config,model):
    a=config['api']
    return OpenAIChatBackend(api_base=a['api_base'],api_key=a['api_key'],generation_model=model,selection_model=model,
        generation_reasoning_effort=None,selection_reasoning_effort=None,
        structured_output_mode=a['structured_output_mode'],image_detail=a['image_detail'],timeout_seconds=a['timeout_seconds'])


def judge_one(case,client,config,model,signature,stop):
    if stop.is_set():return None
    out=Path(config['runtime']['output_dir'])/'models'/model
    base={'case_id':case['case_id'],'negative_id':case['negative_id'],'model':model,'signature':signature,'simulated':False}
    try:
        for _ in range(client.max_retries+1):
            answer,trace=client.call(Judgment,images=case['images'],
                system_prompt=(Path(config['runtime']['output_dir'])/'review_prompt.txt').read_text().strip(),
                user_prompt=json.dumps(payload_for(case),sort_keys=True,ensure_ascii=False,separators=(',',':')),
                stage='single_target_review',max_output_tokens=config['max_output_tokens'])
            if answer.case_id==case['case_id']:break
            client.reject_cached_response(trace,'Wrong case_id')
        else:raise ValueError('Repeated wrong case_id')
        result={**base,'status':'complete','judgment':answer.model_dump(mode='json'),'trace':trace.model_dump(mode='json')}
    except Exception as exc:
        result={**base,'status':'failed','error':redact(exc)}
        if isinstance(exc,FatalProviderError):stop.set()
    write_json(out/'judgments'/f"{case['case_id']}.json",result);return result


def run(config,models=None,max_cases=None):
    meta=prepare(config);out=Path(config['runtime']['output_dir']);models=models or config['models']
    if set(models)-set(config['models']):raise ValueError('Unknown model')
    cases=read_jsonl(out/'cases.private.jsonl')
    with (out/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        clients={};pending={};signatures={};stops={m:threading.Event() for m in models}
        try:
            for model in models:
                backend=backend_for(config,model)
                clients[model]=StructuredVLMClient(backend,out/'models'/model/'cache',
                    prompt_version='single_target_simple_v1',max_retries=config['api']['max_retries'])
                identity={'prepared_signature':meta['signature'],'backend':backend.cache_identity(),
                    'max_output_tokens':config['max_output_tokens'],'code_sha256':sha(__file__),
                    'backend_code_sha256':{n:sha(SOURCE/'src/cir_qdiag/models'/n) for n in ['openai_chat.py','client.py']}}
                marker=out/'models'/model/'identity.json'
                if marker.exists() and read_json(marker)!=identity:raise ValueError('Model/run identity changed; use a new output directory')
                write_json(marker,identity);signatures[model]=digest(identity)
                todo=[]
                for case in cases:
                    p=out/'models'/model/'judgments'/f"{case['case_id']}.json"
                    if p.exists():
                        old=read_json(p)
                        if old['signature']!=signatures[model]:raise ValueError('Case signature mismatch')
                        if old['status']=='complete':continue
                    todo.append(case)
                pending[model]=todo[:max_cases] if max_cases is not None else todo
            # Interleave models so both start promptly under the global worker limit.
            jobs=[]
            for index in range(max((len(v) for v in pending.values()),default=0)):
                for model in models:
                    if index<len(pending[model]):jobs.append((pending[model][index],model))
            done=0;failed=0;started=time.time()
            def status(phase,**extra):
                value={'phase':phase,'pid':os.getpid(),'models':models,'edits_per_model':len(cases),
                    'pending_this_invocation':len(jobs),'processed_this_invocation':done,'failed_this_invocation':failed,
                    'elapsed_seconds':time.time()-started,**extra}
                write_json(out/'status.json',value);print(json.dumps(value),flush=True)
            status('reviewing')
            with concurrent.futures.ThreadPoolExecutor(max_workers=config['runtime']['max_workers']) as pool:
                futures=[pool.submit(judge_one,c,clients[m],config,m,signatures[m],stops[m]) for c,m in jobs]
                for future in concurrent.futures.as_completed(futures):
                    result=future.result()
                    if result is None:continue
                    done+=1;failed+=result['status']=='failed';status('reviewing',last_model=result['model'],last_case=result['case_id'])
            summary=summarize(config);status('complete' if summary['complete'] else 'partial');return summary
        finally:
            for client in clients.values():client.backend.client.close()


def rates(values,total):
    complete=len(values)==total;counts=Counter(values)
    return {'sampled':total,'reviewed':len(values),'accepted':counts['accept'],'rejected':counts['reject'],
        'uncertain':counts['uncertain'],'complete':complete,
        **{f'{k}_percent':100*counts[k]/total if total and complete else None for k in ['accept','reject','uncertain']}}


def summarize(config):
    out=Path(config['runtime']['output_dir']);meta=read_json(out/'preparation.json')
    cases=read_jsonl(out/'cases.private.jsonl');by_id={c['case_id']:c for c in cases}
    decisions={};rows=[];errors={};overall=[];per_cell=[]
    for model in config['models']:
        decisions[model]={};errors[model]=[]
        marker=out/'models'/model/'identity.json';identity=read_json(marker) if marker.exists() else None
        for p in (out/'models'/model/'judgments').glob('*.json'):
            r=read_json(p)
            if identity is None or r['signature']!=digest(identity) or r['simulated']:raise ValueError('Invalid result identity')
            if r['case_id'] not in by_id:raise ValueError('Unknown result case')
            if r['status']!='complete':errors[model].append({'case_id':r['case_id'],'error':r['error']});continue
            j=Judgment.model_validate(r['judgment']);case=by_id[r['case_id']]
            if j.case_id!=case['case_id']:raise ValueError('Wrong result case_id')
            decisions[model][j.case_id]=j.decision
            rows.append({'model':model,**{k:case[k] for k in ['case_id','negative_id','group_id','edit_operation','semantic_target',
                'target_image_id','modification_text','edited_text']},'decision':j.decision,'reason':j.reason})
        overall.append({'model':model,**rates(list(decisions[model].values()),len(cases))})
        for operation,axis in CELLS:
            ids=[c['case_id'] for c in cases if c['edit_operation']==operation and c['semantic_target']==axis]
            values=[decisions[model][k] for k in ids if k in decisions[model]]
            per_cell.append({'model':model,'operation':operation,'semantic_target':axis,**rates(values,len(ids))})
    pairs=[]
    for a,b in itertools.combinations(config['models'],2):
        shared=set(decisions[a])&set(decisions[b]);complete=len(shared)==len(cases)
        decisive=[k for k in shared if 'uncertain' not in [decisions[a][k],decisions[b][k]]]
        pairs.append({'model_a':a,'model_b':b,'shared_reviewed':len(shared),'complete':complete,
            'agreement_percent_all_decisions':100*sum(decisions[a][k]==decisions[b][k] for k in shared)/len(cases) if complete else None,
            'both_decisive':len(decisive),'agreement_percent_both_decisive':100*sum(decisions[a][k]==decisions[b][k] for k in decisive)/len(decisive) if complete and decisive else None})
    result={'complete':all(row['complete'] for row in overall),'protocol':'single_target_simple_v1',
        'sampled_edits':len(cases),'overall':overall,'per_cell':per_cell,'pairwise_agreement':pairs,'errors':errors,'limits':meta['limits']}
    write_json(out/'summary.json',result);write_csv(out/'overall_by_model.csv',overall)
    write_csv(out/'summary_by_cell.csv',per_cell);write_csv(out/'pairwise_agreement.csv',pairs);write_csv(out/'judgment_details.csv',rows)
    matrix=[{**{k:c[k] for k in ['case_id','negative_id','group_id','edit_operation','semantic_target','target_image_id']},
        **{m:decisions[m].get(c['case_id'],'pending') for m in config['models']}} for c in cases]
    write_csv(out/'decision_matrix.csv',matrix)
    return result
