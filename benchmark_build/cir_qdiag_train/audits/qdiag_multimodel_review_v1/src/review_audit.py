"""Frozen, accepted-only cross-model audit using the existing QDiag V7 reviewer."""
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
import yaml

from cir_qdiag.models.base import FatalProviderError
from cir_qdiag.models.client import StructuredVLMClient
from cir_qdiag.models.openai_chat import OpenAIChatBackend,_decode_http_response,_decode_response_container
from cir_qdiag.schemas import GroupReviewResponse,PinPointGroupRecord
from cir_qdiag.utils.prompts import dump_prompt_json,load_prompt
from cir_qdiag.validation.batch_review import review_group_candidates
from review_compat import FORMAT_POLICY,RetainedReasonReview,ReviewClient,reason_warnings,recover_reason_only_errors

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parents[2]/'cir_qdiag_openai'
CELLS=list(itertools.product(['add','substitute','delete'],['object','attribute','predicate']))


def read_json(path):return json.loads(Path(path).read_text())
def read_jsonl(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+f'.tmp.{threading.get_ident()}')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def write_jsonl(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def redact(value):
    text=str(value);key=os.environ.get('OPENAI_API_KEY','')
    return text.replace(key,'[REDACTED]') if key else text
def load_config(path,output=None):
    c=yaml.safe_load(Path(path).read_text())
    c['runtime']['output_dir']=str(Path(output).resolve() if output else (ROOT/c['runtime']['output_dir']).resolve())
    if c['sampling'].get('subset_of'):
        c['sampling']['subset_of']=str((ROOT/c['sampling']['subset_of']).resolve())
    if c['sampling']['population']!='accepted_edits' or c['review']['context']!='sampled_candidates_only':
        raise ValueError('This protocol only samples accepted edits and reviews sampled candidates')
    if c['runtime']['max_workers']<1:raise ValueError('Invalid max_workers')
    if c['review']['baseline_model'] in c['review']['models']:raise ValueError('Baseline is historical and must not be rerun here')
    return c


def sample_edits(rows,per_cell,seed,parent_sample=None):
    buckets=defaultdict(list)
    for r in rows:
        for n in r['negatives']:
            buckets[n['edit_operation'],n['semantic_target']].append({'group_id':r['group_id'],**n})
    parent_buckets=defaultdict(list)
    if parent_sample is not None:
        population_by_id={n['negative_id']:n for bucket in buckets.values() for n in bucket}
        if len({n['negative_id'] for n in parent_sample})!=len(parent_sample):raise ValueError('Duplicate parent sample IDs')
        for n in parent_sample:
            if population_by_id.get(n['negative_id'])!=n:raise ValueError('Parent sample differs from accepted release')
            parent_buckets[n['edit_operation'],n['semantic_target']].append(n)
    selected=[]
    for cell in CELLS:
        population=sorted(buckets[cell],key=lambda n:n['negative_id'])
        if len(population)<per_cell:raise ValueError(f'Insufficient accepted edits in {cell}')
        if parent_sample is not None:
            if len(parent_buckets[cell])<per_cell:raise ValueError(f'Insufficient parent sample in {cell}')
            # Prefix of the previously frozen random draw; never select by model outcomes.
            selected.extend(parent_buckets[cell][:per_cell])
        else:
            rng=random.Random(int(digest([seed,*cell]),16))
            selected.extend(rng.sample(population,per_cell))
    if len({n['negative_id'] for n in selected})!=len(selected):raise ValueError('Repeated edit IDs')
    return selected,{f'{o}/{a}':len(buckets[o,a]) for o,a in CELLS}


def payload_for(record):
    refs=len(record.references);positives=len(record.metadata.validated_positive_image_ids)
    return {'group_id':record.group_id,'image_layout':{'reference_images':list(range(1,refs+1)),
        'positive_images':list(range(refs+1,refs+positives+1))},
        'instructions':[{'instruction_id':x.instruction_id,'text':x.text} for x in record.instructions],
        'candidates':[{'negative_id':n.negative_id,'instruction_id':n.instruction_id,'source_text':n.source_text,
            'negative_text':n.text,'semantic_target':n.semantic_target.value,'edit_operation':n.edit_operation.value,
            'before_span':n.before_span,'after_span':n.after_span} for n in record.candidates]}


def positives_for(record):
    by_id={im.image_id:im for im in record.positive_images}
    return [by_id[k] for k in record.metadata.validated_positive_image_ids]


def prepare(config):
    out=Path(config['runtime']['output_dir']);out.mkdir(parents=True,exist_ok=True)
    prompt_path=SOURCE/'prompts/validate_group_v7.txt'
    identity={'source_final_sha256':sha(config['source_final']),'release_final_sha256':sha(config['release_final']),
        'prompt_sha256':sha(prompt_path),'sampling':config['sampling'],'context':config['review']['context'],
        'baseline_model':config['review']['baseline_model'],
        'source_review_code_sha256':sha(SOURCE/'src/cir_qdiag/validation/batch_review.py')}
    if config['sampling'].get('subset_of'):
        identity['parent_sample_sha256']=sha(config['sampling']['subset_of'])
    marker=out/'preparation.json'
    if marker.exists():
        saved=read_json(marker)
        if saved['identity']!=identity:raise ValueError('Preparation inputs changed; use a new output directory')
        for name,expected in saved['artifacts'].items():
            if sha(out/name)!=expected:raise ValueError('Prepared artifact changed: '+name)
        for p,expected in read_json(out/'image_inventory.json').items():
            st=Path(p).stat()
            if [st.st_size,st.st_mtime_ns]!=expected:raise ValueError('Input image changed: '+p)
        return saved
    released=read_jsonl(config['release_final'])
    raw={r['group_id']:r for r in read_jsonl(config['source_final']) if r['metadata']['operational_status']=='ok'}
    if set(raw)!={r['group_id'] for r in released}:raise ValueError('Release/source groups differ')
    for r in released:
        if [(n['negative_id'],n['text']) for n in r['negatives']]!=[(n['negative_id'],n['text']) for n in raw[r['group_id']]['negatives']]:
            raise ValueError('Release/source accepted edits differ')
    parent=read_jsonl(config['sampling']['subset_of']) if config['sampling'].get('subset_of') else None
    sampled,population=sample_edits(released,config['sampling']['per_cell'],config['sampling']['seed'],parent)
    selection=defaultdict(set)
    for n in sampled:selection[n['group_id']].add(n['negative_id'])
    prompt,_=load_prompt(SOURCE/'prompts','validate_group','v7')
    groups=[];baseline=[];inventory={};verified_original=[]
    for group_id in sorted(selection):
        record=PinPointGroupRecord.model_validate(raw[group_id])
        if record.metadata.validator_model!=config['review']['baseline_model']:raise ValueError('Unexpected baseline model')
        trace=next(t for t in reversed(record.metadata.call_traces) if t.stage=='validate_group')
        cache=read_json(trace.raw_response_path)
        expected_hash=hashlib.sha256((prompt+'\n---\n'+dump_prompt_json(payload_for(record))).encode()).hexdigest()
        if expected_hash!=trace.prompt_hash:raise ValueError('Original reviewer prompt/payload cannot be reproduced: '+group_id)
        images=[im.image_path for im in [*record.references,*positives_for(record)]]
        if cache['images']!=images:raise ValueError('Original review image order changed')
        verdicts={v.negative_id:v for v in GroupReviewResponse.model_validate(cache['parsed']).verdicts}
        for n in record.candidates:
            if n.negative_id not in selection[group_id]:continue
            if n.status!='accepted' or verdicts[n.negative_id].decision!='accept':raise ValueError('Nonaccepted sampled edit')
            baseline.append({'group_id':group_id,'negative_id':n.negative_id,'model':record.metadata.validator_model,
                'decision':'accept','reason':verdicts[n.negative_id].reason,'raw_response_path':trace.raw_response_path})
        for path in images:
            stat=Path(path).stat();inventory[path]=[stat.st_size,stat.st_mtime_ns]
        obj=record.model_dump(mode='json')
        obj['candidates']=[{**n,'status':'generated','rejection_code':None,'selection_reason':None}
                           for n in obj['candidates'] if n['negative_id'] in selection[group_id]]
        obj['negatives']=[]
        obj['metadata'].update(stage='candidates',semantic_status='pending',num_generated_candidates=len(obj['candidates']),
            num_accepted_negatives=0,validator_model=None,call_traces=[],errors=[],notes=[])
        groups.append(PinPointGroupRecord.model_validate(obj).model_dump(mode='json'))
        verified_original.append({'group_id':group_id,'raw_response_path':trace.raw_response_path,
            'original_prompt_hash':trace.prompt_hash,'original_candidate_count':len(record.candidates),
            'sampled_candidate_count':len(obj['candidates']),'original_image_order_reused':True})
    if sum(len(r['candidates']) for r in groups)!=len(sampled):raise ValueError('Sample/group mismatch')
    (out/'prompts').mkdir(exist_ok=True)
    (out/'prompts/validate_group_v7.txt').write_bytes(prompt_path.read_bytes())
    write_jsonl(out/'sampled_edits.private.jsonl',sampled)
    write_jsonl(out/'review_groups.private.jsonl',groups)
    write_jsonl(out/'baseline_decisions.jsonl',baseline)
    write_json(out/'image_inventory.json',inventory)
    write_json(out/'original_request_verification.json',verified_original)
    artifacts=['prompts/validate_group_v7.txt','sampled_edits.private.jsonl','review_groups.private.jsonl',
               'baseline_decisions.jsonl','image_inventory.json','original_request_verification.json']
    value={'identity':identity,'signature':digest(identity),'population_counts':population,
        'sampled_edits':len(sampled),'sampled_groups':len(groups),'per_cell':config['sampling']['per_cell'],
        'new_models':config['review']['models'],'planned_group_calls_per_model':len(groups),
        'planned_total_group_calls':len(groups)*len(config['review']['models']),
        'interpretation':'Acceptance agreement conditional on original gpt-5.6-terra acceptance; not overall accuracy or agreement on rejected candidates. The sample is balanced across nine cells; overall acceptance is their macro-average, not weighted by full-release cell prevalence.',
        'context_limit':'Only sampled candidates are sent; original full-group instructions, review images/order, prompt and response schema are preserved. Candidate-batch context differs from the historical baseline.',
        'artifacts':{name:sha(out/name) for name in artifacts}}
    write_json(marker,value)
    return value


def backend_for(config,model):
    api=config['api']
    return OpenAIChatBackend(api_base=api['api_base'],api_key=api['api_key'],
        generation_model=model,selection_model=model,generation_reasoning_effort=None,selection_reasoning_effort=None,
        image_detail=api['image_detail'],structured_output_mode=api['structured_output_mode'],
        timeout_seconds=api['timeout_seconds'])


def new_client(config,model,cache_path,retries=None):
    recover_reason_only_errors(cache_path,write_json)
    return ReviewClient(backend_for(config,model),cache_path,prompt_version='v7',
        max_retries=config['api']['max_retries'] if retries is None else retries)


def review_one(record,client,config,signature,model,stop):
    if stop.is_set():return None
    out=Path(config['runtime']['output_dir'])/'models'/model
    base={'group_id':record.group_id,'model':model,'signature':signature,'simulated':False}
    try:
        reviewed,traces=review_group_candidates(record,client,Path(config['runtime']['output_dir'])/'prompts',
            'v7',config['review']['max_output_tokens'],positives_for(record))
        value={**base,'status':'complete','verdicts':[{'negative_id':n.negative_id,
            'decision':'accept' if n.status=='accepted' else 'reject',
            'failure_code':n.rejection_code.value if n.rejection_code else None,'reason':n.selection_reason} for n in reviewed],
            'traces':[t.model_dump(mode='json') for t in traces]}
        value['format_warnings']=reason_warnings(value['verdicts'])
    except Exception as exc:
        value={**base,'status':'failed','error':redact(exc)}
        if isinstance(exc,FatalProviderError):stop.set()
    write_json(out/'groups'/f'{record.group_id}.json',value)
    return value


def run(config,models=None,max_groups=None):
    meta=prepare(config);out=Path(config['runtime']['output_dir'])
    models=models or config['review']['models']
    if set(models)-set(config['review']['models']):raise ValueError('Unknown review model')
    groups=[PinPointGroupRecord.model_validate(r) for r in read_jsonl(out/'review_groups.private.jsonl')]
    with (out/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        clients={};jobs=[];stops={m:threading.Event() for m in models};done=0;failed=0;start=time.time()
        try:
            for model in models:
                client=new_client(config,model,out/'models'/model/'cache');clients[model]=client
                identity={'prepared_signature':meta['signature'],'backend':client.backend.cache_identity(),
                    'max_output_tokens':config['review']['max_output_tokens'],'audit_code_sha256':sha(__file__),
                    'format_policy':FORMAT_POLICY,'compatibility_code_sha256':sha(ROOT/'src/review_compat.py'),
                    'source_code_sha256':{name:sha(SOURCE/'src/cir_qdiag'/name) for name in
                        ['models/openai_chat.py','models/client.py','validation/batch_review.py','schemas.py']}}
                marker=out/'models'/model/'identity.json'
                if marker.exists() and read_json(marker)!=identity:raise ValueError('Model/run identity changed; use a new output directory')
                write_json(marker,identity);signature=digest(identity)
                pending=[]
                for group in groups:
                    p=out/'models'/model/'groups'/f'{group.group_id}.json'
                    if p.exists():
                        previous=read_json(p)
                        if previous['signature']!=signature:raise ValueError('Group identity mismatch')
                        if previous['status']=='complete':continue
                    pending.append(group)
                if max_groups is not None:pending=pending[:max_groups]
                jobs.extend((g,client,config,signature,model,stops[model]) for g in pending)
            def status(phase,**extra):
                value={'phase':phase,'pid':os.getpid(),'sampled_edits':meta['sampled_edits'],'groups_per_model':len(groups),
                    'models':models,'pending_calls_this_invocation':len(jobs),'processed_this_invocation':done,
                    'failed_this_invocation':failed,'elapsed_seconds':time.time()-start,**extra}
                write_json(out/'status.json',value);print(json.dumps(value),flush=True)
            status('reviewing')
            with concurrent.futures.ThreadPoolExecutor(max_workers=config['runtime']['max_workers']) as pool:
                futures=[pool.submit(review_one,*job) for job in jobs]
                for future in concurrent.futures.as_completed(futures):
                    value=future.result()
                    if value is None:continue
                    done+=1;failed+=value['status']=='failed'
                    status('reviewing',last_model=value['model'],last_group=value['group_id'])
            summary=summarize(config)
            status('complete' if summary['complete'] else 'partial')
            return summary
        finally:
            for client in clients.values():client.backend.client.close()


def summarize(config):
    out=Path(config['runtime']['output_dir']);meta=read_json(out/'preparation.json')
    sampled=read_jsonl(out/'sampled_edits.private.jsonl');baseline=config['review']['baseline_model']
    models=config['review']['models'];decisions={baseline:{n['negative_id']:'accept' for n in sampled}};errors={};group_status={}
    allowed={r['group_id']:{n['negative_id'] for n in r['candidates']} for r in read_jsonl(out/'review_groups.private.jsonl')}
    details=[]
    for model in models:
        decisions[model]={};errors[model]=[];group_status[model]=Counter()
        marker=out/'models'/model/'identity.json';identity=read_json(marker) if marker.exists() else None
        for p in (out/'models'/model/'groups').glob('*.json'):
            result=read_json(p);gid=result['group_id']
            if identity is None or result['signature']!=digest(identity) or result['simulated']:
                raise ValueError('Invalid, simulated, or mismatched result')
            if gid not in allowed:raise ValueError('Unexpected review group')
            group_status[model][result['status']]+=1
            if result['status']!='complete':errors[model].append({'group_id':gid,'error':result['error']});continue
            parsed=RetainedReasonReview.model_validate({'verdicts':result['verdicts']})
            if len(parsed.verdicts)!=len(allowed[gid]) or {v.negative_id for v in parsed.verdicts}!=allowed[gid]:
                raise ValueError('Result edit IDs do not match frozen sample')
            for v in parsed.verdicts:
                decisions[model][v.negative_id]=v.decision
                details.append({'model':model,'group_id':gid,**v.model_dump(mode='json')})
    table=[]
    for model in [baseline,*models]:
        for operation,target in CELLS:
            subset=[n for n in sampled if n['edit_operation']==operation and n['semantic_target']==target]
            values=[decisions[model][n['negative_id']] for n in subset if n['negative_id'] in decisions[model]]
            complete=len(values)==len(subset)
            table.append({'model':model,'operation':operation,'semantic_target':target,'sampled':len(subset),
                'reviewed':len(values),'accepted':values.count('accept'),'rejected':values.count('reject'),
                'acceptance_percent':100*values.count('accept')/len(values) if complete and values else None,
                'agreement_with_original_acceptance_percent':100*values.count('accept')/len(values) if complete and values else None,
                'complete':complete,'baseline_is_historical':model==baseline})
    overall=[]
    for model in [baseline,*models]:
        values=list(decisions[model].values());complete=len(values)==len(sampled)
        overall.append({'model':model,'sampled':len(sampled),'reviewed':len(values),
            'accepted':values.count('accept'),'rejected':values.count('reject'),
            'acceptance_percent':100*values.count('accept')/len(values) if complete and values else None,
            'complete':complete,'baseline_is_historical':model==baseline})
    pairs=[]
    for a,b in itertools.combinations([baseline,*models],2):
        shared=set(decisions[a])&set(decisions[b]);complete=len(shared)==len(sampled)
        matches=sum(decisions[a][k]==decisions[b][k] for k in shared)
        pairs.append({'model_a':a,'model_b':b,'shared_reviewed':len(shared),'sampled':len(sampled),
            'agreement_percent':100*matches/len(shared) if complete and shared else None,'complete':complete})
    all_done=all(len(decisions[m])==len(sampled) for m in models)
    joint=None
    if all_done:
        unanimous_accept=sum(all(decisions[m][n['negative_id']]=='accept' for m in models) for n in sampled)
        unanimous_reject=sum(all(decisions[m][n['negative_id']]=='reject' for m in models) for n in sampled)
        joint={'all_new_models_accept_percent':100*unanimous_accept/len(sampled),
            'all_new_models_reject_percent':100*unanimous_reject/len(sampled),
            'all_new_models_agreement_percent':100*(unanimous_accept+unanimous_reject)/len(sampled)}
    value={'complete':all_done,'sampled_edits':len(sampled),'groups_per_model':meta['sampled_groups'],
        'interpretation':meta['interpretation'],'context_limit':meta['context_limit'],'per_cell':table,'overall_by_model':overall,
        'pairwise_agreement':pairs,'joint_agreement':joint,'group_status':group_status,'errors':errors,
        'format_policy':FORMAT_POLICY,'reason_length_warnings':reason_warnings(details),
        'kappa_not_reported':'The original baseline is constant accept by sampling design; report conditional acceptance and pairwise agreement, not baseline kappa or accuracy.'}
    write_json(out/'summary.json',value);write_csv(out/'summary_by_cell.csv',table)
    write_csv(out/'overall_by_model.csv',overall)
    write_csv(out/'pairwise_agreement.csv',pairs);write_csv(out/'review_details.csv',details)
    write_csv(out/'rejected_by_new_models.csv',[d for d in details if d['decision']=='reject'])
    matrix=[{'group_id':n['group_id'],'negative_id':n['negative_id'],'operation':n['edit_operation'],
        'semantic_target':n['semantic_target'],'source_text':n['source_text'],'edited_text':n['text'],
        **{m:decisions[m].get(n['negative_id'],'pending') for m in [baseline,*models]}} for n in sampled]
    write_csv(out/'decision_matrix.csv',matrix)
    return value


def probe_one(config,model):
    out=Path(config['runtime']['output_dir']);folder=out/'probes'/model
    client=new_client(config,model,folder/'cache',retries=0);backend=client.backend
    result={'model':model,'text_ok':False,'visual_review_ok':False,'errors':{}}
    try:
        try:
            response=backend.client.post(backend.chat_completions_url,json={
                'model':model,'messages':[{'role':'user','content':'hello'}],'max_tokens':1024})
            raw=_decode_http_response(response)
            write_json(folder/'hello_response.json',{'http_status':response.status_code,
                'content_type':response.headers.get('content-type'),'body':raw})
            response.raise_for_status();parsed=_decode_response_container(raw)
            reply=parsed['choices'][0]['message']['content']
            if not isinstance(reply,str) or not reply.strip():raise ValueError('No nonempty text reply')
            result.update(text_ok=True,hello_reply=reply,reported_model=parsed.get('model'),usage=parsed.get('usage'))
        except Exception as exc:result['errors']['hello']=redact(exc)
        try:
            records=[PinPointGroupRecord.model_validate(r) for r in read_jsonl(out/'review_groups.private.jsonl')]
            # Exercise the maximum actual image count; a single sampled candidate limits test output.
            record=max(records,key=lambda r:(len(r.references)+len(positives_for(r)),r.group_id))
            record=record.model_copy(update={'candidates':record.candidates[:1]})
            reviewed,traces=review_group_candidates(record,client,out/'prompts','v7',
                config['review']['max_output_tokens'],positives_for(record))
            if len(reviewed)!=1:raise ValueError('Wrong probe review count')
            result.update(visual_review_ok=True,probe_group_id=record.group_id,
                probe_negative_id=reviewed[0].negative_id,image_count=len(record.references)+len(positives_for(record)),
                review_decision=reviewed[0].status,review_reason=reviewed[0].selection_reason,
                trace=traces[-1].model_dump(mode='json'))
            result['format_warnings']=reason_warnings([{'negative_id':n.negative_id,'reason':n.selection_reason} for n in reviewed])
        except Exception as exc:result['errors']['visual_review']=redact(exc)
        result['ready_for_audit']=result['visual_review_ok']
        result['readiness_basis']='Successful actual multimodal review; hello is an auxiliary diagnostic'
        result['limit']='A valid multimodal response verifies request/response compatibility, not that the provider/model reliably used the images.'
        write_json(folder/'probe_result.json',result)
        return result
    finally:backend.client.close()


def probe(config,models=None):
    prepare(config);models=models or config['review']['models']
    if set(models)-set([*config['review']['models'],config['review']['baseline_model']]):raise ValueError('Unknown probe model')
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4,len(models))) as pool:
        results=list(pool.map(lambda m:probe_one(config,m),models))
    write_json(Path(config['runtime']['output_dir'])/'probe_summary.json',results)
    for r in results:
        print(json.dumps({k:r[k] for k in ['model','text_ok','visual_review_ok','ready_for_audit','errors']},ensure_ascii=False),flush=True)
    return results


def probe_from_saved(config,models=None):
    """Revalidate the user's saved real API responses without credentials/network."""
    prepare(config);out=Path(config['runtime']['output_dir'])
    models=models or config['review']['models']
    if set(models)-set([*config['review']['models'],config['review']['baseline_model']]):raise ValueError('Unknown probe model')
    records=[PinPointGroupRecord.model_validate(r) for r in read_jsonl(out/'review_groups.private.jsonl')]
    record=max(records,key=lambda r:(len(r.references)+len(positives_for(r)),r.group_id))
    record=record.model_copy(update={'candidates':record.candidates[:1]})
    prompt,_=load_prompt(out/'prompts','validate_group','v7')
    expected_hash=hashlib.sha256((prompt+'\n---\n'+dump_prompt_json(payload_for(record))).encode()).hexdigest()
    images=[im.image_path for im in [*record.references,*positives_for(record)]]
    results=[]
    for model in models:
        folder=out/'probes'/model;result_path=folder/'probe_result.json'
        prior=read_json(result_path) if result_path.exists() else {'model':model,'text_ok':False,'errors':{'hello':'No saved hello probe'}}
        archive=folder/'probe_result.before_reason_compatibility.json'
        if result_path.exists() and not archive.exists():write_json(archive,prior)
        result={**prior,'errors':dict(prior['errors']),'visual_review_ok':False,'cache_only_validation':True}
        try:
            recover_reason_only_errors(folder/'cache',write_json)
            matches=[]
            for path in (folder/'cache/validate_group').glob('*.json'):
                if path.name.endswith('.error.json') or '.rejected.' in path.name:continue
                cached=read_json(path);request=cached.get('request',{})
                if (request.get('prompt_hash')==expected_hash and cached.get('images')==images
                    and request.get('stage')=='validate_group'
                    and request.get('schema')==GroupReviewResponse.model_json_schema()
                    and request.get('max_output_tokens')==config['review']['max_output_tokens']
                    and request.get('backend_identity',{}).get('selection_model')==model):
                    matches.append(cached)
            if len(matches)!=1:raise ValueError(f'Expected one matching saved visual response, found {len(matches)}')
            cached=matches[0];answer=RetainedReasonReview.model_validate(cached['parsed'])
            if len(answer.verdicts)!=1 or answer.verdicts[0].negative_id!=record.candidates[0].negative_id:
                raise ValueError('Saved response has the wrong candidate ID')
            v=answer.verdicts[0]
            result.update(visual_review_ok=True,probe_group_id=record.group_id,probe_negative_id=v.negative_id,
                image_count=len(images),review_decision='accepted' if v.decision=='accept' else 'rejected',
                review_reason=v.reason,trace={**cached['trace'],'cached':True},
                format_warnings=reason_warnings(cached['parsed']['verdicts']))
            result['errors'].pop('visual_review',None)
        except Exception as exc:result['errors']['visual_review']=redact(exc)
        result['ready_for_audit']=result['visual_review_ok']
        result['readiness_basis']='Successful actual multimodal review; hello is an auxiliary diagnostic'
        result['limit']='Saved real responses validate compatibility at probe time, not reliable image grounding or current service availability.'
        write_json(result_path,result);results.append(result)
    summary_path=out/'probe_summary.json';archive=out/'probe_summary.before_reason_compatibility.json'
    if summary_path.exists() and not archive.exists():write_json(archive,read_json(summary_path))
    write_json(summary_path,results)
    for result in results:
        print(json.dumps({k:result[k] for k in ['model','text_ok','visual_review_ok','ready_for_audit','errors','cache_only_validation']},ensure_ascii=False),flush=True)
    return results
