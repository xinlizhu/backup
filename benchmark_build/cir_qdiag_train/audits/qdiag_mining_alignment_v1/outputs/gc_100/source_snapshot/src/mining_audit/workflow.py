import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import threading
import time

from cir_qdiag_openai_cirr.models.base import FatalProviderError
from cir_qdiag_openai_cirr.models.client import StructuredLLMClient
from cir_qdiag_openai_cirr.models.openai_chat import OpenAIChatBackend
from cir_qdiag_openai_cirr.schemas import BackendResult
from mining_audit.common import ROOT, digest, read_json, read_jsonl, redact, sha, write_json
from mining_audit.prepare import prepare
from mining_audit.responses import (ProposalResponse, ReviewResponse, accepted_witnesses,
    proposal_partition_error, review_partition_error)


class AuditBackend(OpenAIChatBackend):
    def _route(self, stage):
        return super()._route('validate_visual' if stage == 'review_witnesses' else stage)


class MockBackend:
    model_name = 'MOCK-NOT-SCIENTIFIC-DATA'
    model_revision = None

    def cache_identity(self):
        return {'backend':self.model_name}

    def generate_json(self, images, system_prompt, user_prompt, response_model, *, stage, max_output_tokens):
        payload = json.loads(user_prompt)
        assert len(images) == 3
        ground = {'status':'uncertain','evidence':'Mock response for pipeline testing only; no image judgment.'}
        base = {'case_id':payload['case_id'],'source_on_positive':ground,'source_on_candidate':ground}
        if stage == 'propose_witnesses':
            data = {**base,'case_status':'source_invalid_or_uncertain','candidate_relation':'uncertain',
                    'violated_source_constraints':[], 'witnesses':[]}
        else:
            data = {**base, 'verdicts':[]}
        return BackendResult(data=data, raw_response={'mock':True,'data':data},
                             model_name=self.model_name, provider='mock')


def call_checked(client, response_model, *, images, system, payload, stage, tokens, check):
    traces = []
    for _ in range(client.max_retries + 1):
        answer, trace = client.call(response_model, images=images, system_prompt=system,
            user_prompt=json.dumps(payload,ensure_ascii=False), stage=stage, max_output_tokens=tokens)
        data = answer.model_dump(mode='json')
        traces.append(trace.model_dump(mode='json'))
        error = check(data)
        if error is None:
            return data, traces
        client.reject_cached_response(trace, error)
    raise RuntimeError('Invalid case/witness ID partition after structural retries')


def process_case(case, config, client, signature, mock, stop):
    if stop.is_set():
        return None
    out = Path(config['runtime']['output_dir'])
    path = out/'results'/f"{case['case_id']}.json"
    base = {'case_id':case['case_id'],'audit_query_id':case['audit_query_id'],
            'signature':signature,'simulated':mock}
    traces, proposal = [], None
    try:
        # No provenance, origin, candidate ranking, or score is sent to the annotator.
        payload = {'case_id':case['case_id'],'source_instruction':case['source_instruction'],
                   'image_layout':{'reference':1,'dataset_positive':2,'candidate':3}}
        proposal, pt = call_checked(client, ProposalResponse, images=case['images'],
            system=(ROOT/'prompts/propose_witnesses.txt').read_text(), payload=payload,
            stage='propose_witnesses',tokens=config['generation']['max_output_tokens'],
            check=lambda x:proposal_partition_error(x,case['case_id']))
        traces += pt
        if stop.is_set():
            return None
        # Fresh review sees edits/spans, never the generator's evidence or original verdicts.
        keys=['witness_id','operation','semantic_target','before_span','after_span','edited_instruction','witness_type']
        review_payload={**payload,'witnesses':[{k:w[k] for k in keys} for w in proposal['witnesses']]}
        review, rt=call_checked(client, ReviewResponse,images=case['images'],
            system=(ROOT/'prompts/review_witnesses.txt').read_text(),payload=review_payload,
            stage='review_witnesses',tokens=config['validation']['max_output_tokens'],
            check=lambda x:review_partition_error(x,case['case_id'],proposal['witnesses']))
        traces += rt
        accepted,checks=accepted_witnesses(case['source_instruction'],proposal,review)
        value={**base,'status':'complete','proposal':proposal,'review':review,
               'accepted_witnesses':accepted,'local_checks':checks,'traces':traces}
        write_json(path,value)
        return value
    except Exception as exc:
        value={**base,'status':'failed','error':redact(exc),'proposal':proposal,'traces':traces}
        write_json(path,value)
        markers=('model_not_found','invalid_api_key','permission_denied','unsupported_model','model does not exist')
        if isinstance(exc,FatalProviderError) or any(x in str(exc).casefold() for x in markers):
            stop.set()
            raise
        return value


def run(config, *, prepare_only=False, summarize_only=False, max_pairs=None, mock=False):
    from mining_audit.report import summarize
    out=Path(config['runtime']['output_dir']);out.mkdir(parents=True,exist_ok=True)
    with (out/'.pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        meta=prepare(config)
        if prepare_only or summarize_only:
            summarize(config)
            return
        m=config['model']
        if not mock:
            # Never store resolved keys in configs, cache identities or status reports.
            key=os.environ.get('OPENAI_API_KEY','').strip()
            if m['api_key']=='${OPENAI_API_KEY}' and (not key or '。。' in key or '...' in key or key.startswith('YOUR_')):
                write_json(out/'status.json',{'phase':'awaiting_credentials','data_prepared':True,
                    'total_pairs':meta['unique_case_count'],'api_calls_this_invocation':0})
                raise ValueError('Data prepared. OPENAI_API_KEY is missing or a placeholder; export a real key before annotation.')
        backend=MockBackend() if mock else AuditBackend(**{k:v for k,v in m.items() if k!='max_retries'})
        prompts=[ROOT/'prompts/propose_witnesses.txt',ROOT/'prompts/review_witnesses.txt',
                 ROOT/'proposal.schema.json',ROOT/'review.schema.json']
        code=list((ROOT/'src/mining_audit').glob('*.py'))
        identity={'prepared_signature':meta['signature'],'backend':backend.cache_identity(),'mock':mock,
            'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in prompts+code},
            'generation':config['generation'],'validation':config['validation']}
        signature=digest(identity)
        identity_path=out/'annotation_identity.json'
        if identity_path.exists() and read_json(identity_path)!=identity:
            raise RuntimeError('Annotation prompts/model/code changed; use another output directory instead of mixing results')
        write_json(identity_path,identity)
        snapshot=out/'source_snapshot'
        for p in prompts+code:
            dest=snapshot/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():dest.write_bytes(p.read_bytes())
        safe_config={**config,'model':{**config['model'],'api_key':'${OPENAI_API_KEY}'}}
        write_json(out/'effective_config.json',safe_config)
        cases=read_jsonl(out/'cases.private.jsonl')
        pending=[]
        for case in cases:
            f=out/'results'/f"{case['case_id']}.json"
            if f.exists():
                old=read_json(f)
                if old.get('signature')!=signature:raise RuntimeError('Per-case signature mismatch')
                if old.get('status')=='complete':continue
            pending.append(case)
        remaining=len(pending)
        if max_pairs is not None:pending=pending[:max_pairs]
        client=StructuredLLMClient(backend,out/'api_cache',prompt_version='qdiag_mining_alignment_v1',max_retries=m['max_retries'])
        stop=threading.Event();started=time.time();failures=0;processed=0
        def status(phase,**kw):
            value={'phase':phase,'simulated':mock,'pid':os.getpid(),'total_pairs':len(cases),
                'already_complete':len(cases)-remaining,'processed_this_invocation':processed,
                'failed_this_invocation':failures,'elapsed_seconds':time.time()-started,**kw}
            write_json(out/'status.json',value)
            print(json.dumps(value,ensure_ascii=False),flush=True)
        status('annotating')
        pool=concurrent.futures.ThreadPoolExecutor(max_workers=config['runtime']['max_workers'])
        futures={pool.submit(process_case,c,config,client,signature,mock,stop):c['case_id'] for c in pending}
        try:
            for future in concurrent.futures.as_completed(futures):
                value=future.result()
                if value is None:continue
                processed+=1;failures+=value['status']=='failed'
                status('annotating',last_case=value['case_id'],last_status=value['status'])
            pool.shutdown(wait=True)
        except BaseException as exc:
            stop.set()
            pool.shutdown(wait=False,cancel_futures=True)
            status('stopped',error=redact(exc))
            summarize(config)
            raise
        finally:
            if hasattr(backend,'client') and not stop.is_set():backend.client.close()
        summary=summarize(config)
        status('complete' if summary['complete'] else 'partial',summary=str(out/'summary.json'))
