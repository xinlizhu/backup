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
from rank_preference.common import ROOT,digest,read_json,read_jsonl,redact,sha,write_json
from rank_preference.prepare import prepare
from rank_preference.schema import PairJudgment,RelativePairJudgment
from rank_preference.report import summarize


class JudgeBackend(OpenAIChatBackend):
    def _route(self,stage):return super()._route('validate_visual')


def judgment_spec(config):
    mode=config['judging'].get('mode','absolute_compliance')
    if mode=='relative_preference':
        return RelativePairJudgment,ROOT/'prompts/judge_pair_relative.txt','rank_pair_relative_preference','rank_preference_relative_v2'
    if mode=='absolute_compliance':
        return PairJudgment,ROOT/'prompts/judge_pair.txt','rank_pair_judgment','rank_preference_v1'
    raise ValueError('Unknown judging mode: '+mode)


class MockBackend:
    model_name='MOCK-NOT-REAL';model_revision=None
    def cache_identity(self):return {'mock':True}
    def generate_json(self,images,system_prompt,user_prompt,response_model,*,stage,max_output_tokens):
        payload=json.loads(user_prompt)
        data={'case_id':payload['case_id'],'constraints':[{'constraint':'Synthetic test only',
            'A_status':'uncertain','B_status':'uncertain','A_evidence':'No image inference was performed.',
            'B_evidence':'No image inference was performed.'}],'A_status':'uncertain','B_status':'uncertain',
            'decision':'uncertain','reason':'Mock pipeline test, not an actual assessment.'}
        if response_model is RelativePairJudgment:
            data={k:v for k,v in data.items() if k not in ['A_status','B_status']}
            data.update(decision='A',confidence='low')
        return BackendResult(data=data,raw_response={'mock':True,'data':data},model_name=self.model_name,provider='mock')


def judge_one(case,client,config,signature,mock,stop):
    if stop.is_set():return None
    out=Path(config['runtime']['output_dir'])
    base={'case_id':case['case_id'],'signature':signature,'simulated':mock}
    try:
        response_model,prompt_path,stage,_=judgment_spec(config)
        payload={'case_id':case['case_id'],'source_instruction':case['source_instruction'],
                 'image_layout':{'reference':1,'candidate_A':2,'candidate_B':3}}
        for _ in range(client.max_retries+1):
            answer,trace=client.call(response_model,images=case['images'],
                system_prompt=prompt_path.read_text(),
                user_prompt=json.dumps(payload,ensure_ascii=False),stage=stage,
                max_output_tokens=config['judging']['max_output_tokens'])
            if answer.case_id==case['case_id']:break
            client.reject_cached_response(trace,'Wrong case_id')
        else:raise ValueError('Wrong case_id after structural retries')
        value={**base,'status':'complete','judgment':answer.model_dump(mode='json'),'trace':trace.model_dump(mode='json')}
        write_json(out/'judgments'/f"{case['case_id']}.json",value);return value
    except Exception as exc:
        value={**base,'status':'failed','error':redact(exc)}
        write_json(out/'judgments'/f"{case['case_id']}.json",value)
        if isinstance(exc,FatalProviderError):stop.set();raise
        return value


def run(config,prepare_only=False,summarize_only=False,max_cases=None,mock=False):
    out=Path(config['runtime']['output_dir']);out.mkdir(parents=True,exist_ok=True)
    with (out/'.pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        meta=prepare(config)
        if prepare_only or summarize_only:
            summary=summarize(config)
            if prepare_only:write_json(out/'status.json',{'phase':'prepared','gpt_case_count':meta['gpt_case_count'],
                'same_image_skipped':meta['same_image_skipped'],'api_calls_this_invocation':0})
            return summary
        cases=read_jsonl(out/'judge_cases.private.jsonl')
        if not cases:return summarize(config)
        m=config['model']
        if not mock and m['api_key']=='${OPENAI_API_KEY}':
            key=os.environ.get('OPENAI_API_KEY','').strip()
            if not key or '。。' in key or '...' in key or key.startswith('YOUR_'):
                write_json(out/'status.json',{'phase':'awaiting_credentials','gpt_case_count':len(cases),
                    'same_image_skipped':meta['same_image_skipped'],'api_calls_this_invocation':0})
                raise ValueError('Rankings prepared. Export a real OPENAI_API_KEY to start GPT judgments.')
        backend=MockBackend() if mock else JudgeBackend(**{k:v for k,v in m.items() if k!='max_retries'})
        _,prompt_path,_,prompt_version=judgment_spec(config)
        source=[prompt_path,*list((ROOT/'src/rank_preference').glob('*.py'))]
        identity={'prepared_signature':meta['signature'],'backend':backend.cache_identity(),'mock':mock,
            'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in source},'judging':config['judging']}
        identity_path=out/'judgment_identity.json'
        if identity_path.exists() and read_json(identity_path)!=identity:raise ValueError('Judging code/model/prompt changed; use a new output directory')
        write_json(identity_path,identity);signature=digest(identity)
        for p in source:
            dest=out/'source_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():dest.write_bytes(p.read_bytes())
        write_json(out/'effective_config.json',{**config,'model':{**m,'api_key':'${OPENAI_API_KEY}'}})
        pending=[]
        for case in cases:
            p=out/'judgments'/f"{case['case_id']}.json"
            if p.exists():
                r=read_json(p)
                if r['signature']!=signature:raise ValueError('Case signature mismatch')
                if r['status']=='complete':continue
            pending.append(case)
        previous=len(cases)-len(pending)
        if max_cases is not None:pending=pending[:max_cases]
        client=StructuredLLMClient(backend,out/'api_cache',prompt_version=prompt_version,max_retries=m['max_retries'])
        stop=threading.Event();started=time.time();done=0;failed=0
        def status(phase,**kw):
            value={'phase':phase,'pid':os.getpid(),'simulated':mock,'planned_comparisons':meta['planned_comparisons'],
                'same_image_skipped':meta['same_image_skipped'],'total_gpt_cases':len(cases),
                'already_complete':previous,'processed_this_invocation':done,'failed_this_invocation':failed,
                'elapsed_seconds':time.time()-started,**kw}
            write_json(out/'status.json',value);print(json.dumps(value,ensure_ascii=False),flush=True)
        status('judging')
        pool=concurrent.futures.ThreadPoolExecutor(max_workers=config['runtime']['max_workers'])
        futures=[pool.submit(judge_one,c,client,config,signature,mock,stop) for c in pending]
        try:
            for future in concurrent.futures.as_completed(futures):
                value=future.result()
                if value is None:continue
                done+=1;failed+=value['status']=='failed';status('judging',last_case=value['case_id'])
            pool.shutdown(wait=True)
        except BaseException as exc:
            stop.set();pool.shutdown(wait=False,cancel_futures=True)
            status('stopped',error=redact(exc));summarize(config);raise
        finally:
            if hasattr(backend,'client') and not stop.is_set():backend.client.close()
        summary=summarize(config);status('complete' if summary['complete'] else 'partial')
        return summary
