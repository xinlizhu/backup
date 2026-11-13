import copy
import json
from pathlib import Path
import sys
import threading
import httpx
from PIL import Image
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import _bootstrap
from mining_audit.responses import accepted_witnesses,valid_span_edit,ProposalResponse,ReviewResponse
from mining_audit.report import CONFLICT_TYPES
from mining_audit.workflow import AuditBackend,process_case
from cir_qdiag_openai_cirr.models.client import StructuredLLMClient
from cir_qdiag_openai_cirr.models.base import FatalProviderError


def grounding(status):
    return {'status':status,'evidence':'Synthetic test fixture, not a real image judgment.'}


def witness(operation='substitute'):
    return {'witness_id':'w1','operation':operation,'semantic_target':'attribute',
        'before_span':'red','after_span':'blue','edited_instruction':'Find a blue dog.',
        'witness_type':'substitution_contrast','changed_constraint':'color',
        'edited_on_positive':grounding('contradicted'),'edited_on_candidate':grounding('supported')}


def review_for(w,original_candidate='contradicted',edited_positive='contradicted'):
    return {'case_id':'case_test','source_on_positive':grounding('supported'),
        'source_on_candidate':grounding(original_candidate),
        'verdicts':[{'witness_id':w['witness_id'],'decision':'accept','failure_code':None,
            'witness_type':w['witness_type'],'reason':'Synthetic test only.',
            'edited_on_positive':grounding(edited_positive),'edited_on_candidate':grounding('supported')}]}


def test_single_span_is_verified_not_trusted():
    w=witness()
    assert valid_span_edit('Find a red dog.',w)
    w['edited_instruction']='Find a blue cat.'
    assert not valid_span_edit('Find a red dog.',w)
    proposal={'witnesses':[w]}
    accepted,checks=accepted_witnesses('Find a red dog.',proposal,review_for(w))
    assert not accepted and 'span_reconstruction_failed' in checks[0]['local_failures']


def test_delete_keeps_positive_supported():
    w={**witness('delete'),'before_span':'red ','after_span':None,'edited_instruction':'Find a dog.',
       'witness_type':'deletion_relaxation'}
    proposal={'witnesses':[w]}
    assert accepted_witnesses('Find a red dog.',proposal,review_for(w,edited_positive='supported'))[0]
    assert not accepted_witnesses('Find a red dog.',proposal,review_for(w,edited_positive='contradicted'))[0]


def test_add_is_not_original_instruction_conflict():
    w={**witness('add'),'before_span':None,'after_span':'blue ','edited_instruction':'Find a blue dog.',
       'witness_type':'addition_disambiguation'}
    proposal={'witnesses':[w]}
    accepted,_=accepted_witnesses('Find a dog.',proposal,review_for(w,original_candidate='supported'))
    assert accepted and not ({v['witness_type'] for v in accepted}&CONFLICT_TYPES)
    assert not accepted_witnesses('Find a dog.',proposal,review_for(w,original_candidate='contradicted'))[0]


def test_bundled_semantics_rejected_even_if_span_reconstructs():
    w={**witness(),'before_span':'red dog','after_span':'blue cat','edited_instruction':'Find a blue cat.'}
    assert valid_span_edit('Find a red dog.',w)
    rev=review_for(w);rev['verdicts'][0].update(decision='reject',failure_code='not_single_edit')
    assert not accepted_witnesses('Find a red dog.',{'witnesses':[w]},rev)[0]


def setup_http(tmp_path,quota=False):
    images=[]
    for i,color in enumerate(['red','green','blue']):
        p=tmp_path/f'{i}.png';Image.new('RGB',(4,4),color).save(p);images.append(str(p))
    requests=[]
    def handler(request):
        payload=json.loads(request.content)
        requests.append(payload)
        if quota:return httpx.Response(401,json={'error':{'code':'invalid_api_key'}})
        content=payload['messages'][1]['content']
        assert sum(x['type']=='image_url' for x in content)==3
        user=json.loads(content[0]['text'])
        assert not ({'teacher_score','epoch','origin','candidate_position','model_name'}&set(user))
        w=witness()
        if payload['model']=='test-generator':
            data={'case_id':user['case_id'],'case_status':'source_valid',
                'source_on_positive':grounding('supported'),'source_on_candidate':grounding('contradicted'),
                'candidate_relation':'single_constraint_conflict','violated_source_constraints':['red'],'witnesses':[w]}
            ProposalResponse.model_validate(data)
        else:
            assert payload['model']=='test-reviewer'
            assert 'evidence' not in json.dumps(user)
            assert 'changed_constraint' not in user['witnesses'][0]
            data=review_for(w);ReviewResponse.model_validate(data)
        return httpx.Response(200,json={'id':'local-mock','choices':[{'message':{'content':json.dumps(data)}}],
            'usage':{'prompt_tokens':1,'completion_tokens':1}})
    backend=AuditBackend(api_base='https://mock.invalid/v1',api_key='dummy-local-test-key',
        generation_model='test-generator',validation_model='test-reviewer',generation_reasoning_effort=None,
        validation_reasoning_effort=None)
    backend.client.close();backend.client=httpx.Client(transport=httpx.MockTransport(handler))
    client=StructuredLLMClient(backend,tmp_path/'cache',prompt_version='test',max_retries=0)
    case={'case_id':'case_test','audit_query_id':'audit_query_001','source_instruction':'Find a red dog.',
          'images':images,'teacher_score':999,'epoch':4,'origin':'mined'}
    config={'runtime':{'output_dir':str(tmp_path)},'generation':{'max_output_tokens':1000},'validation':{'max_output_tokens':1000}}
    return client,case,config,requests


def test_real_http_serialization_blinding_review_and_cache_resume(tmp_path):
    client,case,config,requests=setup_http(tmp_path)
    result=process_case(case,config,client,'test-signature',False,threading.Event())
    assert result['status']=='complete' and len(result['accepted_witnesses'])==1
    assert len(requests)==2
    result2=process_case(case,config,client,'test-signature',False,threading.Event())
    assert len(requests)==2 and all(t['cached'] for t in result2['traces'])


def test_invalid_credentials_stop_further_cases(tmp_path):
    client,case,config,requests=setup_http(tmp_path,quota=True)
    stop=threading.Event()
    with pytest.raises(FatalProviderError):process_case(case,config,client,'test',False,stop)
    assert stop.is_set() and len(requests)==1
    assert process_case(case,config,client,'test',False,stop) is None


@pytest.mark.parametrize('n_queries,single_epoch',[(100,None),(25,4)])
def test_partial_reporting_keeps_denominators_and_deduplicates_operations(tmp_path,monkeypatch,n_queries,single_epoch):
    from mining_audit import report
    queries=[{'audit_query_id':f'q{i}','epoch':single_epoch or i//25+1} for i in range(n_queries)]
    cases=[{'case_id':f'c{i}_{j}','audit_query_id':f'q{i}','source_instruction':'Find a red dog.'}
           for i in range(n_queries) for j in range(31)]
    members=[{'case_id':c['case_id'],'audit_query_id':c['audit_query_id'],'epoch':single_epoch or int(c['audit_query_id'][1:])//25+1,
              'origin':'mined','teacher_margin_saved':.1,'teacher_margin_recomputed_fp32':.1} for c in cases]
    files={'queries.private.jsonl':queries,'cases.private.jsonl':cases,'memberships.private.jsonl':members}
    monkeypatch.setattr(report,'read_jsonl',lambda p:files[p.name])
    monkeypatch.setattr(report,'export_html',lambda *args:None)
    (tmp_path/'results').mkdir()
    w={**witness(),'independent_review':{'reason':'Synthetic fixture'}}
    d={**w,'operation':'delete','witness_type':'deletion_relaxation'}
    result={'case_id':'c0_0','status':'complete','simulated':False,
        'accepted_witnesses':[w,d],'review':review_for(w)}
    (tmp_path/'results/c0_0.json').write_text(json.dumps(result))
    config={'runtime':{'output_dir':str(tmp_path),'bootstrap_seed':42,'bootstrap_replicates':20}}
    summary=report.summarize(config)
    arm=summary['arms']['mined']
    assert not summary['complete'] and summary['pending_unique_pairs']==n_queries*31-1
    assert arm['confirmed_candidate_pairs']==1 and arm['confirmed_queries']==1
    assert arm['queries']==n_queries
    assert arm['confirmed_query_coverage_percent']==100/n_queries
    assert arm['confirmed_candidate_frequency_percent']==pytest.approx(100/(n_queries*31))
    assert arm['teacher_margin_saved']['n']==1
    assert 'query_coverage_95CI_percent' not in arm


def test_epoch4_complete_report_bootstraps_only_selected_queries(tmp_path,monkeypatch):
    from mining_audit import report
    queries=[{'audit_query_id':f'q{i}','epoch':4} for i in range(25)]
    cases=[{'case_id':f'c{i}_{j}','audit_query_id':f'q{i}','source_instruction':'Find a red dog.'}
           for i in range(25) for j in range(31)]
    members=[{'case_id':c['case_id'],'audit_query_id':c['audit_query_id'],'epoch':4,
              'origin':'mined','teacher_margin_saved':.1,'teacher_margin_recomputed_fp32':.1} for c in cases]
    files={'queries.private.jsonl':queries,'cases.private.jsonl':cases,'memberships.private.jsonl':members}
    monkeypatch.setattr(report,'read_jsonl',lambda p:files[p.name])
    monkeypatch.setattr(report,'export_html',lambda *args:None)
    (tmp_path/'results').mkdir()
    w={**witness(),'independent_review':{'reason':'Synthetic test fixture'}}
    for i,c in enumerate(cases):
        result={'case_id':c['case_id'],'status':'complete','simulated':False,
            'accepted_witnesses':[w] if i==0 else [],'review':review_for(w)}
        (tmp_path/'results'/f"{c['case_id']}.json").write_text(json.dumps(result))
    config={'runtime':{'output_dir':str(tmp_path),'bootstrap_seed':42,'bootstrap_replicates':200}}
    result=report.summarize(config);arm=result['arms']['mined']
    assert result['complete'] and result['selected_epochs']==[4]
    assert arm['queries']==25 and arm['candidate_pairs']==775
    assert arm['confirmed_query_coverage_percent']==4
    assert len(arm['query_coverage_95CI_percent'])==2


def test_subset_reuse_preserves_original_and_checks_identical_case(tmp_path):
    from mining_audit.common import write_json,write_jsonl,sha
    from mining_audit.reuse import initialize_annotations
    old=tmp_path/'previous';new=tmp_path/'subset'
    old.mkdir();new.mkdir()
    config={'runtime':{'output_dir':str(old)},'model':{},'generation':{'max_output_tokens':6000},'validation':{'max_output_tokens':6000}}
    backend={'test':'backend'}
    old_id,old_signature=initialize_annotations(config,{'signature':'old-prepared'},backend,False)
    case={'case_id':'case_test','audit_query_id':'q0','source_instruction':'Find a red dog.','images':[]}
    for folder in [old,new]:
        write_jsonl(folder/'cases.private.jsonl',[case]);write_json(folder/'image_inventory.json',{})
    write_json(old/'prepared_metadata.json',{'manifest_sha256':{n:sha(old/n) for n in ['cases.private.jsonl','image_inventory.json']}})
    w=witness()
    proposal={'case_id':'case_test','case_status':'source_valid','source_on_positive':grounding('supported'),
        'source_on_candidate':grounding('contradicted'),'candidate_relation':'single_constraint_conflict',
        'violated_source_constraints':['red'],'witnesses':[w]}
    record={'case_id':'case_test','audit_query_id':'q0','signature':old_signature,'simulated':False,
        'status':'complete','proposal':proposal,'review':review_for(w),'traces':[]}
    write_json(old/'results/case_test.json',record)
    original_hash=sha(old/'results/case_test.json')
    config['runtime']={'output_dir':str(new),'reuse_from':str(old)}
    _,new_signature=initialize_annotations(config,{'signature':'new-prepared'},backend,False)
    migrated=json.loads((new/'results/case_test.json').read_text())
    assert migrated['signature']==new_signature and len(migrated['accepted_witnesses'])==1
    assert migrated['imported_from']['original_signature']==old_signature
    assert sha(old/'results/case_test.json')==original_hash
    altered={**case,'source_instruction':'Find a yellow dog.'}
    write_jsonl(new/'cases.private.jsonl',[altered])
    with pytest.raises(ValueError,match='identical subset'):
        initialize_annotations(config,{'signature':'new-prepared'},backend,False)
