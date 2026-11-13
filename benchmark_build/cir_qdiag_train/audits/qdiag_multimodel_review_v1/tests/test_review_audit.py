import json
from pathlib import Path
import sys

import httpx
from PIL import Image
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import _bootstrap
import review_audit as audit
from cir_qdiag.models.openai_chat import OpenAIChatBackend
from cir_qdiag.schemas import PinPointGroupRecord


def test_stratified_sample_is_frozen_unique_and_order_independent():
    rows=[{'group_id':'group','negatives':[{'negative_id':f'{o}_{a}_{j}',
        'edit_operation':o,'semantic_target':a} for o,a in audit.CELLS for j in range(4)]}]
    selected,_=audit.sample_edits(rows,2,42)
    assert len(selected)==18 and len({n['negative_id'] for n in selected})==18
    rows[0]['negatives'].reverse()
    assert audit.sample_edits(rows,2,42)[0]==selected
    with pytest.raises(ValueError):audit.sample_edits(rows,5,42)


def test_reduced_sample_is_exact_prefix_of_each_frozen_parent_cell():
    rows=[{'group_id':'group','negatives':[{'negative_id':f'{o}_{a}_{j}',
        'edit_operation':o,'semantic_target':a} for o,a in audit.CELLS for j in range(6)]}]
    parent,_=audit.sample_edits(rows,4,42)
    reduced,_=audit.sample_edits(rows,2,42,parent)
    assert len(reduced)==18
    for op,axis in audit.CELLS:
        old=[r for r in parent if r['edit_operation']==op and r['semantic_target']==axis]
        new=[r for r in reduced if r['edit_operation']==op and r['semantic_target']==axis]
        assert new==old[:2]
    assert {n['negative_id'] for n in reduced}<={n['negative_id'] for n in parent}
    with pytest.raises(ValueError,match='Insufficient parent'):audit.sample_edits(rows,5,42,parent)
    changed=[dict(n) for n in parent];changed[0]['text']='changed'
    with pytest.raises(ValueError,match='differs'):audit.sample_edits(rows,2,42,changed)


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    config=audit.load_config(ROOT/'configs/accepted_900_seed42.yaml',tmp_path)
    config['sampling']['per_cell']=1;config['api']['max_retries']=0
    paths=[]
    for name,color in [('ref','red'),('positive','blue')]:
        p=tmp_path/f'{name}.png';Image.new('RGB',(5,5),color).save(p);paths.append(str(p))
    source='Find a red chair.'
    instructions=[{'instruction_id':f'q{i}','text':source,'query_category':'test','token_count':4,
        'length_category':'short','is_question':False,'is_imperative':True,'l1_interest':'test'} for i in range(2)]
    candidates=[{'negative_id':f'n{i}','instruction_id':'q0','source_text':source,'text':f'Synthetic edit {i}',
        'semantic_target':axis,'edit_operation':op,'before_span':None,'after_span':None,'status':'generated'}
        for i,(op,axis) in enumerate(audit.CELLS)]
    group={'group_id':'synthetic_group','references':[{'image_id':'ref','image_path':paths[0]}],
        'positive_images':[{'image_id':'positive','image_path':paths[1]}],'official_negative_image_ids':['unused'],
        'instructions':instructions,'candidates':candidates,'negatives':[],
        'metadata':{'stage':'candidates','num_generated_candidates':9,'validated_positive_image_ids':['positive']}}
    group=PinPointGroupRecord.model_validate(group).model_dump(mode='json')
    sample=[{'group_id':group['group_id'],**n} for n in group['candidates']]
    meta={'signature':'synthetic-only','sampled_edits':9,'sampled_groups':1,
          'interpretation':'Synthetic test; not experiment data.','context_limit':'Synthetic fixture.'}
    audit.write_json(tmp_path/'preparation.json',meta)
    audit.write_jsonl(tmp_path/'sampled_edits.private.jsonl',sample)
    audit.write_jsonl(tmp_path/'review_groups.private.jsonl',[group])
    (tmp_path/'prompts').mkdir()
    (tmp_path/'prompts/validate_group_v7.txt').write_bytes((audit.SOURCE/'prompts/validate_group_v7.txt').read_bytes())
    monkeypatch.setattr(audit,'prepare',lambda c:meta)
    return config,tmp_path


def install_mock(monkeypatch,config,requests,fail_kimi,long_reasons=False,hello_524=False):
    def handler(request):
        body=json.loads(request.content);requests.append(body)
        model=body['model']
        assert request.headers['user-agent']=='curl/8.0'
        if fail_kimi[0] and model=='kimi-k3':return httpx.Response(503,json={'error':{'message':'synthetic outage'}})
        if len(body['messages'])==1:
            if hello_524 and model=='kimi-k3':return httpx.Response(524,json={'error':'synthetic hello timeout'})
            text='Hello from a synthetic HTTP test.'
        else:
            assert body['messages'][0]['role']=='system'
            assert body['messages'][0]['content'].startswith((audit.SOURCE/'prompts/validate_group_v7.txt').read_text().strip())
            content=body['messages'][1]['content']
            assert sum(x['type']=='image_url' for x in content)==2
            payload=json.loads(content[0]['text'])
            assert set(payload)=={'group_id','image_layout','instructions','candidates'}
            verdicts=[]
            for i,c in enumerate(payload['candidates']):
                assert set(c)=={'negative_id','instruction_id','source_text','negative_text','semantic_target',
                    'edit_operation','before_span','after_span'}
                accept=model in ['grok-4.7','kimi-k3'] or (model=='gpt-6.1-sol' and i<3)
                verdicts.append({'negative_id':c['negative_id'],'decision':'accept' if accept else 'reject',
                    'failure_code':None if accept else 'not_visually_verifiable',
                    'reason':'Synthetic fixture only.'*(25 if long_reasons else 1)})
            text=json.dumps({'verdicts':verdicts})
        return httpx.Response(200,json={'id':'synthetic-http','model':model,
            'choices':[{'message':{'role':'assistant','content':text},'finish_reason':'stop'}]})
    def factory(c,model):
        backend=OpenAIChatBackend(api_base='https://mock.invalid/v1',api_key='synthetic-test-key',
            generation_model=model,selection_model=model,generation_reasoning_effort=None,selection_reasoning_effort=None)
        backend.client.close();backend.client=httpx.Client(transport=httpx.MockTransport(handler),
            headers={'User-Agent':'curl/8.0'})
        return backend
    monkeypatch.setattr(audit,'backend_for',factory)


def test_multimodel_run_failure_is_not_rejection_and_resume(prepared,monkeypatch):
    config,out=prepared;requests=[];fail_kimi=[True]
    install_mock(monkeypatch,config,requests,fail_kimi)
    first=audit.run(config)
    assert not first['complete'] and len(requests)==4
    kimi=next(r for r in first['overall_by_model'] if r['model']=='kimi-k3')
    assert kimi['reviewed']==0 and kimi['rejected']==0 and kimi['acceptance_percent'] is None
    assert all(r['agreement_percent'] is None for r in first['pairwise_agreement'] if 'kimi-k3' in [r['model_a'],r['model_b']])
    fail_kimi[0]=False
    second=audit.run(config)
    assert second['complete'] and len(requests)==5
    by_model={r['model']:r['acceptance_percent'] for r in second['overall_by_model']}
    assert by_model['grok-4.7']==100 and by_model['claude-sonnet-5-5']==0
    assert by_model['gpt-6.1-sol']==pytest.approx(100/3)
    assert second['joint_agreement']['all_new_models_agreement_percent']==0
    third=audit.run(config)
    assert third['complete'] and len(requests)==5
    assert {r['model'] for r in requests}==set(config['review']['models'])


def test_probe_uses_text_and_real_review_schema_without_creating_audit_results(prepared,monkeypatch):
    config,out=prepared;requests=[]
    install_mock(monkeypatch,config,requests,[False])
    result=audit.probe(config)
    assert len(requests)==8 and all(r['ready_for_audit'] for r in result)
    assert not (out/'models').exists()
    assert all(r['image_count']==2 for r in result)


def test_hello_timeout_does_not_block_successful_actual_review_and_cache_recheck(prepared,monkeypatch):
    config,out=prepared;requests=[]
    install_mock(monkeypatch,config,requests,[False],hello_524=True)
    results=audit.probe(config)
    kimi=next(r for r in results if r['model']=='kimi-k3')
    assert not kimi['text_ok'] and kimi['visual_review_ok'] and kimi['ready_for_audit']
    assert '524' in kimi['errors']['hello']
    rechecked=audit.probe_from_saved(config)
    assert all(r['ready_for_audit'] and r['cache_only_validation'] for r in rechecked)
    assert len(requests)==8 and not (out/'models').exists()


def test_missing_verdict_cannot_be_counted_as_a_completed_review(prepared,monkeypatch):
    config,out=prepared;requests=[]
    install_mock(monkeypatch,config,requests,[False]);audit.run(config)
    path=out/'models/grok-4.7/groups/synthetic_group.json'
    result=audit.read_json(path);result['verdicts'].pop();audit.write_json(path,result)
    with pytest.raises(ValueError,match='edit IDs'):audit.summarize(config)


def test_requested_schema_unchanged_and_only_reason_length_is_relaxed():
    from pydantic import ValidationError
    from cir_qdiag.schemas import GroupReviewResponse
    from review_compat import RetainedReasonReview
    assert RetainedReasonReview.model_json_schema()==GroupReviewResponse.model_json_schema()
    assert RetainedReasonReview.model_json_schema()['$defs']['CandidateVerdict']['properties']['reason']['maxLength']==300
    data={'verdicts':[{'negative_id':'n','decision':'reject','failure_code':'positive_compatible','reason':'x'*501}]}
    with pytest.raises(ValidationError):GroupReviewResponse.model_validate(data)
    assert RetainedReasonReview.model_validate(data).model_dump(mode='json')==data
    data['verdicts'][0]['decision']='accept'
    with pytest.raises(ValidationError):RetainedReasonReview.model_validate(data)


def test_long_reasons_survive_http_cache_and_summary(prepared,monkeypatch):
    config,out=prepared;requests=[]
    install_mock(monkeypatch,config,requests,[False],long_reasons=True)
    result=audit.run(config)
    assert result['complete'] and len(requests)==4
    assert len(result['reason_length_warnings'])==36
    group=audit.read_json(out/'models/claude-sonnet-5-5/groups/synthetic_group.json')
    assert all(v['decision']=='reject' and v['reason']=='Synthetic fixture only.'*25 for v in group['verdicts'])
    raw=audit.read_json(group['traces'][0]['raw_response_path'])
    assert raw['parsed']['verdicts']==group['verdicts']
    assert audit.run(config)['complete'] and len(requests)==4


def test_recover_only_length_failure_without_reissuing_request(tmp_path):
    from cir_qdiag.schemas import GroupReviewResponse
    from review_compat import recover_reason_only_errors
    data={'verdicts':[{'negative_id':'n','decision':'reject','failure_code':'positive_compatible','reason':'x'*501}]}
    body={'choices':[{'message':{'content':json.dumps(data)},'finish_reason':'stop'}]}
    error={'request':{'stage':'validate_group','schema':GroupReviewResponse.model_json_schema(),
        'backend_identity':{'selection_reasoning_effort':None},'prompt_version':'v7','prompt_hash':'test'},
        'images':[],'errors':['old length error'],'error_details':[{'response_text':json.dumps(data),
        'raw_response':body,'model_name':'claude-sonnet-5-5','attempt':1,'request_id':'synthetic'}]}
    path=tmp_path/'case.error.json';audit.write_json(path,error)
    assert len(recover_reason_only_errors(tmp_path,audit.write_json))==1
    recovered=audit.read_json(tmp_path/'case.json')
    assert recovered['parsed']==data and recovered['raw_response']==body
    assert recovered['local_recovery']['new_api_calls']==0 and path.exists()
    assert recover_reason_only_errors(tmp_path,audit.write_json)==[]
    error['error_details'][0]['response_text']=json.dumps({'verdicts':[dict(data['verdicts'][0],decision='accept')]})
    audit.write_json(tmp_path/'invalid.error.json',error)
    assert recover_reason_only_errors(tmp_path,audit.write_json)==[]
    assert not (tmp_path/'invalid.json').exists()
