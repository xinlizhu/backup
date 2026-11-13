import json
from pathlib import Path
import sys

import httpx
from PIL import Image
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import _bootstrap
import single_target_audit as audit
from cir_qdiag.models.openai_chat import OpenAIChatBackend


def group_fixture(tmp_path):
    paths=[]
    for i,color in enumerate(['red','blue','green','yellow']):
        p=tmp_path/f'{i}.png';Image.new('RGB',(5,5),color).save(p);paths.append(str(p))
    return {'group_id':'test-group','references':[{'image_id':'r1','image_path':paths[0]},
        {'image_id':'r2','image_path':paths[1]}],
        'positive_images':[{'image_id':'p1','image_path':paths[2]},{'image_id':'p2','image_path':paths[3]}],
        'metadata':{'validated_positive_image_ids':['p1','p2']}},paths


def edits():
    return [{'negative_id':f'n{i}','group_id':'test-group','instruction_id':'q',
        'edit_operation':op,'semantic_target':axis,'source_text':'Use <img1> style and <img2> color.',
        'text':f'Synthetic edited instruction {i}'} for i,(op,axis) in enumerate(audit.CELLS)]


def test_one_target_keeps_both_references_and_is_order_independent(tmp_path):
    group,_=group_fixture(tmp_path)
    cases=audit.build_cases(edits(),{group['group_id']:group},42)
    assert len({c['target_image_id'] for c in cases})==1
    assert all(len(c['images'])==3 and c['image_layout']=={'reference_images':[1,2],'target_image':3} for c in cases)
    chosen=audit.choose_target(group,42)
    group['positive_images'].reverse();group['metadata']['validated_positive_image_ids'].reverse()
    assert audit.choose_target(group,42)==chosen
    for c in cases:
        assert set(audit.payload_for(c))=={'case_id','modification_text','edited_text','image_layout'}
        assert 'object' not in c['case_id'] and 'substitute' not in c['case_id']
    group['metadata']['validated_positive_image_ids']=[chosen['image_id']]
    assert audit.choose_target(group,42)==chosen


def test_uncertainty_and_incomplete_results_are_not_rejections():
    row=audit.rates(['accept','reject','uncertain'],3)
    assert row['accepted']==row['rejected']==row['uncertain']==1
    assert row['accept_percent']==pytest.approx(100/3)
    partial=audit.rates(['accept','uncertain'],3)
    assert partial['rejected']==0 and partial['accept_percent'] is None


def test_prompt_preserves_delete_semantics():
    text=(ROOT/'prompts/review.txt').read_text()
    assert 'strictly weaker' in text and 'that alone is not a reason to reject a deletion' in text
    assert 'what could be added' in text and 'uncertain rather than inventing' in text


def test_http_payload_resume_wrong_ids_and_pairwise_summary(tmp_path,monkeypatch):
    config=audit.load_config(ROOT/'configs/accepted_450.yaml',tmp_path)
    config['api']['max_retries']=0
    group,_=group_fixture(tmp_path);cases=audit.build_cases(edits(),{group['group_id']:group},42)
    meta={'signature':'synthetic-only','limits':['Synthetic fixture; not actual experiment results.']}
    audit.write_json(tmp_path/'preparation.json',meta)
    audit.write_jsonl(tmp_path/'cases.private.jsonl',cases)
    (tmp_path/'review_prompt.txt').write_bytes((ROOT/'prompts/review.txt').read_bytes())
    monkeypatch.setattr(audit,'prepare',lambda c:meta)
    requests=[];fail=[True]
    def handler(request):
        body=json.loads(request.content);requests.append(body)
        model=body['model'];parts=body['messages'][1]['content']
        assert sum(p['type']=='image_url' for p in parts)==3
        data=json.loads(parts[0]['text']);assert set(data)=={'case_id','modification_text','edited_text','image_layout'}
        assert body['messages'][0]['content'].startswith((ROOT/'prompts/review.txt').read_text().strip())
        case=next(c for c in cases if c['case_id']==data['case_id'])
        decision='accept' if model=='gpt-6.1-terra' else 'uncertain'
        answer={'case_id':data['case_id'],'decision':decision,'reason':'Synthetic fixture only.'}
        if fail[0] and model=='grok-4.6' and case['negative_id']=='n0':answer['case_id']='wrong'
        return httpx.Response(200,json={'id':'synthetic','choices':[{'message':{'content':json.dumps(answer)}}]})
    def factory(c,model):
        b=OpenAIChatBackend(api_base='https://mock.invalid/v1',api_key='[REDACTED]',generation_model=model,
            selection_model=model,generation_reasoning_effort=None,selection_reasoning_effort=None)
        b.client.close();b.client=httpx.Client(transport=httpx.MockTransport(handler));return b
    monkeypatch.setattr(audit,'backend_for',factory)
    first=audit.run(config)
    assert not first['complete'] and len(requests)==18
    assert first['overall'][1]['reviewed']==8 and first['overall'][1]['rejected']==0
    fail[0]=False;second=audit.run(config)
    assert second['complete'] and len(requests)==19
    assert second['overall'][0]['accept_percent']==100
    assert second['overall'][1]['uncertain_percent']==100
    assert second['pairwise_agreement'][0]['agreement_percent_all_decisions']==0
    assert second['pairwise_agreement'][0]['agreement_percent_both_decisive'] is None
    assert audit.run(config)['complete'] and len(requests)==19
