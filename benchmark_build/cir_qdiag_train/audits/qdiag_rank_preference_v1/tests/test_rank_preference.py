import json
from pathlib import Path
import sys
import threading
import httpx
from PIL import Image
import pytest
from pydantic import ValidationError

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import _bootstrap
from rank_preference.prepare import build_comparisons,score_order
from rank_preference.report import resolve_outcome
from rank_preference.schema import PairJudgment
from rank_preference.workflow import JudgeBackend,judge_one
from cir_qdiag_openai_cirr.models.client import StructuredLLMClient


def ranking(tmp_path):
    paths=[]
    for i,color in enumerate(['green','red','blue','red']):
        p=tmp_path/f'{i}.png';Image.new('RGB',(5,5),color).save(p);paths.append(str(p))
    return {'audit_query_id':'q1','epoch':1,'source_instruction':'Find a red object.',
        'reference_path':paths[0],'candidates':[{'id':str(i),'path':paths[i+1]} for i in range(3)],
        'teacher_order':[0,1,2],'student_order':[1,0,2],'teacher_scores':[.5,.4,.3],
        'student_scores':[.4,.5,.3],'student_checkpoint':'initial.pt','source_global_step':0}


def judgment(case_id):
    return {'case_id':case_id,'constraints':[{'constraint':'red object','A_status':'supported',
        'B_status':'contradicted','A_evidence':'Synthetic fixture: red.','B_evidence':'Synthetic fixture: blue.'}],
        'A_status':'supported','B_status':'contradicted','decision':'A_only','reason':'Synthetic test only.'}


def test_same_image_skip_is_not_a_correctness_label(tmp_path):
    row=ranking(tmp_path);row['student_order']=[0,1,2]
    comparisons,cases,_=build_comparisons([row],[1,2,3],42)
    assert not cases and all(c['same_image_skipped'] for c in comparisons)
    for c in comparisons:
        outcome=resolve_outcome(c,None)
        assert outcome['outcome']=='same_image_skipped' and outcome['teacher_status']=='not_assessed'


def test_distinct_ids_with_identical_rgb_are_also_skipped(tmp_path):
    row=ranking(tmp_path);row['student_order']=[2,1,0]
    comparisons,cases,_=build_comparisons([row],[1],42)
    assert comparisons[0]['teacher_image_id']!=comparisons[0]['student_image_id']
    assert comparisons[0]['same_image_skipped'] and not cases


def test_repeated_unordered_pair_uses_one_order_and_maps_roles(tmp_path):
    row=ranking(tmp_path)
    comparisons,cases,_=build_comparisons([row],[1,2],42)
    assert len(cases)==1 and comparisons[0]['case_id']==comparisons[1]['case_id']
    assert comparisons[0]['teacher_side']!=comparisons[1]['teacher_side']
    result={'status':'complete','judgment':judgment(cases[0]['case_id'])}
    assert {resolve_outcome(c,result)['outcome'] for c in comparisons}=={'teacher_only','student_only'}
    assert build_comparisons([row],[1,2],42)[1]==cases


def test_score_ties_do_not_use_forced_target_position():
    a=['target','other1','other2'];b=['other2','target','other1']
    first=[a[i] for i in score_order([.5,.5,.5],a,'q',42)]
    second=[b[i] for i in score_order([.5,.5,.5],b,'q',42)]
    assert first==second
    assert score_order([.1,.8,.7],a,'q',42)[0]==1


def test_judge_status_and_decision_must_match_constraints():
    x=judgment('test');PairJudgment.model_validate(x)
    x['decision']='both'
    with pytest.raises(ValidationError):PairJudgment.model_validate(x)
    x=judgment('test');x['constraints'][0]['A_status']='uncertain'
    with pytest.raises(ValidationError):PairJudgment.model_validate(x)


def test_one_http_judgment_no_swap_no_scores_and_resume(tmp_path):
    row=ranking(tmp_path);_,cases,_=build_comparisons([row],[1],42);case=cases[0]
    requests=[]
    def handler(request):
        payload=json.loads(request.content);requests.append(payload)
        assert payload['model']=='test-judge'
        content=payload['messages'][1]['content']
        assert sum(x['type']=='image_url' for x in content)==3
        data=json.loads(content[0]['text'])
        assert set(data)=={'case_id','source_instruction','image_layout'}
        answer=judgment(data['case_id'])
        return httpx.Response(200,json={'id':'mock-http','choices':[{'message':{'content':json.dumps(answer)}}]})
    backend=JudgeBackend(api_base='https://mock.invalid/v1',api_key='dummy-test-key',
        generation_model='unused-generator',validation_model='test-judge',generation_reasoning_effort=None,validation_reasoning_effort=None)
    backend.client.close();backend.client=httpx.Client(transport=httpx.MockTransport(handler))
    client=StructuredLLMClient(backend,tmp_path/'api_cache',prompt_version='test',max_retries=0)
    config={'runtime':{'output_dir':str(tmp_path)},'judging':{'max_output_tokens':1000}}
    a=judge_one(case,client,config,'sig',False,threading.Event())
    assert a['status']=='complete' and len(requests)==1
    b=judge_one(case,client,config,'sig',False,threading.Event())
    assert b['trace']['cached'] and len(requests)==1
