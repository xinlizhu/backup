import json
import threading

import httpx
import pytest
from pydantic import ValidationError

from test_rank_preference import judgment,ranking
from rank_preference.prepare import build_comparisons
from rank_preference.relative_report import preference_tables
from rank_preference.report import resolve_outcome
from rank_preference.schema import RelativePairJudgment
from rank_preference.workflow import JudgeBackend,judge_one
from cir_qdiag_openai_cirr.models.client import StructuredLLMClient


def relative_judgment(case_id,decision='A',status='contradicted'):
    return {'case_id':case_id,'constraints':[{'constraint':'red object',
        'A_status':status,'B_status':status,'A_evidence':'Synthetic A evidence.',
        'B_evidence':'Synthetic B evidence.'}],
        'decision':decision,'confidence':'low','reason':'Synthetic relative preference only.'}


@pytest.mark.parametrize('status',['supported','contradicted','uncertain'])
def test_equal_compliance_status_does_not_prevent_relative_choice(status):
    for side in ['A','B']:
        assert RelativePairJudgment.model_validate(relative_judgment('case',side,status)).decision==side


@pytest.mark.parametrize('decision',['both','neither','uncertain','tie','A_only','B_only'])
def test_relative_schema_rejects_nonbinary_and_old_decisions(decision):
    with pytest.raises(ValidationError):
        RelativePairJudgment.model_validate(relative_judgment('case',decision))


def test_reused_relative_pair_maps_to_both_model_roles(tmp_path):
    comparisons,cases,_=build_comparisons([ranking(tmp_path)],[1,2],42)
    assert len(cases)==1
    result={'status':'complete','judgment':relative_judgment(cases[0]['case_id'])}
    assert {resolve_outcome(c,result)['outcome'] for c in comparisons}=={'teacher_preferred','student_preferred'}


def test_percentage_denominator_includes_low_confidence_excludes_same_image():
    rows=[{'epoch':1,'rank':1,'student_checkpoint':'initial.pt','outcome':outcome,'judge_confidence':'low'}
          for outcome in ['teacher_preferred','student_preferred','student_preferred','same_image_skipped']]
    rows[-1]['judge_confidence']=''
    table,details=preference_tables(rows)
    assert table[0]['teacher_preference_percent']==pytest.approx(100/3)
    assert table[0]['student_preference_percent']==pytest.approx(200/3)
    assert details[0]['judged_comparisons']==3 and details[0]['low_confidence_choices']==3
    assert set(table[0])=={'epoch','student_checkpoint','rank','teacher_preference_percent','student_preference_percent'}
    assert preference_tables(rows,simulated=True)[0][0]['teacher_preference_percent'] is None
    rows[-1]['outcome']='failed'
    assert preference_tables(rows)[0][0]['teacher_preference_percent'] is None


def test_http_relative_prompt_blinding_fresh_cache_and_resume(tmp_path):
    _,cases,_=build_comparisons([ranking(tmp_path)],[1],42);case=cases[0]
    requests=[]
    def handler(request):
        payload=json.loads(request.content);requests.append(payload)
        content=payload['messages'][1]['content']
        assert sum(x['type']=='image_url' for x in content)==3
        data=json.loads(content[0]['text'])
        assert set(data)=={'case_id','source_instruction','image_layout'}
        if len(requests)==1:
            answer=judgment(data['case_id'])
        else:
            system=payload['messages'][0]['content']
            assert 'direct A-versus-B comparison' in system
            assert 'If both fail' in system and 'If both fulfill' in system
            answer=relative_judgment(data['case_id'])
        return httpx.Response(200,json={'id':'mock-http','choices':[{'message':{'content':json.dumps(answer)}}]})
    backend=JudgeBackend(api_base='https://mock.invalid/v1',api_key='dummy-test-key',
        generation_model='unused-generator',validation_model='test-judge',
        generation_reasoning_effort=None,validation_reasoning_effort=None)
    backend.client.close();backend.client=httpx.Client(transport=httpx.MockTransport(handler))
    client=StructuredLLMClient(backend,tmp_path/'api_cache',prompt_version='test',max_retries=0)
    config={'runtime':{'output_dir':str(tmp_path)},'judging':{'max_output_tokens':1000}}
    old=judge_one(case,client,config,'old-sig',False,threading.Event())
    assert old['status']=='complete'
    config['judging']['mode']='relative_preference'
    fresh=judge_one(case,client,config,'relative-sig',False,threading.Event())
    assert fresh['status']=='complete' and len(requests)==2 and not fresh['trace']['cached']
    assert fresh['judgment']['decision']=='A' and fresh['trace']['stage']=='rank_pair_relative_preference'
    resumed=judge_one(case,client,config,'relative-sig',False,threading.Event())
    assert resumed['trace']['cached'] and len(requests)==2
    backend.client.close()
