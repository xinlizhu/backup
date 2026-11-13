"""Keep the original wire schema, retain verbose rationales without changing verdicts."""
import json
from pathlib import Path

from pydantic import Field,ValidationError
from cir_qdiag.models.client import StructuredVLMClient
from cir_qdiag.models.openai_chat import _extract_json
from cir_qdiag.schemas import CandidateVerdict,GroupReviewResponse,CallTrace

FORMAT_POLICY='retain_full_reason_original_wire_schema_v1'


class RetainedReasonVerdict(CandidateVerdict):
    # Preserve accept/reject/failure-code consistency and extra-field validation.
    reason: str=Field(min_length=1)


class RetainedReasonReview(GroupReviewResponse):
    verdicts: list[RetainedReasonVerdict]

    @classmethod
    def model_json_schema(cls,*args,**kwargs):
        # Send the EXACT original schema, including maxLength=300.
        return GroupReviewResponse.model_json_schema(*args,**kwargs)


def reason_warnings(verdicts):
    return [{'negative_id':v['negative_id'],'kind':'reason_exceeds_requested_length',
             'requested_max_characters':300,'actual_characters':len(v['reason']),
             'action':'full reason retained; decision and failure_code unchanged'}
            for v in verdicts if len(v['reason'])>300]


class ReviewClient(StructuredVLMClient):
    def call(self,response_model,**kwargs):
        if response_model is GroupReviewResponse:response_model=RetainedReasonReview
        return super().call(response_model,**kwargs)


def recover_reason_only_errors(cache_dir,write_json):
    """Recover saved, otherwise valid responses locally; never send a new request."""
    recovered=[]
    for error_path in Path(cache_dir).rglob('*.error.json'):
        success_path=error_path.with_name(error_path.name.removesuffix('.error.json')+'.json')
        if success_path.exists():continue
        saved=json.loads(error_path.read_text());request=saved.get('request',{})
        if request.get('stage')!='validate_group' or request.get('schema')!=GroupReviewResponse.model_json_schema():continue
        for detail in reversed(saved.get('error_details',[])):
            try:
                data=_extract_json(detail.get('response_text',''))
                parsed=RetainedReasonReview.model_validate(data)
                try:GroupReviewResponse.model_validate(data)
                except ValidationError as exc:
                    errors=exc.errors()
                    if not errors or any(e['type']!='string_too_long' or e['loc'][-1]!='reason' for e in errors):continue
                else:continue
            except (ValueError,TypeError):continue
            raw=detail.get('raw_response')
            if not isinstance(raw,dict):continue
            trace=CallTrace(stage='validate_group',model_name=detail['model_name'],
                model_revision=detail.get('model_revision'),provider='openai-compatible-chat',
                reasoning_effort=request['backend_identity'].get('selection_reasoning_effort'),
                prompt_version=request['prompt_version'],prompt_hash=request['prompt_hash'],
                input_hash=success_path.stem,raw_response_path=str(success_path.resolve()),
                attempts=detail['attempt'],request_id=detail.get('request_id'),
                usage=raw.get('usage') if isinstance(raw.get('usage'),dict) else None)
            value={'request':request,'images':saved['images'],'parsed':parsed.model_dump(mode='json'),
                'raw_response':raw,'request_id':detail.get('request_id'),'trace':trace.model_dump(mode='json'),
                'attempt':detail['attempt'],'previous_errors':saved.get('errors',[]),
                'previous_error_details':saved.get('error_details',[]),
                'local_recovery':{'policy':FORMAT_POLICY,'source_error_path':str(error_path.resolve()),
                    'new_api_calls':0,'format_warnings':reason_warnings(data['verdicts'])}}
            write_json(success_path,value);recovered.append(str(success_path));break
    return recovered
