import json
import re
from typing import ClassVar
from jsonschema import Draft202012Validator
from pydantic import RootModel
from mining_audit.common import ROOT


class SchemaResponse(RootModel[dict]):
    schema_filename: ClassVar[str]

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        return json.loads((ROOT / cls.schema_filename).read_text())

    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        if isinstance(obj, cls):
            obj = obj.root
        errors = list(Draft202012Validator(cls.model_json_schema()).iter_errors(obj))
        if errors:
            raise ValueError('; '.join(e.message for e in errors[:3]))
        return cls.model_construct(root=obj)


class ProposalResponse(SchemaResponse):
    schema_filename = 'proposal.schema.json'


class ReviewResponse(SchemaResponse):
    schema_filename = 'review.schema.json'


TYPES = {'add': 'addition_disambiguation', 'delete': 'deletion_relaxation',
         'substitute': 'substitution_contrast'}
PATTERNS = {'add': ('supported', 'supported', 'contradicted', 'supported'),
            'delete': ('supported', 'contradicted', 'supported', 'supported'),
            'substitute': ('supported', 'contradicted', 'contradicted', 'supported')}


def normalize(text):
    return re.sub(r'\s+', ' ', text).strip().rstrip('.!?').strip()


def locations(text, span):
    if not span:
        return []
    return [i for i in range(len(text)) if text.startswith(span, i)]


def valid_span_edit(source, witness):
    """Syntactic edit reconstruction only; semantic single-constraint review is separate."""
    op, before, after = witness['operation'], witness['before_span'], witness['after_span']
    edited = witness['edited_instruction']
    if normalize(source) == normalize(edited):
        return False
    if op == 'add' and before is None and after:
        return any(normalize(edited[:i] + edited[i+len(after):]) == normalize(source)
                   for i in locations(edited, after))
    if op == 'delete' and before and after is None:
        return any(normalize(source[:i] + source[i+len(before):]) == normalize(edited)
                   for i in locations(source, before))
    if op == 'substitute' and before and after and before != after:
        return any(normalize(source[:i] + after + source[i+len(before):]) == normalize(edited)
                   for i in locations(source, before))
    return False


def proposal_partition_error(data, case_id):
    if data['case_id'] != case_id:
        return 'Wrong case_id'
    ids = [w['witness_id'] for w in data['witnesses']]
    cells = [(w['operation'], w['semantic_target']) for w in data['witnesses']]
    if len(set(ids)) != len(ids) or len(set(cells)) != len(cells):
        return 'Duplicate witness ID or operation/axis cell'
    return None


def review_partition_error(data, case_id, witnesses):
    if data['case_id'] != case_id:
        return 'Wrong case_id'
    ids = [v['witness_id'] for v in data['verdicts']]
    if len(ids) != len(set(ids)) or set(ids) != {w['witness_id'] for w in witnesses}:
        return 'Missing, duplicate or unknown review witness IDs'
    return None


def accepted_witnesses(source, proposal, review):
    """Retain every rejection in the result, including local guard failures."""
    verdicts = {v['witness_id']: v for v in review['verdicts']}
    accepted, checks = [], []
    for witness in proposal['witnesses']:
        v = verdicts[witness['witness_id']]
        pattern = (review['source_on_positive']['status'], review['source_on_candidate']['status'],
                   v['edited_on_positive']['status'], v['edited_on_candidate']['status'])
        failures = []
        if not valid_span_edit(source, witness):
            failures.append('span_reconstruction_failed')
        if witness['witness_type'] != TYPES[witness['operation']] or v['witness_type'] != TYPES[witness['operation']]:
            failures.append('wrong_witness_type')
        if pattern != PATTERNS[witness['operation']]:
            failures.append('wrong_support_pattern')
        if v['decision'] == 'accept' and v['failure_code'] is not None:
            failures.append('accept_with_failure_code')
        ok = v['decision'] == 'accept' and not failures
        checks.append({'witness_id': witness['witness_id'], 'accepted': ok,
                       'local_failures': failures, 'review_decision': v['decision']})
        if ok:
            accepted.append({**witness, 'independent_review': v})
    return accepted, checks
