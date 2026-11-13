from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,model_validator

Status=Literal['supported','contradicted','uncertain']


class Strict(BaseModel):model_config=ConfigDict(extra='forbid')


class Constraint(Strict):
    constraint: str=Field(min_length=1)
    A_status: Status
    B_status: Status
    A_evidence: str=Field(min_length=1,max_length=500)
    B_evidence: str=Field(min_length=1,max_length=500)


def aggregate(statuses):
    if 'contradicted' in statuses:return 'contradicted'
    if 'uncertain' in statuses:return 'uncertain'
    return 'supported'


def decision_for(a,b):
    if 'uncertain' in (a,b):return 'uncertain'
    return {('supported','contradicted'):'A_only',('contradicted','supported'):'B_only',
            ('supported','supported'):'both',('contradicted','contradicted'):'neither'}[a,b]


class PairJudgment(Strict):
    case_id: str=Field(min_length=1)
    constraints: list[Constraint]=Field(min_length=1,max_length=32)
    A_status: Status
    B_status: Status
    decision: Literal['A_only','B_only','both','neither','uncertain']
    reason: str=Field(min_length=1,max_length=700)

    @model_validator(mode='after')
    def check_consistency(self):
        if self.A_status!=aggregate([c.A_status for c in self.constraints]):raise ValueError('A status does not match constraints')
        if self.B_status!=aggregate([c.B_status for c in self.constraints]):raise ValueError('B status does not match constraints')
        if self.decision!=decision_for(self.A_status,self.B_status):raise ValueError('Decision does not match statuses')
        return self


class RelativePairJudgment(Strict):
    """A forced relative preference, including when both images pass or fail."""
    case_id: str=Field(min_length=1)
    constraints: list[Constraint]=Field(min_length=1,max_length=32)
    decision: Literal['A','B']
    confidence: Literal['high','medium','low']
    reason: str=Field(min_length=1,max_length=700)
