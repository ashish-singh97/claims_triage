"""Read-only scoring API; batch capacity applies only to this request."""
from datetime import date, datetime
from functools import lru_cache
import os
from typing import Annotated
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict, model_validator
from src.routing import route_individual, route_batch, QUEUES
from src.explain import explain

Code = Annotated[str, Field(min_length=1,max_length=12,pattern=r'^[A-Za-z0-9]+$')]
Identifier = Annotated[str,Field(min_length=1,max_length=100)]

class Line(BaseModel):
    model_config = ConfigDict(extra='forbid',allow_inf_nan=False)
    provider_id: Identifier
    hcpcs_code: Code
    submitted_amount: float = Field(ge=0,le=10000000)
    diagnosis_code: Code | None = None

class Claim(BaseModel):
    model_config = ConfigDict(extra='forbid')
    claim_id: Identifier
    member_id: Identifier
    submitted_at: datetime
    service_from: date
    service_to: date
    birth_date: date | None = None
    diagnosis_codes: list[Code] = Field(default_factory=list,max_length=30)
    lines: list[Line] = Field(min_length=1,max_length=100)

    @model_validator(mode='after')
    def valid_dates(self):
        if self.service_to < self.service_from:
            raise ValueError('service_to must not precede service_from')
        if self.submitted_at.date() < self.service_to:
            raise ValueError('submission must not precede service end')
        if self.submitted_at.tzinfo is not None:
            raise ValueError('Use timezone-naive timestamps for this date-based prototype')
        if self.birth_date and self.birth_date > self.service_from:
            raise ValueError('birth_date must not follow service date')
        return self

class Batch(BaseModel):
    claims: list[Claim] = Field(min_length=1,max_length=1000)
    @model_validator(mode='after')
    def unique_ids(self):
        if len({c.claim_id for c in self.claims}) != len(self.claims):
            raise ValueError('Duplicate claim_id within batch')
        return self

@lru_cache(maxsize=1)
def get_bundle():
    # Never load untrusted joblib/pickle uploads.
    return joblib.load(os.environ.get('MODEL_BUNDLE','artifacts/bundle.joblib'))

app = FastAPI(title='Synthetic Claims Triage',version='1.0')

def prepare(claims,bundle):
    records = [c.model_dump(mode='json') for c in claims]
    if any(c['submitted_at'][:10] < bundle['store'].reference_end for c in records):
        raise HTTPException(status_code=422,detail='Submission precedes reference snapshot')
    X = bundle['store'].transform(records)
    return records,X,bundle['model'].predict_proba(X)

def response(c,row,p,action,bundle,capacity):
    return {'claim_id':c['claim_id'],
            'probabilities':dict(zip(['clean','denial_risk','fwa_risk'],map(float,p))),
            'risk_score':float(p[1]+p[2]),'queue':QUEUES[action],
            'reasons':explain(bundle['model'],row,bundle['reference'],bundle['config']) if action else [],
            'model_version':bundle['model_version'],'policy_version':bundle['policy_version'],
            'data_mode':bundle['data_mode'],'capacity_applied_to_request_batch':capacity,
            'history_snapshot_as_of':bundle['store'].history_as_of,
            'history_includes_submission_day':False}

@app.get('/health')
def health():
    b = get_bundle()
    return {'status':'ok','model':b['selected_model'],'data_mode':b['data_mode']}

@app.post('/score')
def score(claim: Claim):
    b = get_bundle(); claims,X,p = prepare([claim],b)
    a = route_individual(p[0],b['config'],b['penalty'])
    return response(claims[0],X.iloc[0].to_dict(),p[0],a,b,False)

@app.post('/score-batch')
def score_batch(batch: Batch):
    b = get_bundle(); claims,X,p = prepare(batch.claims,b)
    actions = route_batch(p,[c['submitted_at'][:10] for c in claims],b['config'],b['penalty'])
    return {'results':[response(c,X.iloc[i].to_dict(),p[i],int(actions[i]),b,True) for i,c in enumerate(claims)],
            'capacity_scope':'This request only; persistent daily capacity needs a transactional queue manager.'}
