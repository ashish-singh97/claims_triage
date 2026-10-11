import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src.data import demo_claims, simulate, load_cms
from src.features import FeatureStore, FEATURES
from src.routing import route_batch, route_individual
from src.api import app, get_bundle
ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT/'config.json').read_text())

@pytest.fixture(scope='module')
def generated():
    return simulate(demo_claims(1500,42),CFG)

def test_future_and_same_day_history_excluded(generated):
    claims,_ = generated
    c = next(c for c in claims if c['submitted_at'][:10] >= '2009-01-01')
    future = copy.deepcopy(c); future['submitted_at']='2012-01-01T00:00:00'
    same_day = copy.deepcopy(c); same_day['claim_id']='another'
    assert FeatureStore(claims).row(c)==FeatureStore(claims+[future,same_day]).row(c)

def test_previous_history_used(generated):
    claims,_ = generated
    c = next(c for c in claims if c['submitted_at'][:10] >= '2009-01-01')
    earlier = copy.deepcopy(c)
    earlier['submitted_at']=str(pd.Timestamp(c['submitted_at'])-pd.Timedelta(days=1))
    a,b = FeatureStore(claims).row(c),FeatureStore(claims+[earlier]).row(c)
    assert b['member_claims_30d']==a['member_claims_30d']+1
    assert b['duplicate_matches_30d']==a['duplicate_matches_30d']+1

def test_feature_allowlist():
    assert not {'target','scenario','reference_amount','LINE_NCH_PMT_AMT','member_id','provider_id'}.intersection(FEATURES)

def test_capacity_and_cost_boundaries():
    cfg = dict(CFG,review_capacity_per_day=2,senior_capacity_per_day=1)
    p = np.tile([.01,.19,.8],(30,1)); dates=['2010-01-01']*15+['2010-01-02']*15
    a = route_batch(p,dates,cfg)
    for day in set(dates):
        subset=a[np.array(dates)==day]
        assert (subset==1).sum()<=2 and (subset==2).sum()<=1
    assert route_individual([1,0,0],CFG,0)==0
    assert route_individual([0,1,0],CFG,0)==1
    assert route_individual([0,0,1],CFG,0)==2

def test_cms_zip_ingestion(tmp_path):
    import zipfile
    common={'DESYNPUF_ID':'000M','CLM_FROM_DT':'20080101','CLM_THRU_DT':'20080102',
            'HCPCS_CD_1':'00123','PRF_PHYSN_NPI_1':'000N','LINE_ALOWD_CHRG_AMT_1':'50',
            'ICD9_DGNS_CD_1':'0010','LINE_ICD9_DGNS_CD_1':'0010'}
    for letter in ['A','B']:
        row=dict(common,CLM_ID=f'C{letter}')
        with zipfile.ZipFile(tmp_path/f'DE1_0_Carrier_Claims_Sample_1{letter}.zip','w') as z:
            z.writestr(f'Carrier_Sample_1{letter}.csv',pd.DataFrame([row]).to_csv(index=False))
    result,audit=load_cms(tmp_path,CFG)
    assert len(result)==2 and result[0]['lines'][0]['hcpcs_code']=='00123'
    assert result[0]['diagnosis_codes']==['0010'] and audit['rejected_count']==0

@pytest.fixture(scope='module')
def bundle():
    if not (ROOT/'artifacts/bundle.joblib').exists():
        pytest.fail('Run python -m src.pipeline --demo before integration tests')
    import os
    os.environ['MODEL_BUNDLE']=str(ROOT/'artifacts/bundle.joblib')
    get_bundle.cache_clear()
    return get_bundle()

def test_api_offline_parity_unknown_codes(bundle):
    c=json.loads((ROOT/'artifacts/example_claim.json').read_text()); client=TestClient(app)
    result=client.post('/score',json=c)
    assert result.status_code==200,result.text
    p=bundle['model'].predict_proba(bundle['store'].transform([c]))[0]
    assert np.allclose(list(result.json()['probabilities'].values()),p) and np.isclose(p.sum(),1)
    c['lines'][0].update(provider_id='NEW',hcpcs_code='Z9999')
    assert client.post('/score',json=c).status_code==200

def test_api_validation(bundle):
    client=TestClient(app); c=json.loads((ROOT/'artifacts/example_claim.json').read_text())
    c['lines'][0]['submitted_amount']=-1
    assert client.post('/score',json=c).status_code==422
    c['lines'][0]['submitted_amount']=10; c['service_to']='1900-01-01'
    assert client.post('/score',json=c).status_code==422

def test_api_idempotence_batch(bundle):
    client=TestClient(app); c=json.loads((ROOT/'artifacts/example_claim.json').read_text())
    assert client.post('/score',json=c).json()==client.post('/score',json=c).json()
    result=client.post('/score-batch',json={'claims':[dict(c,claim_id=f'new-{i}') for i in range(30)]})
    assert result.status_code==200,result.text
    queues=[r['queue'] for r in result.json()['results']]
    assert queues.count('high_risk')<=CFG['senior_capacity_per_day']
    assert queues.count('review')<=CFG['review_capacity_per_day']
