"""CLI: python -m src.pipeline --demo OR --raw data/raw."""
import argparse
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, log_loss, precision_recall_curve
from src.data import load_cms, demo_claims, simulate
from src.features import FeatureStore, NUMERIC, CATEGORICAL
from src.model import make_model, CalibratedModel
from src.routing import route_batch, costs, select_penalty, realized_expected_cost, QUEUES
from src.explain import explain


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')


def metrics(y, p, actions, cfg):
    bad = np.asarray(y) != 0
    reviewed = actions != 0
    senior = actions == 2
    fwa = np.asarray(y) == 2
    return {'adverse_average_precision': float(average_precision_score(bad,p[:,1]+p[:,2])),
            'fwa_average_precision': float(average_precision_score(fwa,p[:,2])),
            'log_loss': float(log_loss(y,p,labels=[0,1,2])),
            'multiclass_brier': float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1))),
            'review_precision': float(bad[reviewed].mean()) if reviewed.any() else 0,
            'review_recall': float((bad & reviewed).sum()/max(bad.sum(),1)),
            'fwa_senior_recall': float((fwa & senior).sum()/max(fwa.sum(),1)),
            'auto_approved_adverse_rate': float(bad[~reviewed].mean()) if (~reviewed).any() else 0,
            'review_fraction': float(reviewed.mean()), 'senior_fraction': float(senior.mean()),
            'expected_cost_per_1000': float(realized_expected_cost(y,actions,cfg).mean()*1000)}


def eda(claims, audit, out):
    frame = pd.DataFrame({'date': [c['submitted_at'][:10] for c in claims],
                          'amount': [sum(l['submitted_amount'] for l in c['lines']) for c in claims],
                          'lines': [len(c['lines']) for c in claims],
                          'member': [c['member_id'] for c in claims],
                          'missing_diagnosis': [not c['diagnosis_codes'] for c in claims]})
    summary = {'cleaning': audit, 'claims': len(frame), 'members': int(frame.member.nunique()),
               'amount_percentiles': frame.amount.quantile([0,.5,.9,.95,.99,1]).to_dict(),
               'missing_diagnosis_rate': float(frame.missing_diagnosis.mean()),
               'line_count_summary': frame.lines.describe().to_dict(),
               'note': 'Amounts in plots are simulated submitted charges, not original CMS billed amounts.'}
    save_json(out/'eda_summary.json',summary)
    fig, ax = plt.subplots(1,3,figsize=(12,3.5))
    frame.assign(month=pd.to_datetime(frame.date).dt.to_period('M').astype(str)).groupby('month').size().plot(ax=ax[0])
    ax[0].set_title('Claims by submission month'); ax[0].tick_params(axis='x',rotation=70)
    ax[1].hist(np.log1p(frame.amount),bins=45); ax[1].set_title('Log simulated submitted amount')
    ax[2].hist(frame.groupby('member').size(),bins=40); ax[2].set_title('Claims per sampled member')
    fig.tight_layout(); fig.savefig(out/'eda.png',dpi=150); plt.close(fig)


def source_eda(claims, out):
    """Inspect original cleaned inputs BEFORE synthetic augmentation or labels."""
    lines = [line for claim in claims for line in claim['lines']]
    reference = pd.Series([line['reference_amount'] for line in lines],dtype=float)
    summary = {'claim_count':len(claims), 'line_count':len(lines),
        'source_amount_field':'LINE_ALOWD_CHRG_AMT (allowed, not submitted charge)',
        'missing_reference_amount_rate':float(reference.isna().mean()),
        'negative_reference_amount_count':int(reference.lt(0).sum()),
        'zero_reference_amount_count':int(reference.eq(0).sum()),
        'unknown_provider_rate':float(np.mean([l['provider_id']=='UNKNOWN' for l in lines])),
        'missing_claim_diagnosis_rate':float(np.mean([not c['diagnosis_codes'] for c in claims])),
        'missing_birth_date_rate':float(np.mean([not c.get('birth_date') for c in claims])),
        'allowed_amount_quantiles':reference.quantile([0,.5,.9,.95,.99,1]).to_dict()}
    save_json(out/'source_eda.json',summary)
    pd.Series([l['hcpcs_code'] for l in lines]).value_counts().rename_axis('hcpcs').rename('count').to_csv(out/'source_procedure_counts.csv')
    pd.Series([d for c in claims for d in c['diagnosis_codes']],dtype=str).value_counts().rename_axis('icd9').rename('count').to_csv(out/'source_diagnosis_counts.csv')
    pd.Series([c['service_from'][:7] for c in claims]).value_counts().sort_index().rename_axis('month').rename('count').to_csv(out/'source_monthly_counts.csv')


def write_note(out, report, selected, mode):
    m = report[selected]
    content = f'''# Claims triage technical note

## Data, labels and trust
Mode: **{mode}**. Carrier claim is the scoring unit; member histories are retained after fixed-seed member sampling. The CMS path reads both carrier parts in chunks, preserves code strings, removes exact ingestion duplicates, rejects conflicting claim IDs and quarantines invalid dates or empty procedure claims. ICD-9 is retained rather than falsely converted to ICD-10. Submitted amounts are simulated from source allowed-charge scales; actual allowed/payment/processing fields never enter the model. No real patient data is used. Place of service and exact coverage dates are unavailable, so coverage and unbundling rules are not claimed.

Synthetic scenarios are ordinary (86%), inflation (4%), duplicate (4%), unusual pairing (3%) and provider cluster (3%). Outcomes are stochastic conditional on scenario; every scenario allows clean outcomes and ordinary claims can be adverse. Labels 0/1/2 mean clean/denial risk/FWA risk. They are mutually exclusive simulation classes, not verified adjudication or fraud. Actual balance is in class_balance.csv. Simulation metadata is excluded from features. Metrics measure recovery of the simulation, not clinical validity; adding noise does not remove circularity. A rule baseline exposes that limitation.

## Features and leakage
Three initial months form a warm-up reference and are excluded from model fitting. Features include log amount, claim complexity, static birth-date-derived age, ICD-9 categories, broad HCPCS demonstration groups, smoothed diagnosis/procedure co-occurrence, procedure amount ratios and member/provider/duplicate rolling counts. Peer and pairing references freeze after warm-up. These groups are not a validated coverage policy. Timestamp is service end plus one day, an explicit submission-time assumption. History excludes the whole submission day, avoiding arbitrary same-day ordering. Raw IDs are lookup keys, not predictors. Online and offline scoring share FeatureStore. The immutable serving snapshot does not ingest new traffic.

## Models, imbalance and calibration
Baseline: regularized multinomial logistic regression with median imputation, standard scaling of numerics and bounded one-hot categories. Stronger model: histogram gradient boosting with the same encoding, no scaling, modest depth and fixed iterations. Both use moderate class weights (1,2,4); SMOTE is rejected because mixed claim/code interpolation is not meaningful. An unweighted boosting candidate provides an imbalance ablation. Temperature scaling is fitted on a separate 2009 calibration period with natural balance. A scalar temperature cannot fully correct class-specific bias; inspect reliability plots before deploying. The subsequent 2009 policy period selects model and review penalty by realized conditional cost. 2010 remains the untouched test period.

## Evaluation and routing
Selected model: **{selected}**. Test adverse average precision: **{m['adverse_average_precision']:.3f}**; FWA average precision: **{m['fwa_average_precision']:.3f}**; recall among reviewed claims: **{m['review_recall']:.3f}**. Expected cost per 1,000 test claims: **${m['expected_cost_per_1000']:,.0f}**. Full comparison, calibration, queue counts and sensitivity results are generated separately. These numbers are conditional on synthetic outcomes and assumed reviewer performance, not real savings.

Costs: review $10, senior review $30, wrongly paid denial $200 and FWA $2,000. Standard review detects 90% of denial and 50% of FWA; senior detects 95% and 90%. Cost-based action boundaries replace arbitrary probability cutoffs. Daily batch routing greedily allocates positive expected-loss reductions under separate capacities (100 standard, 20 senior). This is an approximation, not a globally optimal allocator. API single-claim recommendations cannot guarantee capacity; the batch endpoint applies capacity for the supplied batch only. Cost comparison includes review nothing and review everything (the latter ignores capacity deliberately as a reference). Sensitivity varies loss and capacity, without retuning on test outcomes.

## Explanations, deployment and next week
Reasons use local one-feature replacement by training medians/modes, ranked by reduction in predicted auto-payment loss. Each reason includes actual claim/history values. This is an approximate model attribution, not SHAP or a causal conclusion; correlated replacements can be unrealistic. Wording avoids calling rarity fraud or medical incompatibility. Persist request hash, timestamp, feature snapshot, model and policy versions, action, reasons and reviewer outcome in a production audit store; the demo returns versions but does not persist sensitive request logs.

Deploy behind authentication with schema validation, an append-only history store, idempotent ingestion and an actual queue manager. Monitor volume, missingness, unknown codes, feature and score drift, daily queue occupancy, calibration and delayed outcome performance. Version annual code sets. Reviewer decisions need adjudication-quality checks; audit a sample of auto-approved claims to address feedback selection bias. Retrain after sufficient new adjudicated labels, with temporal challenger validation and rollback. With another week: validate clinical mappings, add coverage/places of service, test unseen-provider cohorts, replace simulation with genuine outcomes, add provider-cluster bootstrap intervals and transactional capacity allocation.
'''
    (out/'technical_note.md').write_text(content, encoding='utf-8')


def run(args):
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.output); out.mkdir(parents=True,exist_ok=True)
    if args.demo:
        base = demo_claims(args.demo_rows,cfg['seed'])
        audit = {'mode': 'generated Option C DEMO, not CMS', 'raw_claims': len(base)}
    else:
        base, audit = load_cms(args.raw,cfg)
    print(f'Cleaned source claims: {len(base)}',flush=True)
    source_eda(base,out)
    claims, labels = simulate(base,cfg)
    eda(claims,audit,out)
    store = FeatureStore(claims)
    valid = [i for i,c in enumerate(claims) if c['submitted_at'][:10] >= store.reference_end]
    claims = [claims[i] for i in valid]
    labels = labels.iloc[valid].reset_index(drop=True)
    X = store.transform(claims)
    y = labels.target.to_numpy()
    dates = np.array([c['submitted_at'][:10] for c in claims])
    balance = pd.crosstab(labels.partition,labels.target).reindex(columns=[0,1,2],fill_value=0)
    balance.columns = ['clean','denial_risk','fwa_risk']
    balance.to_csv(out/'class_balance.csv')
    print(balance,flush=True)
    indexes = {name: np.flatnonzero(labels.partition.eq(name)) for name in ['train','calibration','policy','test']}
    for name, idx in indexes.items():
        if len(idx) < 50 or set(y[idx]) != {0,1,2}:
            raise ValueError(f'{name}: requires at least 50 rows and all 3 classes; increase sample_members/demo_rows.')
    save_json(out/'simulation_audit.json',labels.to_dict('records'))
    X.to_csv(out/'features.csv',index=False)
    reference = {col: float(X.iloc[indexes['train']][col].median()) for col in NUMERIC}
    reference.update({col: X.iloc[indexes['train']][col].mode()[0] for col in CATEGORICAL})
    models, policy_results = {}, {}
    tr, cal, pol, test = [indexes[k] for k in ['train','calibration','policy','test']]
    for name in ['logistic','gradient_boosting','gradient_boosting_unweighted']:
        print(f'Training {name}',flush=True)
        estimator = make_model('logistic' if name == 'logistic' else 'gradient_boosting',cfg['seed'])
        if name.endswith('unweighted'):
            estimator.set_params(classifier__class_weight=None)
        estimator.fit(X.iloc[tr],y[tr])
        calibrated = CalibratedModel(estimator).fit_calibration(X.iloc[cal],y[cal])
        p = calibrated.predict_proba(X.iloc[pol])
        penalty = select_penalty(p,y[pol],dates[pol],cfg)
        actions = route_batch(p,dates[pol],cfg,penalty)
        models[name] = (calibrated,penalty)
        policy_results[name] = metrics(y[pol],p,actions,cfg)
    selected = min(policy_results,key=lambda k: policy_results[k]['expected_cost_per_1000'])
    print(f'Selected on policy validation: {selected}',flush=True)
    report, predictions = {}, {}
    for name,(model,penalty) in models.items():
        p = model.predict_proba(X.iloc[test])
        a = route_batch(p,dates[test],cfg,penalty)
        report[name] = metrics(y[test],p,a,cfg)
        predictions[name] = p
    # Transparent rule reference, not a claimed fraud detector.
    x = X.iloc[test]
    rule_score = (x.duplicate_matches_30d.gt(0).astype(float)*.55 +
                  x.amount_peer_ratio_max.gt(3).astype(float)*.35 +
                  x.provider_recent_ratio.gt(3).astype(float)*.15).clip(.01,.9).to_numpy()
    rp = np.column_stack([1-rule_score,rule_score*.7,rule_score*.3])
    ra = route_batch(rp,dates[test],cfg)
    report['rules'] = metrics(y[test],rp,ra,cfg)
    for name,action in [('review_nothing',0),('review_everything',1)]:
        report[name] = {'expected_cost_per_1000': float(realized_expected_cost(y[test],np.full(len(test),action),cfg).mean()*1000)}
    pd.DataFrame(report).T.to_csv(out/'evaluation.csv')
    save_json(out/'policy_validation.json',policy_results)
    p = predictions[selected]; model, penalty = models[selected]
    actions = route_batch(p,dates[test],cfg,penalty)
    scored = pd.DataFrame({'claim_id': labels.iloc[test].claim_id.to_numpy(), 'submitted_date':dates[test],
                          'target':y[test], 'p_clean':p[:,0], 'p_denial':p[:,1], 'p_fwa':p[:,2],
                          'queue':[QUEUES[a] for a in actions]})
    scored.to_csv(out/'test_predictions.csv',index=False)
    scored.groupby(['submitted_date','queue']).size().unstack(fill_value=0).to_csv(out/'daily_queue_counts.csv')
    sensitivity = []
    for multiplier in [.5,1,2]:
        for capacity in [.5,1,2]:
            changed = dict(cfg)
            changed['denial_loss'] *= multiplier; changed['fwa_loss'] *= multiplier
            changed['review_capacity_per_day'] = int(cfg['review_capacity_per_day']*capacity)
            changed['senior_capacity_per_day'] = int(cfg['senior_capacity_per_day']*capacity)
            a = route_batch(p,dates[test],changed,penalty)
            sensitivity.append({'loss_multiplier':multiplier,'capacity_multiplier':capacity,
                'expected_cost_per_1000':float(realized_expected_cost(y[test],a,changed).mean()*1000)})
    pd.DataFrame(sensitivity).to_csv(out/'sensitivity.csv',index=False)
    fig,axs = plt.subplots(1,2,figsize=(10,4))
    for name,pred in predictions.items():
        precision,recall,_ = precision_recall_curve(y[test]!=0,pred[:,1]+pred[:,2])
        axs[0].plot(recall,precision,label=name)
    axs[0].set(xlabel='Recall',ylabel='Precision',title='Synthetic adverse outcome PR'); axs[0].legend(fontsize=7)
    risk = p[:,1]+p[:,2]
    bins = pd.cut(risk,bins=np.linspace(0,1,11),include_lowest=True)
    reliability = pd.DataFrame({'bin':bins,'score':risk,'outcome':y[test]!=0}).groupby('bin',observed=True).agg(mean_score=('score','mean'),rate=('outcome','mean'),count=('outcome','size'))
    reliability.to_csv(out/'calibration_bins.csv')
    axs[1].plot([0,1],[0,1],'--',color='gray'); axs[1].plot(reliability.mean_score,reliability.rate,'o-')
    axs[1].set(xlabel='Predicted adverse probability',ylabel='Observed synthetic rate',title='Held-out calibration')
    fig.tight_layout(); fig.savefig(out/'evaluation.png',dpi=150); plt.close(fig)
    explanations = []
    # All routed claims receive explanations in the deliverable.
    for position,index in enumerate(test):
        if actions[position] != 0:
            explanations.append({'claim_id':claims[index]['claim_id'],'queue':QUEUES[actions[position]],
                'reasons':explain(model,X.iloc[index].to_dict(),reference,cfg)})
    save_json(out/'review_explanations.json',explanations)
    bundle = {'model':model,'store':store,'config':cfg,'penalty':penalty,'reference':reference,
              'selected_model':selected,'model_version':'1.0','policy_version':'1.0',
              'data_mode':'DEMO Option C' if args.demo else 'CMS DE-SynPUF Sample 1 augmented'}
    joblib.dump(bundle,out/'bundle.joblib')
    save_json(out/'example_claim.json',claims[test[0]])
    save_json(out/'run_manifest.json',{'config':cfg,'mode':bundle['data_mode'],'python':platform.python_version(),
               'model':selected,'penalty':penalty,'warmup_end':store.reference_end,
               'class_order':['clean','denial_risk','fwa_risk'],
               'note':'artifact includes immutable historical input snapshot; it contains no outcome labels'})
    write_note(out,report,selected,bundle['data_mode'])
    print(pd.DataFrame(report).T.to_string(),flush=True)
    print(f'Artifacts saved in {out.resolve()}',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw',default='data/raw')
    parser.add_argument('--config',default='config.json')
    parser.add_argument('--output',default='artifacts')
    parser.add_argument('--demo',action='store_true')
    parser.add_argument('--demo-rows',type=int,default=9000)
    run(parser.parse_args())
