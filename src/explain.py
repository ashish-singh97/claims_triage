"""Local single-feature perturbations; associations, not causal explanations."""
import numpy as np
import pandas as pd
from src.features import FEATURES


def reason_text(feature, value):
    if feature == 'primary_diagnosis_group':
        return f'The primary ICD-9 category is {value}.'
    if feature == 'primary_procedure_group':
        return f'The primary procedure demonstration group is {value}.'
    value = float(value)
    messages = {
        'log_amount': f'Total submitted amount is ${np.expm1(value):,.2f}.',
        'line_count': f'This claim contains {value:.0f} service lines.',
        'diagnosis_count': f'This claim contains {value:.0f} distinct diagnosis codes.',
        'provider_count': f'This claim contains {value:.0f} distinct providers.',
        'service_days': f'The claim spans {value:.0f} service days.',
        'age': f'The member age at service is approximately {value:.0f} years.',
        'age_missing': 'Member age is unavailable or outside the supported range.',
        'diagnosis_missing': 'No claim-level diagnosis was supplied.',
        'provider_unknown': 'At least one service line has an unknown provider.',
        'member_claims_30d': f'This member has {value:.0f} earlier submissions in the previous 30 days.',
        'provider_claims_7d_max': f'A provider on this claim has {value:.0f} earlier claims in the previous 7 days.',
        'provider_claims_30d_max': f'A provider on this claim has {value:.0f} earlier claims in the previous 30 days.',
        'provider_recent_ratio': f'The largest provider recent-volume ratio is {value:.1f} relative to the preceding 23-day rate.',
        'duplicate_matches_30d': f'{value:.0f} earlier submissions match this member, providers, procedures and service dates within 30 days; repeated services may be legitimate.',
        'amount_peer_ratio_max': f'One submitted service amount is {value:.1f} times the warm-up-period median for its procedure (or fallback).',
        'pair_rarity_max': f'A diagnosis/procedure group pairing has historical rarity score {value:.2f}; this does not establish medical incompatibility.',
        'unseen_code': 'At least one procedure was absent from the historical reference period.',
        'primary_diagnosis_group': f'The primary ICD-9 category is {value}.',
        'primary_procedure_group': f'The primary procedure demonstration group is {value}.'}
    return messages[feature]


def explain(model, row, reference, cfg, top_k=3):
    original = pd.DataFrame([row], columns=FEATURES)
    p = model.predict_proba(original)[0]
    variants = []
    for feature in FEATURES:
        replacement = dict(row)
        replacement[feature] = reference[feature]
        variants.append(replacement)
    predictions = model.predict_proba(pd.DataFrame(variants, columns=FEATURES))
    weights = np.array([0,cfg['denial_loss'],cfg['fwa_loss']])
    deltas = p @ weights - predictions @ weights
    indexes = np.argsort(-deltas)
    reasons = []
    for index in indexes:
        if deltas[index] <= 1e-6:
            continue
        feature = FEATURES[index]
        value = row[feature]
        if pd.isna(value):
            continue
        reasons.append({'reason': reason_text(feature, value),
                        'local_loss_contribution': round(float(deltas[index]), 4)})
        if len(reasons) == top_k:
            break
    if not reasons:
        reasons = [{'reason': 'The combined model score exceeds the policy threshold; no positive single-feature perturbation reason was found.',
                    'local_loss_contribution': 0.0}]
    return reasons
