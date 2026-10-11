"""Same point-in-time feature implementation for offline and online scoring."""
from bisect import bisect_left
from collections import defaultdict
import numpy as np
import pandas as pd

NUMERIC = ['log_amount', 'line_count', 'diagnosis_count', 'provider_count', 'service_days',
           'age', 'age_missing', 'diagnosis_missing', 'provider_unknown', 'member_claims_30d',
           'provider_claims_7d_max', 'provider_claims_30d_max', 'provider_recent_ratio',
           'duplicate_matches_30d', 'amount_peer_ratio_max', 'pair_rarity_max', 'unseen_code']
CATEGORICAL = ['primary_diagnosis_group', 'primary_procedure_group']
FEATURES = NUMERIC + CATEGORICAL


def diagnosis_group(code):
    code = str(code or '').upper().replace('.', '')
    return code[:3] if code and code[0] not in 'VE' else code[:3] or 'UNKNOWN'


def procedure_group(code):
    """Broad demonstration groups, not a validated clinical/coding hierarchy."""
    code = str(code or '').upper()
    if not code:
        return 'UNKNOWN'
    if not code.isdigit():
        return f'HCPCS_{code[0]}'
    n = int(code)
    if 99201 <= n <= 99499:
        return 'evaluation_management'
    if 70000 <= n <= 79999:
        return 'radiology'
    if 80000 <= n <= 89999:
        return 'laboratory'
    if 90000 <= n <= 99199:
        return 'medicine'
    return 'other_numeric'


def duplicate_key(c):
    # Ignore claim ID/amount; legitimate repeated services can match this key.
    return (c['member_id'], c['service_from'], c['service_to'],
            tuple(sorted((l['provider_id'], l['hcpcs_code']) for l in c['lines'])))


class FeatureStore:
    """Immutable snapshot; all historical queries exclude the submission day."""
    def __init__(self, claims, reference_end='2008-04-01'):
        self.member = defaultdict(list)
        self.provider = defaultdict(list)
        self.duplicates = defaultdict(list)
        peer = defaultdict(list)
        pairs = defaultdict(int)
        diagnosis_totals = defaultdict(int)
        reference_count = 0
        for c in sorted(claims, key=lambda r: r['submitted_at']):
            day = pd.Timestamp(c['submitted_at']).normalize().value // 86400000000000
            self.member[c['member_id']].append(day)
            for provider in set(l['provider_id'] for l in c['lines']):
                self.provider[provider].append(day)
            self.duplicates[duplicate_key(c)].append(day)
            if c['submitted_at'][:10] < reference_end:
                reference_count += 1
                for l in c['lines']:
                    peer[l['hcpcs_code']].append(l['submitted_amount'])
                    dg = diagnosis_group(l.get('diagnosis_code') or next(iter(c['diagnosis_codes']), ''))
                    pairs[(dg, procedure_group(l['hcpcs_code']))] += 1
                    diagnosis_totals[dg] += 1
        if reference_count < 20:
            raise ValueError('Need at least 20 warm-up claims before 2008-04-01 for point-in-time references.')
        self.peer = {k: float(np.median(v)) for k, v in peer.items()}
        self.global_peer = float(np.median([v for values in peer.values() for v in values]))
        self.pairs = dict(pairs)
        self.diagnosis_totals = dict(diagnosis_totals)
        self.reference_end = reference_end
        self.history_as_of = max(c['submitted_at'] for c in claims)

    @staticmethod
    def count(index, key, day, window):
        dates = index.get(key, [])
        return bisect_left(dates, day) - bisect_left(dates, day-window)

    def row(self, c):
        timestamp = pd.Timestamp(c['submitted_at'])
        if timestamp.date().isoformat() < self.reference_end:
            raise ValueError('Score date precedes the feature reference snapshot.')
        day = timestamp.normalize().value // 86400000000000
        providers = set(l['provider_id'] for l in c['lines'])
        p7 = [self.count(self.provider, p, day, 7) for p in providers]
        p30 = [self.count(self.provider, p, day, 30) for p in providers]
        ratios = [(a+1)/((b-a)/23*7+1) for a, b in zip(p7, p30)]
        peers = [l['submitted_amount']/max(self.peer.get(l['hcpcs_code'], self.global_peer), 1)
                 for l in c['lines']]
        rarity = []
        for line in c['lines']:
            dg = diagnosis_group(line.get('diagnosis_code') or next(iter(c['diagnosis_codes']), ''))
            pg = procedure_group(line['hcpcs_code'])
            frequency = (self.pairs.get((dg, pg), 0)+1)/(self.diagnosis_totals.get(dg, 0)+10)
            rarity.append(-np.log(max(frequency, 1e-9)))
        birth = pd.to_datetime(c.get('birth_date'), errors='coerce')
        age = (pd.Timestamp(c['service_from'])-birth).days/365.25 if pd.notna(birth) else np.nan
        if pd.notna(age) and not 0 <= age <= 120:
            age = np.nan
        return {'log_amount': np.log1p(sum(l['submitted_amount'] for l in c['lines'])),
                'line_count': len(c['lines']), 'diagnosis_count': len(set(c['diagnosis_codes'])),
                'provider_count': len(providers),
                'service_days': (pd.Timestamp(c['service_to'])-pd.Timestamp(c['service_from'])).days+1,
                'age': age, 'age_missing': int(pd.isna(age)),
                'diagnosis_missing': int(not c['diagnosis_codes']),
                'provider_unknown': int('UNKNOWN' in providers),
                'member_claims_30d': self.count(self.member, c['member_id'], day, 30),
                'provider_claims_7d_max': max(p7, default=0),
                'provider_claims_30d_max': max(p30, default=0),
                'provider_recent_ratio': max(ratios, default=1),
                'duplicate_matches_30d': self.count(self.duplicates, duplicate_key(c), day, 30),
                'amount_peer_ratio_max': max(peers), 'pair_rarity_max': max(rarity),
                'unseen_code': int(any(l['hcpcs_code'] not in self.peer for l in c['lines'])),
                'primary_diagnosis_group': diagnosis_group(next(iter(c['diagnosis_codes']), '')),
                'primary_procedure_group': procedure_group(c['lines'][0]['hcpcs_code'])}

    def transform(self, claims):
        return pd.DataFrame([self.row(c) for c in claims], columns=FEATURES)
