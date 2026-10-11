"""Read CMS CSV/ZIP files in chunks; retain complete sampled-member histories."""
from pathlib import Path
import hashlib
import json
import zipfile
import re
import numpy as np
import pandas as pd


def csv_sources(raw):
    for path in sorted(Path(raw).iterdir()):
        if path.suffix.lower() == '.csv':
            yield path, None
        elif path.suffix.lower() == '.zip':
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    if name.lower().endswith('.csv'):
                        yield path, name


def chunks(path, name, size):
    if name is None:
        yield from pd.read_csv(path, dtype=str, chunksize=size)
    else:
        with zipfile.ZipFile(path) as archive, archive.open(name) as stream:
            yield from pd.read_csv(stream, dtype=str, chunksize=size)


def load_cms(raw, cfg):
    sources = list(csv_sources(raw))
    carriers = [(p, n) for p, n in sources if 'carrier' in (n or p.name).lower()]
    if len(carriers) != 2:
        raise ValueError('Place exactly Carrier Sample 1A and 1B CSVs OR ZIPs in data/raw; do not keep both extracted and ZIP copies.')
    if any(not re.search(r'sample_1[ab](?:\.|$)', (n or p.name).lower()) for p, n in carriers):
        raise ValueError('This implementation expects CMS Sample 1 files.')
    members = set()
    for p, n in carriers:
        for chunk in chunks(p, n, cfg['chunksize']):
            members.update(chunk['DESYNPUF_ID'].dropna().astype(str))
    rng = np.random.default_rng(cfg['seed'])
    selected = set(rng.choice(sorted(members), min(cfg['sample_members'], len(members)), replace=False))
    pieces = []
    for p, n in carriers:
        for chunk in chunks(p, n, cfg['chunksize']):
            pieces.append(chunk[chunk.DESYNPUF_ID.isin(selected)].copy())
    source = pd.concat(pieces, ignore_index=True)
    rows_before = len(source)
    source = source.drop_duplicates()  # exact ingestion duplicates only
    required = ['DESYNPUF_ID', 'CLM_ID', 'CLM_FROM_DT', 'CLM_THRU_DT']
    if any(c not in source for c in required):
        raise ValueError('Unexpected carrier schema: missing claim identifiers/dates.')
    if source.CLM_ID.duplicated().any():
        raise ValueError('Conflicting repeated CLM_ID records: investigate before aggregation.')
    claims, rejected = [], []
    for row in source.to_dict('records'):
        start = pd.to_datetime(row['CLM_FROM_DT'], format='%Y%m%d', errors='coerce')
        end = pd.to_datetime(row['CLM_THRU_DT'], format='%Y%m%d', errors='coerce')
        if pd.isna(start) or pd.isna(end) or end < start:
            rejected.append({'claim_id': row['CLM_ID'], 'reason': 'invalid_service_dates'})
            continue
        lines = []
        for i in range(1, 14):
            code = clean_code(row.get(f'HCPCS_CD_{i}'))
            if not code:
                continue
            amount = pd.to_numeric(row.get(f'LINE_ALOWD_CHRG_AMT_{i}'), errors='coerce')
            lines.append({'provider_id': clean_code(row.get(f'PRF_PHYSN_NPI_{i}')) or 'UNKNOWN',
                          'hcpcs_code': code, 'reference_amount': float(amount) if pd.notna(amount) else None,
                          'diagnosis_code': clean_code(row.get(f'LINE_ICD9_DGNS_CD_{i}'))})
        if not lines:
            rejected.append({'claim_id': row['CLM_ID'], 'reason': 'no_populated_procedure_lines'})
            continue
        diagnoses = list(dict.fromkeys(clean_code(row.get(f'ICD9_DGNS_CD_{i}')) for i in range(1, 9)))
        diagnoses = [d for d in diagnoses if d]
        claims.append({'claim_id': str(row['CLM_ID']), 'member_id': str(row['DESYNPUF_ID']),
                       'service_from': str(start.date()), 'service_to': str(end.date()),
                       'diagnosis_codes': diagnoses, 'lines': lines})
    # Baseline 2008 DOB is static and safe. Annual condition/reimbursement/death fields are NOT predictors.
    beneficiaries = [(p, n) for p, n in sources if 'beneficiary' in (n or p.name).lower()]
    if any(not re.search(r'sample_1(?:\.|$)', (n or p.name).lower()) for p, n in beneficiaries):
        raise ValueError('Mismatched beneficiary sample supplied; use Sample 1 only.')
    births = {}
    for p, n in beneficiaries:
        if '2008' in (n or p.name):
            for chunk in chunks(p, n, cfg['chunksize']):
                for r in chunk[chunk.DESYNPUF_ID.isin(selected)].to_dict('records'):
                    dob = pd.to_datetime(r.get('BENE_BIRTH_DT'), format='%Y%m%d', errors='coerce')
                    if pd.notna(dob):
                        births[str(r['DESYNPUF_ID'])] = str(dob.date())
    for claim in claims:
        claim['birth_date'] = births.get(claim['member_id'])
    manifest = []
    for p in sorted(set(p for p, _ in sources)):
        digest = hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        manifest.append({'file': p.name, 'sha256': digest.hexdigest()})
    audit = {'raw_sample_rows': rows_before, 'exact_duplicates_removed': rows_before-len(source),
             'selected_members': len(selected), 'accepted_claims': len(claims),
             'rejected_count': len(rejected), 'rejected': rejected, 'source_manifest': manifest,
             'birth_dates_available': len(births)}
    if not claims:
        raise ValueError('No valid claims after cleaning.')
    return claims, audit


def clean_code(value):
    if value is None or pd.isna(value):
        return ''
    return str(value).strip().upper().replace('.', '')


def demo_claims(n=9000, seed=42):
    """Option C fixture for testing only; never present as downloaded CMS data."""
    rng = np.random.default_rng(seed)
    procedures = ['99213', '99214', '93000', '80053', '71020', 'J1100']
    diagnoses = ['25000', '4019', '78650', '2724', '49390', 'V700']
    result = []
    for i in range(n):
        day = pd.Timestamp('2008-01-01') + pd.Timedelta(days=int(rng.integers(0, 1080)))
        j = int(rng.integers(0, len(procedures)))
        result.append({'claim_id': f'DEMO-{i}', 'member_id': f'M{rng.integers(0, 600)}',
                       'service_from': str(day.date()), 'service_to': str(day.date()),
                       'birth_date': '1940-01-01', 'diagnosis_codes': [diagnoses[j]],
                       'lines': [{'provider_id': f'P{rng.integers(0, 80)}', 'hcpcs_code': procedures[j],
                                  'diagnosis_code': diagnoses[j],
                                  'reference_amount': float(rng.lognormal(4.4+j*0.12, 0.45))}]})
    return result


def split_name(date, cfg):
    if date < cfg['train_end']:
        return 'train'
    if date < cfg['calibration_end']:
        return 'calibration'
    if date < cfg['policy_end']:
        return 'policy'
    return 'test'


def simulate(claims, cfg):
    """Draw noisy latent outcomes from mutations, never expose metadata to model."""
    import copy
    rng = np.random.default_rng(cfg['seed'])
    historical = {}
    for c in claims:
        if c['service_to'] < cfg['train_end']:
            for line in c['lines']:
                v = line['reference_amount']
                if v is not None and v > 0:
                    historical.setdefault(line['hcpcs_code'], []).append(v)
    fallback = float(np.median([v for values in historical.values() for v in values])) if historical else 100.0
    medians = {k: float(np.median(v)) for k, v in historical.items()}
    by_partition = {k: [] for k in ['train', 'calibration', 'policy', 'test']}
    output, metadata = [], []
    all_diagnoses = sorted({d for c in claims if c['service_to'] < cfg['train_end'] for d in c['diagnosis_codes']}) or ['25000']
    for source in sorted(claims, key=lambda c: (c['service_to'], c['claim_id'])):
        c = copy.deepcopy(source)
        # Submission timestamp is explicitly simulated; same-day history is excluded later.
        c['submitted_at'] = str(pd.Timestamp(c['service_to']) + pd.Timedelta(days=1))
        partition = split_name(c['submitted_at'][:10], cfg)
        scenario = rng.choice(['ordinary', 'inflation', 'duplicate', 'pairing', 'burst'], p=[.86, .04, .04, .03, .03])
        # Use allowed charge only as a simulation scale, never retain it in model/API data.
        for line in c['lines']:
            ref = line.pop('reference_amount')
            ref = ref if ref is not None and ref > 0 else medians.get(line['hcpcs_code'], fallback)
            line['submitted_amount'] = float(max(1.0, ref * rng.lognormal(.25, .25)))
        original_id = c['claim_id']
        if scenario == 'inflation':
            for line in c['lines']:
                line['submitted_amount'] *= float(rng.uniform(2.0, 7.0))
        elif scenario == 'duplicate':
            candidates = by_partition[partition]
            if candidates:
                # Resubmit an earlier claim within this same temporal partition.
                old = candidates[int(rng.integers(max(0, len(candidates)-300), len(candidates)))]
                submission = c['submitted_at']
                c = copy.deepcopy(old)
                c['claim_id'] = original_id
                c['submitted_at'] = submission
                original_id = old['claim_id']
            else:
                scenario = 'ordinary'
        elif scenario == 'pairing':
            c['diagnosis_codes'] = [str(rng.choice(all_diagnoses))]
            for line in c['lines']:
                line['diagnosis_code'] = c['diagnosis_codes'][0]
        elif scenario == 'burst':
            # Artificial cluster spans days so prior-day history can detect it.
            cluster = pd.Timestamp(c['submitted_at']).dayofyear // 14
            for line in c['lines']:
                line['provider_id'] = f'SIM-PROVIDER-{partition}-{cluster % 2}'
        probabilities = {'ordinary': [.985, .012, .003], 'inflation': [.40, .30, .30],
                         'duplicate': [.30, .65, .05], 'pairing': [.75, .23, .02],
                         'burst': [.65, .15, .20]}[scenario]
        y = int(rng.choice(3, p=probabilities))
        output.append(c)
        metadata.append({'claim_id': c['claim_id'], 'source_claim_id': original_id,
                         'scenario': scenario, 'target': y, 'partition': partition})
        by_partition[partition].append(c)
    return output, pd.DataFrame(metadata)
