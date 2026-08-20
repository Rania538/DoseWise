import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.api_lookup import lookup_drug

ROOT = Path(__file__).resolve().parents[1]
MASTER_INTERACTIONS_PATH = ROOT / 'data' / 'processed' / 'master_interactions.csv'

# Small verified alias layer for common demo/user terms. This is intentionally narrow and
# only meant to normalize user input; it does not replace RxNorm.
SAFE_ALIASES = {
    'panadol': 'acetaminophen',
    'pandol': 'acetaminophen',
    'بنادول': 'acetaminophen',
    'bnadol': 'acetaminophen',
    'paracetamol': 'acetaminophen',
    'paracetmol': 'acetaminophen',
    'acetaminophen': 'acetaminophen',
    'doliprane': 'acetaminophen',
    'dolipran': 'acetaminophen',
    'amoxicillin': 'amoxicillin',
    'amoxicilin': 'amoxicillin',
    'simvastatin': 'simvastatin',
    'سيمفاستاتين': 'simvastatin',
    'metformin': 'metformin',
}


def normalize_text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ''
    s = unicodedata.normalize('NFKC', str(value)).strip().lower()
    s = re.sub(r'[^\w\s\u0600-\u06ff]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def _safe_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [p.strip() for p in value.split(',') if p.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()]


def _pick_exact_api_candidate(term, api_result):
    norm_term = normalize_text(term)
    if not api_result:
        return None

    generic_name = api_result.get('generic_name')
    if generic_name:
        generic_norm = normalize_text(generic_name)
        if generic_norm == norm_term:
            return generic_name

    ingredients = api_result.get('ingredients') or []
    for item in ingredients:
        if normalize_text(item) == norm_term:
            return item

    for name in api_result.get('brand_names') or []:
        if normalize_text(name) == norm_term:
            return generic_name or (ingredients[0] if ingredients else None)

    return None


def _load_ddinter_index(ddinter_csv_path=MASTER_INTERACTIONS_PATH):
    if not ddinter_csv_path.exists():
        return {}

    df = pd.read_csv(ddinter_csv_path, dtype=str)
    names = pd.concat([df['drug_a'], df['drug_b']], ignore_index=True).astype(str)
    index = {}
    for name in names:
        key = normalize_text(name)
        if key:
            index.setdefault(key, []).append(str(name))
    return index


def map_to_ddinter(generic_name, ddinter_csv_path=MASTER_INTERACTIONS_PATH):
    generic_norm = normalize_text(generic_name)
    if not generic_norm:
        return {'ddinter_name': None, 'ddinter_rows': 0, 'matches': []}

    if not ddinter_csv_path.exists():
        return {'ddinter_name': None, 'ddinter_rows': 0, 'matches': []}

    df = pd.read_csv(ddinter_csv_path, dtype=str)
    exact_mask = ((df['drug_a'].map(normalize_text) == generic_norm) |
                  (df['drug_b'].map(normalize_text) == generic_norm))
    matches = df.loc[exact_mask, ['drug_a', 'drug_b', 'interaction_level', 'support_count']]

    if matches.empty:
        return {'ddinter_name': None, 'ddinter_rows': 0, 'matches': []}

    return {
        'ddinter_name': generic_name,
        'ddinter_rows': int(len(matches)),
        'matches': matches.head(5).to_dict(orient='records'),
    }


def resolve_drug(term, ddinter_csv_path=MASTER_INTERACTIONS_PATH):
    raw_input = term if term is not None else ''
    normalized_input = normalize_text(raw_input)

    result = {
        'input': str(raw_input),
        'normalized_input': normalized_input,
        'resolved_name': None,
        'generic_name': None,
        'active_ingredients': [],
        'rxcui': None,
        'match_type': 'no_match',
        'confidence': 0.0,
        'verified': False,
        'source': 'none',
        'reason': 'No drug identity resolved.',
        'ddinter_mapping': {'ddinter_name': None, 'ddinter_rows': 0, 'matches': []},
    }

    if not normalized_input:
        result['reason'] = 'Empty input.'
        return result

    # 1) local alias map: exact, verified alias normalization only
    if normalized_input in SAFE_ALIASES:
        canonical = SAFE_ALIASES[normalized_input]
        result.update({
            'resolved_name': canonical,
            'generic_name': canonical,
            'active_ingredients': [canonical],
            'match_type': 'alias',
            'confidence': 1.0,
            'verified': True,
            'source': 'local_alias',
            'reason': 'Verified local alias normalized to a known drug name.'
        })
        result['ddinter_mapping'] = map_to_ddinter(canonical, ddinter_csv_path)
        return result

    # 2) RxNav API: only accept exact, non-fuzzy matches that map directly to the term or its ingredient
    api_result = lookup_drug(raw_input)
    if api_result and (api_result.get('generic_name') or api_result.get('ingredients')):
        exact = _pick_exact_api_candidate(raw_input, api_result)
        if exact:
            exact_norm = normalize_text(exact)
            result.update({
                'resolved_name': exact,
                'generic_name': exact,
                'active_ingredients': [item for item in _safe_list(api_result.get('ingredients')) if item],
                'match_type': 'api',
                'confidence': 0.99,
                'verified': True,
                'source': 'rxnav',
                'reason': 'Exact RxNav generic/ingredient match; no fuzzy DDInter comparison used.'
            })
            result['ddinter_mapping'] = map_to_ddinter(exact, ddinter_csv_path)
            return result

    # 3) Safety gate: ambiguous or no_match only. Never fuzzy-match against DDInter to decide.
    result['match_type'] = 'ambiguous' if normalized_input else 'no_match'
    result['reason'] = 'The term is not a verified exact alias or direct RxNav match; fuzzy matching is not used to decide the identity.'
    return result


if __name__ == '__main__':
    test_terms = [
        'Panadol',
        'panadol',
        'Pandol',
        'بنادول',
        'bnadol',
        'Paracetamol',
        'Paracetmol',
        'Acetaminophen',
        'Doliprane',
        'Dolipran',
        'amoxicillin',
        'amoxicilin',
    ]

    print('=== SAFE DRUG RESOLVER TEST REPORT ===')
    for term in test_terms:
        result = resolve_drug(term)
        print(json.dumps({
            'input': result['input'],
            'normalized_input': result['normalized_input'],
            'resolved_name': result['resolved_name'],
            'generic_name': result['generic_name'],
            'active_ingredients': result['active_ingredients'],
            'rxcui': result['rxcui'],
            'match_type': result['match_type'],
            'confidence': result['confidence'],
            'verified': result['verified'],
            'source': result['source'],
            'reason': result['reason'],
            'ddinter_matches': result['ddinter_mapping']['ddinter_rows'],
        }, ensure_ascii=False, indent=2))
        print('---')
