import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

try:
    from rapidfuzz import process as rapidfuzz_process, fuzz
except Exception:  # pragma: no cover
    rapidfuzz_process = None
    fuzz = None

from src.api_lookup import lookup_drug, _extract_ingredient_parts

ROOT = Path(__file__).resolve().parents[1]
MASTER_INTERACTIONS_PATH = ROOT / 'data' / 'processed' / 'master_interactions.csv'
DRUG_MAPPING_PATH = ROOT / 'data' / 'processed' / 'drug_mapping.csv'

# Small verified alias layer for common demo/user terms. This is intentionally narrow and
# only meant to normalize user input; it does not replace RxNorm.
SAFE_ALIASES = {
    'aspirin': 'acetylsalicylic acid',
    'asa': 'acetylsalicylic acid',
    'اسبرين': 'acetylsalicylic acid',
    'أسبرين': 'acetylsalicylic acid',
    'الاسبرين': 'acetylsalicylic acid',
    'الأسبرين': 'acetylsalicylic acid',
    'panadol': 'acetaminophen',
    'pandol': 'acetaminophen',
    'بنادول': 'acetaminophen',
    'بانادول': 'acetaminophen',
    'البنادول': 'acetaminophen',
    'البانادول': 'acetaminophen',
    'bnadol': 'acetaminophen',
    'paracetamol': 'acetaminophen',
    'paracetmol': 'acetaminophen',
    'tylenol': 'acetaminophen',
    'doliprane': 'acetaminophen',
    'dolipran': 'acetaminophen',
    'advil': 'ibuprofen',
    'motrin': 'ibuprofen',
    'nurofen': 'ibuprofen',
    'brufen': 'ibuprofen',
    'ايبوبروفين': 'ibuprofen',
    'إيبوبروفين': 'ibuprofen',
    'الايبوبروفين': 'ibuprofen',
    'الإيبوبروفين': 'ibuprofen',
    'بروفين': 'ibuprofen',
    'البروفين': 'ibuprofen',
    'amoxicillin': 'amoxicillin',
    'amoxicilin': 'amoxicillin',
    'glucophage': 'metformin',
    'simvastatin': 'simvastatin',
    'سيمفاستاتين': 'simvastatin',
    'السيمفاستاتين': 'simvastatin',
    'zocor': 'simvastatin',
    'coumadin': 'warfarin',
    'jantoven': 'warfarin',
    'prilosec': 'omeprazole',
    'losec': 'omeprazole',
    'lipitor': 'atorvastatin',
    'norvasc': 'amlodipine',
    'prinivil': 'lisinopril',
    'zestril': 'lisinopril',
    'zithromax': 'azithromycin',
    'vibramycin': 'doxycycline',
    'valium': 'diazepam',
    'xanax': 'alprazolam',
    'zoloft': 'sertraline',
    'prozac': 'fluoxetine',
    'aleve': 'naproxen',
    'naprosyn': 'naproxen',
    'metformin': 'metformin',
}


def normalize_text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ''
    s = unicodedata.normalize('NFKC', str(value)).strip().lower()
    s = ''.join(char for char in s if unicodedata.category(char) != 'Mn')
    s = s.replace('\u0640', '')
    s = re.sub(r'[^\w\s\u0600-\u06ff]', ' ', s)
    s = re.sub(r'[\u060c\u061b\u061f\u064b-\u065f\u066a\u06d4]', ' ', s)
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


_MASTER_INTERACTIONS_CACHE = None
_DRUG_MAPPING_CACHE = None

def _get_master_interactions_df(ddinter_csv_path):
    global _MASTER_INTERACTIONS_CACHE
    if _MASTER_INTERACTIONS_CACHE is not None:
        return _MASTER_INTERACTIONS_CACHE
    if not ddinter_csv_path.exists():
        return None
    _MASTER_INTERACTIONS_CACHE = pd.read_csv(ddinter_csv_path, dtype=str)
    return _MASTER_INTERACTIONS_CACHE


def _load_ddinter_index(ddinter_csv_path=MASTER_INTERACTIONS_PATH):
    df = _get_master_interactions_df(ddinter_csv_path)
    if df is None:
        return {}

    names = pd.concat([df['drug_a'], df['drug_b']], ignore_index=True).astype(str)
    index = {}
    for name in names:
        key = normalize_text(name)
        if key:
            index.setdefault(key, []).append(str(name))
    return index


def _load_drug_mapping_index(mapping_path=DRUG_MAPPING_PATH):
    global _DRUG_MAPPING_CACHE
    if _DRUG_MAPPING_CACHE is not None:
        return _DRUG_MAPPING_CACHE
    if not mapping_path.exists():
        return {}
    df = pd.read_csv(mapping_path, dtype=str).fillna('')
    index = {}
    for row in df.itertuples(index=False):
        term = normalize_text(getattr(row, 'user_term', ''))
        canonical = normalize_text(getattr(row, 'generic_name', ''))
        if term and canonical:
            index.setdefault(term, set()).add(canonical)
    _DRUG_MAPPING_CACHE = index
    return index


def suggest_drug(term, mapping_path=DRUG_MAPPING_PATH):
    """Return a high-confidence local mapping suggestion without resolving it."""
    normalized = normalize_text(term)
    index = _load_drug_mapping_index(mapping_path)
    if not normalized or not index or rapidfuzz_process is None or fuzz is None:
        return None

    match = rapidfuzz_process.extractOne(normalized, list(index), scorer=fuzz.ratio)
    if not match or match[1] < 0.86:
        return None
    candidate, score, _ = match
    candidates = index.get(candidate, set())
    if len(candidates) != 1:
        return None
    canonical = next(iter(candidates))
    return {
        'user_term': candidate,
        'generic_name': canonical,
        'confidence': round(score / 100, 3),
    }


def map_to_ddinter(generic_name, ddinter_csv_path=MASTER_INTERACTIONS_PATH):
    generic_norm = normalize_text(generic_name)
    if not generic_norm:
        return {'ddinter_name': None, 'ddinter_rows': 0, 'matches': []}

    df = _get_master_interactions_df(ddinter_csv_path)
    if df is None:
        return {'ddinter_name': None, 'ddinter_rows': 0, 'matches': []}
    exact_mask = ((df['drug_a'].map(normalize_text) == generic_norm) |
                  (df['drug_b'].map(normalize_text) == generic_norm))
    matches = df.loc[exact_mask, ['drug_a', 'drug_b', 'interaction_level', 'support_count']]

    if matches.empty:
        return {'ddinter_name': None, 'ddinter_rows': 0, 'matches': []}

    ddinter_id = None
    mapping_index = _load_drug_mapping_index()
    mapped_names = mapping_index.get(generic_norm, set())
    if len(mapped_names) == 1:
        mapping_df = pd.read_csv(DRUG_MAPPING_PATH, dtype=str).fillna('')
        ids = mapping_df.loc[
            mapping_df['generic_name'].map(normalize_text) == generic_norm, 'ddinter_id'
        ].dropna().unique().tolist()
        if len(ids) == 1:
            ddinter_id = ids[0]

    return {
        'ddinter_name': generic_name,
        'ddinter_id': ddinter_id,
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
        'suggestion': None,
        'ddinter_mapping': {'ddinter_name': None, 'ddinter_rows': 0, 'matches': []},
    }

    if not normalized_input:
        result['reason'] = 'Empty input.'
        return result

    # 1) dataset-derived local mapping and verified aliases
    local_candidates = _load_drug_mapping_index().get(normalized_input, set())
    if normalized_input in SAFE_ALIASES:
        local_candidates = {SAFE_ALIASES[normalized_input]}
    if len(local_candidates) == 1:
        canonical = next(iter(local_candidates))
        result.update({
            'resolved_name': canonical,
            'generic_name': canonical,
            'active_ingredients': [canonical],
            'match_type': 'alias',
            'confidence': 1.0,
            'verified': True,
            'source': 'local_alias',
            'reason': 'Verified local mapping normalized to a DDInter drug name.'
        })
        result['ddinter_mapping'] = map_to_ddinter(canonical, ddinter_csv_path)
        return result

    # 2) RxNav API: only accept exact, non-fuzzy matches that map directly to the term or its ingredient
    api_result = lookup_drug(raw_input)
    if api_result and (api_result.get('generic_name') or api_result.get('ingredients')):
        exact = _pick_exact_api_candidate(raw_input, api_result)
        if exact:
            exact_norm = normalize_text(exact)
            ddinter_mapping = map_to_ddinter(exact, ddinter_csv_path)
            if not ddinter_mapping['ddinter_rows']:
                exact = None
            else:
                result.update({
                    'resolved_name': exact,
                    'generic_name': exact,
                    'active_ingredients': [item for item in _safe_list(_extract_ingredient_parts(exact)) if item],
                    'match_type': 'api',
                    'confidence': 0.99,
                    'verified': True,
                    'source': 'rxnav',
                    'reason': 'Exact RxNav generic/ingredient match confirmed in DDInter.',
                    'ddinter_mapping': ddinter_mapping,
                })
                return result

    # 3) Safety gate: ambiguous or no_match only. Never fuzzy-match against DDInter to decide.
    result['match_type'] = 'ambiguous' if normalized_input else 'no_match'
    result['reason'] = 'The term is not a verified exact alias or direct RxNav match; fuzzy matching is not used to decide the identity.'
    result['suggestion'] = suggest_drug(raw_input)
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
