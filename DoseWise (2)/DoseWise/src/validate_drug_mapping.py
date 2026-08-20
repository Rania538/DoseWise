import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover
    fuzz = None

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / 'data' / 'processed' / 'drug_mapping.csv'

TEST_TERMS = [
    'Panadol',
    'panadol',
    'Pandol',
    'بنادول',
    'Paracetamol',
    'Doliprane',
    'Dolipran',
    'Paracetmol',
    'amoxicillin',
    'amoxicilin',
]


def normalize_text(value):
    if pd.isna(value):
        return ''
    value = str(value)
    value = unicodedata.normalize('NFKC', value)
    value = value.strip().lower()
    value = re.sub(r'[^\w\s\u0600-\u06ff]', ' ', value)
    value = re.sub(r'\s+', ' ', value)
    return value.strip()


def similarity(a, b):
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if fuzz is not None:
        s = max(
            fuzz.ratio(a, b),
            fuzz.partial_ratio(a, b),
            fuzz.token_sort_ratio(a, b),
            fuzz.token_set_ratio(a, b),
        )
        return s / 100.0
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()


def split_brand_values(value):
    if pd.isna(value):
        return []
    raw = str(value)
    parts = [p.strip() for p in raw.split(',') if p.strip()]
    out = []
    for p in parts:
        out.extend([x.strip() for x in p.split('/') if x.strip()])
    return out


def classify(term, row_matches):
    if not row_matches:
        return ('no_match', 0.0)
    best = max(row_matches, key=lambda x: x['score'])
    top_score = best['score']
    top_generic = best['generic_name']
    top_candidates = [m for m in row_matches if m['score'] >= 0.8]
    unique_generics = {m['generic_name'] for m in top_candidates if m['generic_name']}

    if top_score >= 0.99 and normalize_text(term) == normalize_text(best['user_term']):
        return ('exact', top_score)
    if top_score >= 0.99 and normalize_text(term) == normalize_text(best['generic_name']):
        return ('API', top_score)
    if top_score >= 0.95 and len(unique_generics) == 1 and best['matched_field'] in {'alias', 'brand_name'}:
        return ('alias', top_score)
    if len(unique_generics) > 1:
        return ('ambiguous', top_score)
    if top_score >= 0.75:
        return ('fuzzy', top_score)
    return ('no_match', top_score)


def main():
    df = pd.read_csv(CSV_PATH, dtype=str)
    df['user_term'] = df['user_term'].fillna('')
    df['generic_name'] = df['generic_name'].fillna('')
    df['active_ingredient'] = df['active_ingredient'].fillna('')

    def get_count_missing(series_col):
        return int(series_col.isna().sum() + (series_col.astype(str).str.strip() == '').sum())

    unique_user_terms = df['user_term'].map(normalize_text).dropna().drop_duplicates().shape[0]
    unique_generic_names = df['generic_name'].map(normalize_text).dropna().drop_duplicates().shape[0]

    ambiguous_rows = 0
    for uname, grp in df.groupby('user_term', dropna=False):
        generic_set = {normalize_text(v) for v in grp['generic_name'].tolist() if normalize_text(v)}
        if len(generic_set) > 1:
            ambiguous_rows += 1

    dupes = int(df.duplicated(subset=['user_term', 'generic_name', 'active_ingredient'], keep=False).sum())

    print('=== DRUG MAPPING VALIDATION REPORT ===')
    print(f'CSV path: {CSV_PATH}')
    print(f'Rows: {len(df)}')
    print(f'Unique user terms: {unique_user_terms}')
    print(f'Unique generic names: {unique_generic_names}')
    print(f'Missing generic_name values: {get_count_missing(df["generic_name"])}')
    print(f'Missing active_ingredient values: {get_count_missing(df["active_ingredient"])}')
    print(f'Ambiguous mappings: {ambiguous_rows}')
    print(f'Duplicate mappings: {dupes}')

    print('\n=== TEST TERM VALIDATION ===')
    for term in TEST_TERMS:
        nq = normalize_text(term)
        candidates = []
        for _, row in df.iterrows():
            user_term = str(row['user_term'])
            generic_name = str(row['generic_name'])
            active_ingredient = str(row['active_ingredient'])
            alias_values = split_brand_values(row.get('aliases', '')) + split_brand_values(row.get('brand_name', ''))

            for field_name, value in [('user_term', user_term), ('generic_name', generic_name), ('active_ingredient', active_ingredient), *[(f'alias_{i}', v) for i, v in enumerate(alias_values)]]:
                if not value or value == 'nan':
                    continue
                score = similarity(term, value)
                if score >= 0.75:
                    candidates.append({
                        'user_term': user_term,
                        'generic_name': generic_name,
                        'active_ingredient': active_ingredient,
                        'score': score,
                        'matched_field': 'alias' if field_name.startswith('alias_') else field_name,
                    })

        if not candidates:
            best_score = 0.0
            best_generic = ''
            best_active = ''
            match_type = 'no_match'
        else:
            best = max(candidates, key=lambda x: x['score'])
            best_score = float(best['score'])
            best_generic = best['generic_name'] or ''
            best_active = best['active_ingredient'] or ''
            match_type, _ = classify(term, candidates)

        print(f'Original user input: {term}')
        print(f'Normalized input: {nq}')
        print(f'Matched generic name: {best_generic}')
        print(f'Active ingredient: {best_active}')
        print(f'Match type: {match_type}')
        print(f'Confidence score: {best_score:.4f}')
        print('---')

    # explicit safety check on Panadol vs nadolol
    panadol_candidates = []
    for _, row in df.iterrows():
        for value in [row['user_term'], row['generic_name'], row['active_ingredient']] + split_brand_values(row.get('aliases', '')) + split_brand_values(row.get('brand_name', '')):
            if value and value != 'nan':
                score = similarity('Panadol', value)
                if score >= 0.75:
                    panadol_candidates.append((value, row['generic_name'], row['active_ingredient'], score))
    panadol_candidates = sorted(panadol_candidates, key=lambda x: x[3], reverse=True)
    print('\nPanadol safety check:')
    if panadol_candidates:
        for item in panadol_candidates[:10]:
            print(item)
    else:
        print('No Panadol candidates found above fuzzy threshold; safe from nadolol confusion.')

    # low-confidence fuzzy mappings count using a conservative threshold
    low_conf_fuzzy = 0
    for _, row in df.iterrows():
        terms_to_check = [row['user_term'], row['generic_name'], row['active_ingredient']] + split_brand_values(row.get('brand_name', ''))
        scores = [similarity(str(t), str(row['user_term'])) for t in terms_to_check if t and t != 'nan']
        if scores:
            best = max(scores)
            if 0.75 <= best < 0.9:
                low_conf_fuzzy += 1
    print(f'Low-confidence fuzzy mappings: {low_conf_fuzzy}')

if __name__ == '__main__':
    main()
