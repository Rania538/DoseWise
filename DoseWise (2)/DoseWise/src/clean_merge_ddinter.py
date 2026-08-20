import pandas as pd
import unicodedata
import re
import json
from pathlib import Path


def normalize_name(s: str) -> str:
    if pd.isna(s):
        return ''
    s = str(s)
    s = unicodedata.normalize('NFKC', s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    # remove punctuation but keep Arabic letters (\u0600-\u06FF) and basic word chars
    s = re.sub(r"[^\w\s\u0600-\u06FF]", " ", s)
    s = s.replace('_', ' ')
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def priority_level(levels):
    # levels: iterable of strings
    rank = {'major': 3, 'moderate': 2, 'minor': 1}
    best = 'minor'
    best_score = 0
    for l in levels:
        if not isinstance(l, str):
            continue
        score = rank.get(l.strip().lower(), 0)
        if score > best_score:
            best_score = score
            # canonicalize capitalization
            if score == 3:
                best = 'Major'
            elif score == 2:
                best = 'Moderate'
            elif score == 1:
                best = 'Minor'
    return best


def main():
    base = Path('d:/Documents/tessst')
    files = [base / 'ddinter_downloads_code_A.csv', base / 'ddinter_downloads_code_B.csv', base / 'ddinter_downloads_code_R.csv']
    dfs = []
    for f in files:
        print('Loading', f.name)
        df = pd.read_csv(f, dtype=str)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    original_rows = len(combined)

    # basic cleaning: strip names, normalize unicode, collapse spaces
    combined['Drug_A_orig'] = combined['Drug_A']
    combined['Drug_B_orig'] = combined['Drug_B']
    combined['Drug_A_norm'] = combined['Drug_A'].map(normalize_name)
    combined['Drug_B_norm'] = combined['Drug_B'].map(normalize_name)

    # remove exact duplicate rows (all columns equal)
    before = len(combined)
    combined = combined.drop_duplicates()
    after = len(combined)
    exact_dups = before - after

    # build unordered pair key using normalized names
    combined['pair_key'] = combined.apply(lambda r: '||'.join(sorted([r['Drug_A_norm'], r['Drug_B_norm']])), axis=1)

    # track mapping of normalized -> original variants
    norm_map = {}
    for _, r in combined.iterrows():
        a_norm = r['Drug_A_norm']
        b_norm = r['Drug_B_norm']
        norm_map.setdefault(a_norm, set()).add(r['Drug_A_orig'])
        norm_map.setdefault(b_norm, set()).add(r['Drug_B_orig'])

    # aggregate by pair_key and interaction Level
    groups = combined.groupby('pair_key')

    rows = []
    for key, g in groups:
        norms = key.split('||')
        drug_a_norm, drug_b_norm = norms[0], norms[1] if len(norms) > 1 else (norms[0], '')
        levels = g['Level'].tolist()
        agg_level = priority_level(levels)
        # collect ddinter id pairs
        dd_pairs = list(g[['DDInterID_A', 'DDInterID_B']].itertuples(index=False, name=None))
        # collect sample originals for display
        sample_a = list(norm_map.get(drug_a_norm, []))[:3]
        sample_b = list(norm_map.get(drug_b_norm, []))[:3]
        rows.append({
            'drug_a': drug_a_norm,
            'drug_b': drug_b_norm,
            'interaction_level': agg_level,
            'ddinter_pairs': json.dumps(dd_pairs, ensure_ascii=False),
            'sample_origins_a': json.dumps(sample_a, ensure_ascii=False),
            'sample_origins_b': json.dumps(sample_b, ensure_ascii=False),
            'support_count': len(dd_pairs),
        })

    out_df = pd.DataFrame(rows)
    out_path = base / 'data' / 'processed' / 'master_interactions.csv'
    out_df.to_csv(out_path, index=False)

    # write unique drug names and variants
    uniq = []
    for k, variants in norm_map.items():
        uniq.append({'normalized': k, 'variants': json.dumps(list(variants), ensure_ascii=False)})
    uniq_df = pd.DataFrame(uniq)
    uniq_df.to_csv(base / 'data' / 'processed' / 'unique_drug_names.csv', index=False)

    # summary
    print('\nSummary:')
    print('Original rows:', original_rows)
    print('After exact dedupe:', after)
    print('Exact duplicate rows removed:', exact_dups)
    print('Canonical unique pairs:', len(out_df))
    print('Master CSV written to', out_path)
    print('\nSample rows:')
    print(out_df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
