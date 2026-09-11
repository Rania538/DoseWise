import json
import re
import unicodedata
from pathlib import Path
from api_lookup import lookup_drug


DATASET_FILES = (
    'ddinter_downloads_code_A.csv',
    'ddinter_downloads_code_B.csv',
    'ddinter_downloads_code_R.csv',
)

# Only identity-equivalent names are included here. Product names that can refer
# to multiple ingredients are intentionally omitted.
VERIFIED_ALIASES = {
    'aspirin': 'acetylsalicylic acid',
    'asa': 'acetylsalicylic acid',
    'panadol': 'acetaminophen',
    'pandol': 'acetaminophen',
    'بنادول': 'acetaminophen',
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
    'amoxil': 'amoxicillin',
    'amoxicilin': 'amoxicillin',
    'glucophage': 'metformin',
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
}


def normalize_mapping_text(value):
    value = unicodedata.normalize('NFKC', str(value or '')).strip().lower()
    value = re.sub(r'[^\w\s\u0600-\u06ff]', ' ', value)
    return re.sub(r'\s+', ' ', value).strip()


def build_ddinter_mapping(root_path=None, out_csv_path=None):
    """Build one canonical row for every DDInter drug and verified alias rows."""
    import pandas as pd

    root = Path(root_path or Path(__file__).resolve().parents[1])
    files = [root / filename for filename in DATASET_FILES]
    frames = [pd.read_csv(path, dtype=str) for path in files]
    combined = pd.concat(frames, ignore_index=True)
    records = pd.concat([
        combined[['Drug_A', 'DDInterID_A']].rename(columns={'Drug_A': 'name', 'DDInterID_A': 'ddinter_id'}),
        combined[['Drug_B', 'DDInterID_B']].rename(columns={'Drug_B': 'name', 'DDInterID_B': 'ddinter_id'}),
    ], ignore_index=True).dropna(subset=['name', 'ddinter_id'])

    canonical = {}
    for row in records.itertuples(index=False):
        name = normalize_mapping_text(row.name)
        ddinter_id = str(row.ddinter_id).strip()
        if name and ddinter_id:
            canonical.setdefault(name, ddinter_id)

    rows = []
    for name in sorted(canonical):
        rows.append({
            'user_term': name,
            'brand_name': '',
            'generic_name': name,
            'active_ingredient': name,
            'normalized_name': name,
            'aliases': '',
            'language': '',
            'source': 'ddinter_dataset',
            'ddinter_id': canonical[name],
        })

    for alias, name in sorted(VERIFIED_ALIASES.items()):
        if name in canonical and alias not in canonical:
            rows.append({
                'user_term': alias,
                'brand_name': alias,
                'generic_name': name,
                'active_ingredient': name,
                'normalized_name': alias,
                'aliases': alias,
                'language': 'ar' if re.search(r'[\u0600-\u06ff]', alias) else '',
                'source': 'verified_alias',
                'ddinter_id': canonical[name],
            })

    if out_csv_path:
        output = Path(out_csv_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output, index=False)
    return rows


def build_mapping_from_list(terms, out_csv_path=None):
    """Given a list of user terms, query API (with cache) and produce mapping rows.
    This is a helper to build `drug_mapping.csv` progressively.
    """
    rows = []
    for t in terms:
        info = lookup_drug(t)
        rows.append({
            'user_term': t,
            'brand_name': ','.join(info.get('brand_names', [])),
            'generic_name': info.get('generic_name') or '',
            'active_ingredient': ','.join(info.get('ingredients', [])),
            'normalized_name': '',
            'aliases': '',
            'language': '',
            'source': ';'.join(info.get('sources', []))
        })
    if out_csv_path:
        import pandas as pd
        df = pd.DataFrame(rows)
        Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv_path, index=False)
    return rows


if __name__ == '__main__':
    print('drug_mapping builder available. Use build_mapping_from_list() to create mapping CSV.')
