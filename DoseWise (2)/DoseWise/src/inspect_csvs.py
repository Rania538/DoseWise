import pandas as pd
from pathlib import Path

base = Path('d:/Documents/tessst')
files = [base / 'ddinter_downloads_code_A.csv', base / 'ddinter_downloads_code_B.csv', base / 'ddinter_downloads_code_R.csv']

def inspect(df):
    info = {}
    info['rows'] = len(df)
    info['columns'] = list(df.columns)
    info['missing_per_col'] = df.isna().sum().to_dict()
    info['duplicate_rows'] = df.duplicated().sum()
    info['sample'] = df.head(5).to_dict(orient='records')
    return info

all_info = {}
for f in files:
    df = pd.read_csv(f)
    all_info[f.name] = inspect(df)

# Combined basic stats
combined = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
combined_info = inspect(combined)

# Unique drug names
unique_drugs = pd.unique(combined[['Drug_A','Drug_B']].values.ravel('K'))

# Check pair duplicates considering direction
pairs = combined[['Drug_A','Drug_B','Level']].copy()
# normalized pair key (unordered)
pairs['pair_key'] = pairs.apply(lambda r: '||'.join(sorted([str(r['Drug_A']), str(r['Drug_B'])])), axis=1)
pair_counts = pairs['pair_key'].value_counts()

def to_py(x):
    try:
        return int(x)
    except Exception:
        try:
            return float(x)
        except Exception:
            return x

pair_key_counts_sample = {k: int(v) for k,v in pair_counts.head(20).items()}

report = {
    'files': all_info,
    'combined': combined_info,
    'unique_drug_count': int(len(unique_drugs)),
    'unique_drug_sample': [str(x) for x in list(unique_drugs)[:50]],
    'pair_directional_duplicates': int(len(combined) - len(pairs.drop_duplicates(subset=['pair_key','Level']))),
    'pair_key_counts_sample': pair_key_counts_sample,
}

import json
import numpy as np

def normalize(obj):
    if isinstance(obj, dict):
        return {str(k): normalize(v) for k,v in obj.items()}
    if isinstance(obj, list):
        return [normalize(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(normalize(v) for v in obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return [normalize(v) for v in obj.tolist()]
    if hasattr(obj, 'tolist') and not isinstance(obj, str):
        try:
            return normalize(obj.tolist())
        except Exception:
            pass
    return obj

print(json.dumps(normalize(report), indent=2, ensure_ascii=False))
