import json
from pathlib import Path
from api_lookup import lookup_drug


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
