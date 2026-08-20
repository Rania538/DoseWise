import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from drug_mapping import build_mapping_from_list


def main(batch_size=500):
    base = Path('d:/Documents/tessst')
    uniq_path = base / 'data' / 'processed' / 'unique_drug_names.csv'
    if not uniq_path.exists():
        print('unique_drug_names.csv not found at', uniq_path)
        return
    df = pd.read_csv(uniq_path, dtype=str)
    names = df['normalized'].dropna().tolist()
    # limit for this run to avoid excessive API calls
    names_batch = names[:batch_size]
    out_csv = base / 'data' / 'processed' / 'drug_mapping.csv'
    print(f'Building drug mapping for {len(names_batch)} terms. Output: {out_csv}')
    build_mapping_from_list(names_batch, out_csv_path=str(out_csv))
    print('Done.')


if __name__ == '__main__':
    import sys
    try:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    except Exception:
        n = 500
    main(batch_size=n)
