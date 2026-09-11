import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from drug_mapping import build_ddinter_mapping


def main():
    base = Path(__file__).resolve().parents[1]
    out_csv = base / 'data' / 'processed' / 'drug_mapping.csv'
    rows = build_ddinter_mapping(root_path=base, out_csv_path=out_csv)
    print(f'Built {len(rows)} mapping rows for {out_csv}')


if __name__ == '__main__':
    import sys
    main()
