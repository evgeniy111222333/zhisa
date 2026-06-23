import pandas as pd
import json
from pathlib import Path

root = Path('data/prepared/s1_15m_v1')
m = json.load(open(root/'manifest.json'))
print('=== MANIFEST ===')
print(f'version:        {m["version"]}')
print(f'rows_total:     {m["rows_total"]:,}')
print(f'window:         {m["start"]} .. {m["end"]}')
print(f'checksum:       {m["output_checksum"]}')
print(f'feature_cols:   {len(m["feature_columns"])} columns')
print(f'                {m["feature_columns"][:8]}...')
print()
print('=== PER-SYMBOL ===')
for sym, rows in m['rows_per_symbol'].items():
    print(f'  {sym:<12} {rows:>8,} bars')
print()
print('=== SYMBOL FILES ===')
for p in sorted((root/'symbols').glob('*.parquet')):
    df = pd.read_parquet(p)
    ctx_cols = [c for c in df.columns if c.startswith('ctx_')]
    print(f'  {p.name:<24} rows={len(df):>8,}  total_cols={len(df.columns):>3}  ctx_cols={len(ctx_cols)}')
print()
print('=== SPLITS ===')
for split in ['train','val','test']:
    df = pd.read_parquet(root/'splits'/f'{split}.parquet')
    print(f'  {split:<6} rows={len(df):>8,}  symbols={sorted(df["symbol"].unique())}')
