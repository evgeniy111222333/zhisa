import pandas as pd
from pathlib import Path

for sym in ['BTC_USDT', 'ETH_USDT', 'SOL_USDT']:
    print(f'\n=== {sym} ===')
    df = pd.read_parquet(f'D:/zhisa/data/tsdb/{sym}/15m/data.parquet')
    expected_idx = pd.date_range(df.index.min(), df.index.max(), freq='15min')
    missing = expected_idx.difference(df.index)
    print(f'Total missing 15m slots: {len(missing):,}')
    if len(missing) == 0:
        continue
    diffs = missing.to_series().diff()
    span_starts = missing[diffs != pd.Timedelta('15min')]
    print(f'Contiguous gap spans: {len(span_starts)}')
    for i, start in enumerate(span_starts[:8]):
        j = list(missing).index(start)
        k = j
        while k+1 < len(missing) and missing[k+1] - missing[k] == pd.Timedelta('15min'):
            k += 1
        end = missing[k]
        n_bars = len(pd.date_range(start, end, freq='15min'))
        print(f'  gap {i+1}: {start} -> {end}  ({n_bars} bars = {(end-start).total_seconds()/3600:.1f}h)')
