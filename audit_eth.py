import pandas as pd
import numpy as np

try:
    df = pd.read_parquet(r'D:\zhisa\data\tsdb\binance\ETH_USDT\15m\futures_context.parquet')
    
    print('--- ETH/USDT 15m Context Quality Audit ---')
    print(f'Total Rows: {len(df)}')
    print(f'Start Time: {df.index[0]}')
    print(f'End Time:   {df.index[-1]}')
    
    # 1. Monotonicity and Gaps
    is_monotonic = df.index.is_monotonic_increasing
    print(f'\nIs time strictly monotonic? {is_monotonic}')
    
    diffs = df.index.to_series().diff().dropna()
    expected_diff = pd.Timedelta(minutes=15)
    exact_15m_pct = (diffs == expected_diff).mean() * 100
    print(f'Percentage of exactly 15m steps: {exact_15m_pct:.4f}%')
    
    if exact_15m_pct < 100:
        bad_diffs = diffs[diffs != expected_diff]
        print(f'Found {len(bad_diffs)} gaps/jumps not equal to 15m (likely exchange downtime or missing data).')
        print('Top irregular gaps:')
        print(bad_diffs.value_counts().head(5))
        
    # 2. Check for Duplicates
    dupes = df.index.duplicated().sum()
    print(f'\nDuplicate timestamps: {dupes}')
    
    # 3. Null values
    print('\n--- Null Values (Missing Data) ---')
    nulls = df.isna().sum()
    nulls_pct = (nulls / len(df)) * 100
    missing = pd.DataFrame({'Missing': nulls, '%': nulls_pct})
    print(missing[missing['Missing'] > 0])
    if missing['Missing'].sum() == 0:
        print('Perfect! 0 missing values across all columns.')
    
    # 4. Data sanity check (funding rate)
    print('\n--- Funding Rate Sanity Check ---')
    max_f = df['funding_rate'].max()
    min_f = df['funding_rate'].min()
    print(f'Max funding rate: {max_f:.6f}')
    print(f'Min funding rate: {min_f:.6f}')
    
except Exception as e:
    print('Failed to analyze:', e)
