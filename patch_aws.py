import os

real_data_path = os.path.expanduser('~/zhisa/src/zhisa/scripts/_real_data.py')
with open(real_data_path, 'r') as f:
    content = f.read()

content = content.replace(
    'df = pd.read_csv(path)', 
    'df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)'
)

content = content.replace(
    'df = normalize_ohlcv_frame(df)',
    'df = normalize_ohlcv_frame(df, keep_extra=True)'
)

with open(real_data_path, 'w') as f:
    f.write(content)

train_s1_path = os.path.expanduser('~/zhisa/src/zhisa/scripts/train_s1.py')
with open(train_s1_path, 'r') as f:
    content = f.read()

content = content.replace(
    'df = load_market_dataframe(sym_args, seed=seed, default_bars=args.bars)',
    '''if hasattr(sym_args, "csv") and sym_args.csv and "{symbol}" in sym_args.csv:
                sym_args.csv = sym_args.csv.replace("{symbol}", sym_args.symbol.replace("/", "_"))
            df = load_market_dataframe(sym_args, seed=seed, default_bars=args.bars)'''
)

with open(train_s1_path, 'w') as f:
    f.write(content)

print('Patch applied successfully!')
