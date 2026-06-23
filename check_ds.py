import pandas as pd
import glob
files = glob.glob("/mnt/zhisa/project/data/prepared/s1_15m_12m_v2/symbols/*.parquet")
print("Found files:", len(files))
for f in files:
    df = pd.read_parquet(f)
    print(f"File: {f.split('/')[-1]}")
    print(f"Rows: {len(df)}")
    print(f"Cols: {len(df.columns)}")
    print(f"Start: {df.index.min()}")
    print(f"End: {df.index.max()}")
    print("-" * 20)
