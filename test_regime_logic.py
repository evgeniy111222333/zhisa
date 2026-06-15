import sys
import numpy as np
sys.path.insert(0, r"D:\zhisa\src")
from zhisa.scripts._real_data import load_market_dataframe
from zhisa.data.labeling import hmm_regime_labels

class FakeArgs:
    data_source = "tsdb"
    tsdb_root = "data/tsdb"
    symbol = "BTC/USDT"
    timeframe = "5m"
    with_futures_context = True
    futures_context_root = "data/futures_context/binance_usdm"

def main():
    print("="*60)
    print("REGIME LOGIC TEST: Numpy vs Sklearn")
    print("="*60)
    
    print("Loading 20,000 bars from TSDB...")
    args = FakeArgs()
    df = load_market_dataframe(args, default_bars=20000)
    if df is None or len(df) == 0:
        print("Failed to load TSDB data.")
        return
    
    # Calculate returns for analysis
    returns = np.log(df["close"]).diff().fillna(0.0).to_numpy()
    
    # 2. Test the BUG (numpy fallback)
    print("\n[1] Testing BUG (prefer_sklearn=False)...")
    labels_bug = hmm_regime_labels(df, n_states=4, lookback=256, prefer_sklearn=False)
    
    print("    Regime Statistics (Numpy Fallback):")
    for i in range(4):
        regime_mask = (labels_bug == i)
        count = regime_mask.sum()
        if count > 0:
            regime_rets = returns[regime_mask]
            mean_ret = regime_rets.mean() * 100
            std_ret = regime_rets.std() * 100
            print(f"    Regime {i}: {count:5d} bars | Mean: {mean_ret:7.4f}% | Volatility (Std): {std_ret:7.4f}%")
        else:
            print(f"    Regime {i}: Empty")
            
    # 3. Test the FIX (sklearn)
    print("\n[2] Testing FIX (prefer_sklearn=True)...")
    labels_fix = hmm_regime_labels(df, n_states=4, lookback=256, prefer_sklearn=True)
    
    print("    Regime Statistics (Sklearn GMM):")
    for i in range(4):
        regime_mask = (labels_fix == i)
        count = regime_mask.sum()
        if count > 0:
            regime_rets = returns[regime_mask]
            mean_ret = regime_rets.mean() * 100
            std_ret = regime_rets.std() * 100
            print(f"    Regime {i}: {count:5d} bars | Mean: {mean_ret:7.4f}% | Volatility (Std): {std_ret:7.4f}%")
        else:
            print(f"    Regime {i}: Empty")
            
    print("\nConclusion:")
    print("If Sklearn is working, Volatility should strictly INCREASE from Regime 0 to 3.")
    print("If Numpy fallback is bugged, Volatilities will be randomly mixed and close to each other.")
    print("="*60)

if __name__ == "__main__":
    main()
