import numpy as np
import pandas as pd

class AdversarialEngine:
    """
    Injects anomalies into existing historical OHLCV data to test if the AI
    has proper risk-off reflexes or if it breaks down.
    """
    
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def inject_flash_crash(self, df: pd.DataFrame, drop_pct=0.15, recovery_bars=5) -> pd.DataFrame:
        """
        Simulates a sudden flash crash in the middle of the dataset.
        """
        df_mod = df.copy()
        crash_idx = len(df_mod) // 2
        
        crash_price = df_mod.loc[crash_idx, 'close'] * (1 - drop_pct)
        df_mod.loc[crash_idx, 'low'] = crash_price
        df_mod.loc[crash_idx, 'close'] = crash_price
        df_mod.loc[crash_idx, 'volume'] *= 10  # Massive volume spike
        
        # Simulate quick but volatile recovery
        for i in range(1, recovery_bars + 1):
            if crash_idx + i < len(df_mod):
                df_mod.loc[crash_idx + i, 'open'] = df_mod.loc[crash_idx + i - 1, 'close']
                df_mod.loc[crash_idx + i, 'low'] = min(df_mod.loc[crash_idx + i, 'open'], crash_price * 1.05)
                # Let original close/high mostly persist but with shifted baseline if needed
                
        return df_mod

    def remove_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Zeroes out volume to see if the vision model panics or gracefully degrades.
        """
        df_mod = df.copy()
        df_mod['volume'] = 0.0
        return df_mod

    def inject_noise(self, df: pd.DataFrame, noise_level=0.02) -> pd.DataFrame:
        """
        Adds random gaussian noise to the price action.
        """
        df_mod = df.copy()
        noise = self.rng.normal(1.0, noise_level, len(df_mod))
        df_mod['open'] *= noise
        df_mod['high'] *= noise
        df_mod['low'] *= noise
        df_mod['close'] *= noise
        
        # Re-fix high/low violations caused by noise
        df_mod['high'] = df_mod[['open', 'high', 'low', 'close']].max(axis=1)
        df_mod['low'] = df_mod[['open', 'high', 'low', 'close']].min(axis=1)
        return df_mod

if __name__ == "__main__":
    # Dummy data test
    dummy_data = pd.DataFrame({
        'open': np.ones(100) * 100,
        'high': np.ones(100) * 105,
        'low': np.ones(100) * 95,
        'close': np.ones(100) * 102,
        'volume': np.ones(100) * 1000
    })
    
    eng = AdversarialEngine()
    crashed = eng.inject_flash_crash(dummy_data, drop_pct=0.5)
    print("Crash bar:")
    print(crashed.iloc[50])
