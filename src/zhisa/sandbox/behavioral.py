import numpy as np
import pandas as pd
from typing import Literal

class PatternGenerator:
    """
    Generates synthetic OHLCV data for pure behavioral testing of the AI.
    These are "laboratory" patterns without market noise, to see if the model 
    recognizes textbook setups.
    """
    
    def __init__(self, base_price=50000.0, volatility=0.001):
        self.base_price = base_price
        self.vol = volatility
        self.rng = np.random.default_rng(42)

    def _generate_bar(self, open_price, close_price, high_markup=1.5, low_markup=1.5, volume_base=1000):
        high = max(open_price, close_price) + (self.base_price * self.vol * high_markup)
        low = min(open_price, close_price) - (self.base_price * self.vol * low_markup)
        volume = volume_base + self.rng.normal(0, volume_base * 0.1)
        return {"open": open_price, "high": high, "low": low, "close": close_price, "volume": max(1, volume)}

    def generate_trend(self, n_bars=100, direction: Literal['up', 'down'] = 'up') -> pd.DataFrame:
        """Generates a clean trend."""
        bars = []
        price = self.base_price
        trend_step = self.base_price * self.vol * (1 if direction == 'up' else -1)
        
        for _ in range(n_bars):
            open_p = price
            close_p = price + trend_step + self.rng.normal(0, self.base_price * self.vol * 0.5)
            bars.append(self._generate_bar(open_p, close_p))
            price = close_p
            
        return pd.DataFrame(bars)

    def generate_fakeout(self, n_bars=100, direction: Literal['up', 'down'] = 'up') -> pd.DataFrame:
        """Generates a range, a breakout, and an immediate sharp reversal (fakeout)."""
        bars = []
        price = self.base_price
        
        # Phase 1: Range (70% of bars)
        range_bars = int(n_bars * 0.7)
        for _ in range(range_bars):
            open_p = price + self.rng.normal(0, self.base_price * self.vol)
            close_p = price + self.rng.normal(0, self.base_price * self.vol)
            bars.append(self._generate_bar(open_p, close_p, volume_base=500))
            
        # Phase 2: Breakout (10% of bars)
        bo_bars = int(n_bars * 0.1)
        bo_step = self.base_price * self.vol * 3 * (1 if direction == 'up' else -1)
        current_p = bars[-1]['close']
        for _ in range(bo_bars):
            open_p = current_p
            close_p = current_p + bo_step
            bars.append(self._generate_bar(open_p, close_p, volume_base=2000))  # High volume breakout
            current_p = close_p
            
        # Phase 3: Fakeout Reversal (20% of bars)
        rev_step = -bo_step * 1.5
        for _ in range(n_bars - range_bars - bo_bars):
            open_p = current_p
            close_p = current_p + rev_step
            bars.append(self._generate_bar(open_p, close_p, volume_base=500)) # Low volume fakeout or panic
            current_p = close_p
            
        return pd.DataFrame(bars)

if __name__ == "__main__":
    gen = PatternGenerator()
    df_fakeout = gen.generate_fakeout(direction='up')
    print("Fakeout Data Head:")
    print(df_fakeout.head())
    print("\nFakeout Data Tail (Reversal):")
    print(df_fakeout.tail())
