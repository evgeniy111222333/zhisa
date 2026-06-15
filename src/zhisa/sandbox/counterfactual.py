import pandas as pd
from typing import Callable

class CounterfactualEngine:
    """
    Evaluates 'What-If' scenarios by perturbing a single aspect of a historical window
    and running it through the model to observe changes in action probabilities.
    """
    
    def __init__(self, model_inference_fn: Callable):
        """
        model_inference_fn: A function that takes a DataFrame of OHLCV and returns 
        a dictionary of action probabilities, e.g., {'long': 0.8, 'short': 0.1, 'wait': 0.1}
        """
        self.model_inference_fn = model_inference_fn

    def evaluate_sensitivity(self, base_df: pd.DataFrame, perturbation_fn: Callable) -> dict:
        """
        Compares model output on the base dataframe vs the perturbed dataframe.
        """
        # Baseline inference
        base_probs = self.model_inference_fn(base_df)
        
        # Perturb
        perturbed_df = perturbation_fn(base_df)
        
        # Counterfactual inference
        cf_probs = self.model_inference_fn(perturbed_df)
        
        # Calculate delta
        delta = {k: cf_probs.get(k, 0) - base_probs.get(k, 0) for k in set(base_probs) | set(cf_probs)}
        
        return {
            'baseline': base_probs,
            'counterfactual': cf_probs,
            'delta': delta
        }

    # Common perturbation functions
    @staticmethod
    def halve_breakout_volume(df: pd.DataFrame) -> pd.DataFrame:
        """Finds the bar with the max volume and cuts it in half."""
        df_mod = df.copy()
        max_vol_idx = df_mod['volume'].idxmax()
        df_mod.loc[max_vol_idx, 'volume'] *= 0.5
        return df_mod
        
    @staticmethod
    def compress_recent_volatility(df: pd.DataFrame, window=10) -> pd.DataFrame:
        """Squeezes the high/low range of the last N bars."""
        df_mod = df.copy()
        for i in range(len(df_mod) - window, len(df_mod)):
            mid = (df_mod.loc[i, 'high'] + df_mod.loc[i, 'low']) / 2
            df_mod.loc[i, 'high'] = mid + (df_mod.loc[i, 'high'] - mid) * 0.5
            df_mod.loc[i, 'low'] = mid - (mid - df_mod.loc[i, 'low']) * 0.5
        return df_mod

if __name__ == "__main__":
    # Dummy inference function for testing
    def dummy_model(df):
        # If volume is huge on last bar, long
        if df['volume'].iloc[-1] > 1500:
            return {'long': 0.9, 'short': 0.05, 'wait': 0.05}
        return {'long': 0.3, 'short': 0.3, 'wait': 0.4}
        
    engine = CounterfactualEngine(dummy_model)
    
    test_df = pd.DataFrame({
        'open': [100]*5, 'high': [105]*5, 'low': [95]*5, 'close': [102]*5,
        'volume': [1000, 1000, 1000, 1000, 2000] # Huge volume at end
    })
    
    res = engine.evaluate_sensitivity(test_df, CounterfactualEngine.halve_breakout_volume)
    print("Sensitivity Analysis:", res)
