import numpy as np
import torch
import torch.nn.functional as F

def cvar_numpy(returns: np.ndarray, alpha: float) -> float:
    """Mock cvar for testing."""
    if len(returns) == 0:
        return 0.0
    k = max(1, int(np.ceil(alpha * len(returns))))
    return float(np.mean(np.sort(returns)[:k]))

def test_cvar_logic():
    print("--- CVaR Math Diagnostics ---")
    cvar_threshold = 18.0
    cvar_alpha = 0.1
    cvar_lambda_lr = 0.05
    cvar_lambda_max = 100.0
    
    current_lambda = 0.0662
    
    # Scenario 1: Model is performing well (like currently)
    # Returns between -10% and -15%
    returns_good = np.array([-10.0, -12.0, -13.0, -14.0, -15.0, -5.0, -2.0, 5.0, 10.0, 15.0])
    cvar_val_good = cvar_numpy(returns_good, cvar_alpha)
    violation_good = max(0.0, -cvar_val_good - cvar_threshold)
    new_lambda_good = np.clip(current_lambda + cvar_lambda_lr * violation_good, 0.0, cvar_lambda_max)
    
    print("\n[Scenario 1: Model is Safe (Current State)]")
    print(f"Tail returns (worst 10%): {cvar_val_good:.2f}%")
    print(f"Threshold limit: {cvar_threshold:.2f}%")
    print(f"Violation: max(0, {-cvar_val_good:.2f} - {cvar_threshold:.2f}) = {violation_good:.4f}")
    print(f"New Lambda: {current_lambda:.4f} + ({cvar_lambda_lr} * {violation_good:.4f}) = {new_lambda_good:.4f}")
    
    # Scenario 2: Model performs poorly and breaches the 18% limit
    # Returns hit -25%
    returns_bad = np.array([-25.0, -22.0, -20.0, -18.0, -15.0, -5.0, -2.0, 5.0, 10.0, 15.0])
    cvar_val_bad = cvar_numpy(returns_bad, cvar_alpha)
    violation_bad = max(0.0, -cvar_val_bad - cvar_threshold)
    new_lambda_bad = np.clip(current_lambda + cvar_lambda_lr * violation_bad, 0.0, cvar_lambda_max)
    
    print("\n[Scenario 2: Model is Unsafe (Hits -25% loss)]")
    print(f"Tail returns (worst 10%): {cvar_val_bad:.2f}%")
    print(f"Threshold limit: {cvar_threshold:.2f}%")
    print(f"Violation: max(0, {-cvar_val_bad:.2f} - {cvar_threshold:.2f}) = {violation_bad:.4f}")
    print(f"New Lambda: {current_lambda:.4f} + ({cvar_lambda_lr} * {violation_bad:.4f}) = {new_lambda_bad:.4f} (Lambda INCREASES!)")
    
    # Check Advantage Penalization
    advantages = np.array([1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5])
    step_to_ep = np.array([15.0, 10.0, 5.0, -2.0, -5.0, -15.0, -18.0, -20.0, -22.0, -25.0])
    
    print("\n[Advantage Penalization Check]")
    print(f"Original Advantages of worst steps: {advantages[-3:]}")
    violation_mask = (step_to_ep <= cvar_val_bad) & (step_to_ep < -cvar_threshold)
    if np.any(violation_mask) and new_lambda_bad > 0:
        advantages[violation_mask] -= float(new_lambda_bad)
    print(f"Penalized Advantages of worst steps: {advantages[-3:]}")
    print("-> The network will be strongly discouraged from taking those actions again!")
    
if __name__ == "__main__":
    test_cvar_logic()
