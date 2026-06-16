import torch
import numpy as np
from zhisa.risk.cvar import cvar_torch

def test_cvar_gradients():
    print("--- DIAGNOSTIC TEST FOR CVAR GRADIENTS ---")
    
    # 1. Simulate policy outputs (logits have requires_grad=True to simulate network)
    policy_logits = torch.randn(10, 5, requires_grad=True)
    
    # 2. Simulate standard PPO loss
    # (Just a dummy loss connected to logits to show normal gradients work)
    ppo_loss = policy_logits.sum() 
    
    # 3. Simulate environment rewards (from numpy, as in cvar_ppo.py)
    ep_returns_np = np.array([-10.0, -20.0, 5.0, 15.0], dtype=np.float32)
    
    # 4. Simulate the CVaR calculation exactly as in cvar_ppo.py
    ep_returns_t = torch.from_numpy(ep_returns_np) # Notice: no requires_grad
    
    # The actual cvar_torch calculation
    cvar_value = cvar_torch(ep_returns_t, alpha=0.5)
    cvar_penalty = torch.nn.functional.relu(-cvar_value - 0.1)
    
    lambda_cvar = 1.0
    cvar_term = lambda_cvar * cvar_penalty
    
    print("Does ep_returns_t require grad?", ep_returns_t.requires_grad)
    print("Does cvar_penalty require grad?", cvar_penalty.requires_grad)
    print("Does cvar_term require grad?", cvar_term.requires_grad)
    
    total_loss = ppo_loss + cvar_term
    total_loss.backward()
    
    print("Policy Logits Gradient is None?", policy_logits.grad is None)
    
    # Test if cvar_term alone can flow gradients to policy
    policy_logits.grad = None
    try:
        cvar_term.backward()
        print("cvar_term.backward() succeeded. Did logits get grad?", policy_logits.grad is not None)
    except RuntimeError as e:
        print("cvar_term.backward() failed with RuntimeError:", e)

if __name__ == "__main__":
    test_cvar_gradients()
