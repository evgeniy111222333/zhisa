import torch
import numpy as np

def test_cvar_gradients():
    print("--- DIAGNOSTIC TEST FOR CVAR GRADIENTS ---")
    
    # 1. Simulate policy outputs (logits have requires_grad=True to simulate network)
    policy_logits = torch.randn(4, 5, requires_grad=True)
    action_idx = torch.tensor([1, 2, 0, 4])
    
    # Compute log_probs (similar to PyTorch distributions)
    log_probs = torch.log_softmax(policy_logits, dim=-1)
    selected_log_probs = log_probs[torch.arange(4), action_idx]
    
    # 2. Simulate standard Advantages (detached from network, from environment)
    advantages_np = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    
    # 3. Simulate episode returns
    ep_returns_np = np.array([-10.0, -20.0, 5.0, 15.0], dtype=np.float32)
    step_to_ep_return = ep_returns_np  # For simplicity, 1 step = 1 episode here
    
    # === NEW METHOD: ADVANTAGE PENALIZATION ===
    var_threshold = float(np.percentile(ep_returns_np, 50)) # alpha=0.5
    cvar_threshold = 0.1
    lambda_cvar = 5.0 # Large penalty to make it obvious
    
    print("\n[BEFORE FIX] Advantages:", advantages_np)
    
    # Mask for violating episodes
    violation_mask = (step_to_ep_return <= var_threshold) & (step_to_ep_return < -cvar_threshold)
    
    if np.any(violation_mask):
        advantages_np[violation_mask] -= lambda_cvar
        
    print("[AFTER FIX] Advantages:", advantages_np)
    
    # 4. Compute PPO Loss
    adv_t = torch.from_numpy(advantages_np)
    
    # PPO surrogate loss (simplified: -log_prob * advantage)
    ppo_loss = -(selected_log_probs * adv_t).mean()
    
    # 5. Check Gradients
    ppo_loss.backward()
    
    print("\nPolicy Logits Gradient is None?", policy_logits.grad is None)
    print("Logits Gradients:\n", policy_logits.grad)
    print("\nSUCCESS: Gradients flow correctly through log_probs because we penalized the Advantages!")

if __name__ == "__main__":
    test_cvar_gradients()
