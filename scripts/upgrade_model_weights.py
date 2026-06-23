import torch
import sys

def main():
    if len(sys.argv) < 4:
        print("Usage: upgrade_model_weights.py <old_ckpt> <new_ckpt> <new_feature_count>")
        sys.exit(1)
        
    old_path = sys.argv[1]
    new_path = sys.argv[2]
    target_features = int(sys.argv[3])
    
    print(f"Loading old model from {old_path}...")
    ckpt = torch.load(old_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]
    
    old_weight = state_dict["numeric.patch_proj.weight"]
    d_model, old_in_dim = old_weight.shape
    
    # We now have target_features features. patch_size is 4.
    new_in_dim = target_features * 4
    
    print(f"Old patch_proj.weight shape: {old_weight.shape}")
    print(f"Expanding patch_proj.weight to: [{d_model}, {new_in_dim}]")
    
    new_weight = torch.zeros(d_model, new_in_dim, dtype=old_weight.dtype, device=old_weight.device)
    # Copy the old weights into the first 128 dimensions (the first 32 features)
    new_weight[:, :old_in_dim] = old_weight
    
    # Replace in state_dict
    state_dict["numeric.patch_proj.weight"] = new_weight
    
    # Update config in checkpoint so visualizers know it has 71 features
    if "config" not in ckpt:
        ckpt["config"] = {}
    ckpt["config"]["in_numeric_features"] = 71
    
    print(f"Saving upgraded model to {new_path}...")
    torch.save(ckpt, new_path)
    print("Done! The model can now be loaded into an environment with 71 features.")

if __name__ == "__main__":
    main()
