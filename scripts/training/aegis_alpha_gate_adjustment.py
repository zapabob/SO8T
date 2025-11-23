import torch
import os
import math
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def adjust_and_fuse():
    parser = argparse.ArgumentParser(description="AGIASI Alpha Gate Adjustment & Fusion")
    parser.add_argument("--base-model", type=str, default="Borea/Phi-3.5-instinct-jp", help="Original Base Model ID")
    parser.add_argument("--soul-path", type=str, required=True, help="Path to soul_params.pt")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to save the adjusted model")
    parser.add_argument("--scale-factor", type=float, default=0.6, help="Manual scaling factor for Alpha influence")
    
    args = parser.parse_args()

    print(f"⚖️  Alpha Calibration Mode: Scaling Factor = {args.scale_factor}")
    
    # 1. Load Base Model (FP16)
    print(f"   Loading Base Vessel: {args.base_model}")
    # Check if local path exists for base model, otherwise use HuggingFace ID
    model_id = args.base_model
    if os.path.exists(os.path.join("models", "Borea-Phi-3.5-mini-Instruct-Jp")):
         model_id = os.path.join("models", "Borea-Phi-3.5-mini-Instruct-Jp")
         print(f"   Found local base model at: {model_id}")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # 2. Load Soul
    print(f"   Loading Soul Geometry from: {args.soul_path}")
    if not os.path.exists(args.soul_path):
        raise FileNotFoundError(f"Soul file not found: {args.soul_path}")
        
    soul_data = torch.load(args.soul_path, map_location="cuda")
    
    # Alpha Value & Rotation
    alpha_raw = soul_data["alpha"].item()
    rotation_state = soul_data["rotation"]
    
    # 物理的ゲート開度 (Sigmoid)
    original_gate = sigmoid(alpha_raw)
    
    # ★ ここが調整の肝: 元のSigmoid値に、手動スケールを掛ける
    # Effective Alpha = Sigmoid(α) * Scale
    effective_gate = original_gate * args.scale_factor
    
    print(f"   Original Alpha: {alpha_raw:.4f} (Sigmoid: {original_gate:.4f})")
    print(f"   🎯 Target Effective Gate: {effective_gate:.4f} (Adjusted by x{args.scale_factor})")

    # Reconstruct Rotation Matrix R
    hidden_dim = base_model.config.hidden_size
    dummy_layer = torch.nn.utils.parametrizations.orthogonal(
        torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
    ).to("cuda")
    
    print(f"   Rotation State Type: {type(rotation_state)}")
    if isinstance(rotation_state, dict):
        print(f"   Rotation Keys: {list(rotation_state.keys())}")
        # Try to load known keys directly
        if "parametrizations.weight.0.base" in rotation_state:
            print("   Loading 'parametrizations.weight.0.base' directly...")
            dummy_layer.parametrizations.weight[0].base.data = rotation_state["parametrizations.weight.0.base"].to("cuda")
        elif "weight" in rotation_state:
             print("   Loading 'weight' directly (assuming raw orthogonal matrix)...")
             # If it's just the weight, maybe we don't need the parametrization wrapper?
             # But if we want R, we can just take it.
             dummy_layer = torch.nn.Linear(hidden_dim, hidden_dim, bias=False).to("cuda")
             dummy_layer.weight.data = rotation_state["weight"].to("cuda")
        else:
            print("   ⚠️ Unknown keys. Trying strict load_state_dict...")
            dummy_layer.load_state_dict(rotation_state)
    elif isinstance(rotation_state, torch.Tensor):
        print("   Rotation is a Tensor. Loading directly...")
        dummy_layer = torch.nn.Linear(hidden_dim, hidden_dim, bias=False).to("cuda")
        dummy_layer.weight.data = rotation_state.to("cuda")
    
    R = dummy_layer.weight.data.float()

    # 3. Fusion Process
    print("   ⚗️  Re-Fusing with adjusted parameters...")
    
    with torch.no_grad():
        W_head = base_model.lm_head.weight.data.float()
        
        # Perturbation = W_head @ R
        perturbation = torch.matmul(W_head, R)
        
        # Weighted by NEW Effective Gate
        W_fused = W_head + (effective_gate * perturbation)
        
        # Update Weights
        base_model.lm_head.weight.data = W_fused.to(base_model.lm_head.weight.dtype)

    # 4. Save
    print(f"   💾 Saving Adjusted AEGIS to {args.output_dir}...")
    base_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("   ✅ Calibration Complete.")

if __name__ == "__main__":
    adjust_and_fuse()
