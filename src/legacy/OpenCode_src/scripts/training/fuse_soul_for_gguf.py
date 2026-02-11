import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Config ---
BASE_MODEL = "models/Borea-Phi-3.5-mini-Instruct-Jp"
CHECKPOINT_PATH = "outputs/so8t_ppo_training/20251201_223144/checkpoint_1000.pt"  # PPO training checkpoint
EXPORT_DIR = "models/AEGIS-Phi3.5-Thinking-v2"
SOUL_PATH = True  # Not used for PPO training

def fuse_and_export():
    print("[FUSION] Alchemist Mode: Fusing Soul into Weights...")

    # 1. Load Base Model in FP16 (CPU/GPU) for merging
    # Note: Merging requires higher precision than 4bit
    print("   Loading Base Model (FP16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # 2. Load PPO checkpoint and merge weights
    print("   Loading PPO checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        print("   Merging PPO trained weights...")
        # Load the trained weights into the base model
        model_state_dict = checkpoint['model_state_dict']

        # Filter out SO8T adapter keys that might cause issues during loading
        filtered_state_dict = {}
        for key, value in model_state_dict.items():
            # Skip SO8T adapter parameters for now, or handle them specially
            if 'so8_adapter' not in key:
                filtered_state_dict[key] = value
            else:
                print(f"   Skipping SO8T adapter parameter: {key}")

        # Load the filtered state dict
        missing_keys, unexpected_keys = base_model.load_state_dict(filtered_state_dict, strict=False)
        if missing_keys:
            print(f"   Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"   Unexpected keys: {len(unexpected_keys)}")

        model = base_model
    else:
        print("   No model_state_dict found in checkpoint, using base model")
        model = base_model

    # 3. SO(8) Integration
    if SOUL_PATH and os.path.exists(SOUL_PATH):
        print("   Loading Soul Parameters...")
        soul_data = torch.load(SOUL_PATH)
        alpha = soul_data["alpha"].to(model.device).float() # Compute in float32 for precision
        rotation_state = soul_data["rotation"]

        # Reconstruct Rotation Matrix R
        # Orthogonal parametrization stores weights differently, need to instantiate to get W
        hidden_dim = model.config.hidden_size
        rot_layer = torch.nn.utils.parametrizations.orthogonal(
            torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        ).to(model.device)
        rot_layer.load_state_dict(rotation_state)
        R = rot_layer.weight.data.float() # Matrix R

        # 4. Mathematical Fusion (The Magic)
        # y = W_head * (h + sigmoid(alpha) * R * h)
        # y = (W_head + sigmoid(alpha) * W_head * R) * h
        # New_Head = W_head + (sigma(alpha) * W_head @ R)

        print(f"   Fusing Alpha ({alpha.item():.4f}) and Rotation into LM Head...")
    else:
        print("   Skipping Soul fusion (PPO training doesn't use separate soul parameters)...")

    with torch.no_grad():
        W_head = model.lm_head.weight.data.float() # [vocab, hidden]

        # Calculate term: sigma(alpha) * W_head @ R
        # Note: Linear layer x is usually x @ W.T.
        # But PyTorch stores weight as [out_features, in_features].
        # So y = x @ W.T.
        # h' = h + sigma * h @ R.T
        # y = h' @ W_head.T
        # y = (h + sigma * h @ R.T) @ W_head.T
        # y = h @ W_head.T + sigma * h @ R.T @ W_head.T
        # y = h @ (W_head + sigma * W_head @ R).T
        # So New_Weight = W_head + sigma * (W_head @ R)

        sigma_alpha = torch.sigmoid(alpha)

        # Modification matrix
        # W_head: [32064, 3072], R: [3072, 3072]
        # Result: [32064, 3072]
        delta_W = sigma_alpha * torch.matmul(W_head, R)

        # Update Head
        model.lm_head.weight.data = (W_head + delta_W).to(model.lm_head.weight.dtype)

    print("   [FUSION] Fusion Complete. The Soul is now part of the structure.")

    # 5. Save Final Model
    print(f"   Saving to {EXPORT_DIR}...")
    model.save_pretrained(EXPORT_DIR)
    tokenizer.save_pretrained(EXPORT_DIR)
    print("[FUSION] Done! Ready for GGUF conversion.")

if __name__ == "__main__":
    fuse_and_export()
