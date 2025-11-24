import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Config ---
BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
ADAPTER_DIR = "models/Borea-Phi-3.5-mini-Instruct-Jp-so8t-adapter/final_adapter"
SOUL_PATH = "models/Borea-Phi-3.5-mini-Instruct-Jp-so8t-adapter/final_adapter/soul_params.pt"
EXPORT_DIR = "models/AEGIS-Phi3.5-Hybrid"

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

    # 2. Merge LoRA
    print("   Merging LoRA Adapters...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model = model.merge_and_unload() # LoRA is now baked into standard weights

    # 3. Load Soul Parameters
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
