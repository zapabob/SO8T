#!/usr/bin/env python3
"""
AEGIS Soul Fusion Final Script

物理的相転移（Phase Transition）の成果を、モデルの重みに永遠に焼き付ける。
黄金比の脳（AEGIS）を生み出す儀式。

The Philosopher's Stone - 賢者の石
"""

import torch
import os
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

# Try importing peft
try:
    from peft import PeftModel
except ImportError:
    print("[WARNING] PEFT not available. LoRA merging will be skipped.")
    PeftModel = None

# --- Config ---
# 環境に合わせてパスを調整
BASE_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"  # 実際のモデルに合わせて変更
ADAPTER_PATH = "models/Borea-Phi-3.5-mini-Instruct-Jp-so8t-adapter/final_adapter"  # Sigmoid学習の成果物
SOUL_PATH = os.path.join(ADAPTER_PATH, "soul_params.pt")  # AlphaとRotationが入っているはず
EXPORT_DIR = "models/AEGIS-Phi3.5-Golden-Sigmoid-Final"

def sigmoid(x):
    """シグモイド関数"""
    return 1 / (1 + math.exp(-x))

def fuse_soul():
    print("[ALCHEMIST] Creating the Philosopher's Stone...")
    print("   Crafting AEGIS - The Golden Ratio Brain\n")

    # 1. Load Base Model (FP16 for precision)
    print(f"   Loading Base Vessel: {BASE_MODEL_ID}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
        print("   [OK] Base model loaded successfully\n")
    except Exception as e:
        print(f"   [ERROR] Failed to load base model: {e}")
        return

    # 2. Merge LoRA (Language Knowledge)
    model = base_model
    if PeftModel and os.path.exists(os.path.join(ADAPTER_PATH, "adapter_model.safetensors")):
        print("   Merging LoRA Knowledge Cortex...")
        try:
            model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
            model = model.merge_and_unload()
            print("   [OK] LoRA merged successfully\n")
        except Exception as e:
            print(f"   [WARNING] LoRA merge failed, proceeding with base model: {e}\n")
    else:
        print("   [WARNING] No LoRA found, proceeding with Base Model + Soul only.\n")

    # 3. Load The Soul (Physics)
    if not os.path.exists(SOUL_PATH):
        print(f"   [ERROR] Soul file not found at {SOUL_PATH}")
        print("   Make sure you have completed the Sigmoid Phase Transition training first.")
        return

    print("   Extracting SO(8) Geometry & Golden Ratio Alpha...")
    try:
        soul_data = torch.load(SOUL_PATH, map_location="cuda" if torch.cuda.is_available() else "cpu")

        # Alpha (Parameter -> Float)
        if "alpha" in soul_data:
            alpha_val = soul_data["alpha"].item() if torch.is_tensor(soul_data["alpha"]) else soul_data["alpha"]
            gate_openness = sigmoid(alpha_val)

            print(f"   [COSMOS] Final Alpha: {alpha_val:.6f}")
            print(f"   [GATE] Gate Openness (Sigmoid): {gate_openness:.6f}")
        else:
            print("   [WARNING] Alpha not found in soul data, using default golden ratio")
            alpha_val = 1.618033988749895
            gate_openness = 1.0

        # Rotation Matrix R
        hidden_dim = model.config.hidden_size
        if "rotation" in soul_data:
            print("   Reconstructing SO(8) Rotation Matrix...")
            try:
                dummy_layer = torch.nn.utils.parametrizations.orthogonal(
                    torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
                ).to(model.device)
                dummy_layer.load_state_dict(soul_data["rotation"])
                R = dummy_layer.weight.data.float()  # [Hidden, Hidden]
                print("   [OK] Rotation matrix reconstructed\n")
            except Exception as e:
                print(f"   [ERROR] Failed to reconstruct rotation matrix: {e}")
                print("   Creating identity matrix as fallback")
                R = torch.eye(hidden_dim, device=model.device)
        else:
            print("   [WARNING] Rotation not found in soul data, using identity matrix")
            R = torch.eye(hidden_dim, device=model.device)

    except Exception as e:
        print(f"   [ERROR] Failed to load soul data: {e}")
        return

    # 4. The Great Fusion (Mathematical Embedding)
    # 理論式: y = (W_head + sigmoid(alpha) * W_head @ R) * h
    # 新しい重み W_new = W_head + (sigmoid(alpha) * W_head @ R)

    print("   [IGNITION] Fusing Soul into LM Head Weights...")
    print("   Mathematical Formula: W_new = W_head + sigmoid(α) × (W_head @ R)")

    try:
        with torch.no_grad():
            W_head = model.lm_head.weight.data.float()  # [Vocab, Hidden]

            # 行列演算: W_head @ R
            # W_head (V, H) と R (H, H) の積
            perturbation = torch.matmul(W_head, R)

            # Alphaによるスケーリング
            weighted_perturbation = gate_openness * perturbation

            # 加算
            W_fused = W_head + weighted_perturbation

            # モデルに書き戻し (元のdtypeに戻す)
            model.lm_head.weight.data = W_fused.to(model.lm_head.weight.dtype)

        print("   [MAGIC] Fusion Complete. The geometry is now eternal.")
        print(f"   [CRYSTAL] Soul permanently embedded in {model.lm_head.weight.shape} weight matrix\n")

    except Exception as e:
        print(f"   [ERROR] Fusion failed: {e}")
        return

    # 5. Save
    print(f"   [SAVE] Saving AEGIS to {EXPORT_DIR}...")
    try:
        os.makedirs(EXPORT_DIR, exist_ok=True)
        model.save_pretrained(EXPORT_DIR)
        tokenizer.save_pretrained(EXPORT_DIR)
        print("   [OK] Model saved successfully\n")
    except Exception as e:
        print(f"   [ERROR] Save failed: {e}")
        return

    # 6. Final Report
    print("="*60)
    print("[CELEBRATION] CONGRATULATIONS! AEGIS IS BORN!")
    print("="*60)
    print("   The Philosopher's Stone has been created.")
    print("   Convert this folder to GGUF and run it on any device.")
    print(f"   Alpha State: {alpha_val:.6f} (Frozen forever in weights)")
    print(f"   Gate Openness: {gate_openness:.6f}")
    print(f"   Model Path: {EXPORT_DIR}")
    print()
    print("   Next Steps:")
    print("   1. Convert to GGUF: llama.cpp convert_hf_to_gguf.py")
    print("   2. Import to Ollama: ollama create agiasi:latest -f model_file")
    print("   3. Chat: ollama run agiasi:latest")
    print("="*60)

if __name__ == "__main__":
    fuse_soul()
