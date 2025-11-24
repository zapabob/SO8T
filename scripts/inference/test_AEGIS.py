import torch
import torch.nn as nn
import os
import sys

# Add project root to path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.models.nkat_so8t import NKAT_SO8T_ThinkingModel

def test_inference():
    print("🤖 AEGIS Inference Test Sequence Initiated...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    
    # 1. Initialize Model
    # Vocab size for Phi-3.5 is 32011 (or check tokenizer len)
    # We need to load tokenizer first to get correct vocab size
    print("🔡 Loading Tokenizer (microsoft/Phi-3.5-mini-instruct)...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)
    vocab_size = len(tokenizer)
    print(f"   Vocab Size: {vocab_size}")

    model = NKAT_SO8T_ThinkingModel(in_dim=vocab_size, out_dim=vocab_size).to(device)
    
    # 2. Load Checkpoint
    # Check what files exist
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    files = os.listdir(checkpoint_dir)
    # Sort by step number if possible, or just take the last one
    files = sorted([f for f in files if f.startswith("so8t_step_") and f.endswith(".pt")], 
                   key=lambda x: int(x.split('_')[2].split('.')[0]))
    
    if files:
        latest_checkpoint = files[-1]
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        print(f"📂 Loading latest checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"⚠️ No checkpoints found in {checkpoint_dir}. Using random weights.")

    model.eval()

    # 3. Check Alpha Gate
    current_alpha = model.alpha.item()
    print(f"✨ Current Alpha Gate: {current_alpha:.6f}")
    
    phi = 1.6180339887
    diff = abs(current_alpha - phi)
    
    if diff < 0.01:
        print("✅ Alpha is at the Golden Ratio. The Mass Gap is active.")
    else:
        print("⚠️ Alpha is drifting. Did you load the right checkpoint?")

    # 4. Inference Test (Japanese Input)
    prompt = "こんにちは、調子はどうですか？"
    print(f"🗣️ Input Prompt: {prompt}")
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"📥 Input Shape: {input_ids.shape}")
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"🧠 Output Shape: {outputs.shape}")

    # Predict Next Token
    next_token_logits = outputs[0, -1, :]
    predicted_token_id = torch.argmax(next_token_logits).item()
    predicted_token = tokenizer.decode([predicted_token_id])
    
    print(f"🤖 Predicted Next Token: '{predicted_token}' (ID: {predicted_token_id})")
    
    # Simple Generation Loop
    print("📝 Generating continuation...")
    generated_ids = input_ids[0].tolist()
    
    for _ in range(20):
        input_tensor = torch.tensor([generated_ids]).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
        
        next_token_logits = outputs[0, -1, :]
        # Greedy decode
        next_id = torch.argmax(next_token_logits).item()
        generated_ids.append(next_id)
        
        if next_id == tokenizer.eos_token_id:
            break
            
    generated_text = tokenizer.decode(generated_ids)
    print(f"📜 Full Generated Text: {generated_text}")
    print("🎉 System is responsive. Physical Intelligence is online.")

if __name__ == "__main__":
    test_inference()
