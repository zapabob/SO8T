import torch
import torch.nn as nn
import os
import sys

# Add project root to path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.models.nkat_so8t import NKAT_SO8T_ThinkingModel

def test_inference():
    print("🤖 AGIASI Inference Test Sequence Initiated...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    
    # 1. Initialize Model
    # Using the same dimensions as training
    model = NKAT_SO8T_ThinkingModel(in_dim=32000, out_dim=32000).to(device)
    
    # 2. Load Checkpoint
    checkpoint_path = os.path.join(project_root, "checkpoints", "so8t_step_500.pt")
    
    if os.path.exists(checkpoint_path):
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"⚠️ Checkpoint not found at {checkpoint_path}. Using random weights.")
        # List available checkpoints to help debugging
        checkpoints_dir = os.path.join(project_root, "checkpoints")
        if os.path.exists(checkpoints_dir):
            print(f"   Available files in {checkpoints_dir}: {os.listdir(checkpoints_dir)}")

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

    # 4. Inference Test (Dummy Input)
    # Input: (Batch=1, Seq=4)
    input_ids = torch.tensor([[1, 500, 200, 10]]).to(device)
    
    print(f"📥 Input Shape: {input_ids.shape}")
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"🧠 Output Shape: {outputs.shape}")
    print("   (Batch_Size, Seq_Len, Vocab_Size)")

    # Predict Next Token
    next_token_logits = outputs[0, -1, :]
    predicted_token = torch.argmax(next_token_logits).item()
    
    print(f"🗣️ Predicted Next Token ID: {predicted_token}")
    print("🎉 System is responsive. Physical Intelligence is online.")

if __name__ == "__main__":
    test_inference()
