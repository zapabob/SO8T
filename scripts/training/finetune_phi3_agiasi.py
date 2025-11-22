import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import math
from tqdm import tqdm

# Set protobuf env var
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Add project root to path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Try importing datasets and transformers
try:
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
    from torch.utils.data import DataLoader
except ImportError:
    print("❌ Missing required libraries: `datasets` or `transformers`.")
    sys.exit(1)

def linear_annealing(step, warmup_steps, annealing_steps, start_alpha, target_alpha):
    """Alpha Gate linear annealing schedule"""
    if step < warmup_steps:
        return start_alpha
    elif step < warmup_steps + annealing_steps:
        # Progress (0.0 -> 1.0)
        progress = (step - warmup_steps) / annealing_steps
        return start_alpha + progress * (target_alpha - start_alpha)
    else:
        return target_alpha

def calculate_orthogonality_loss(model):
    """
    Calculate SO(8) Orthogonality Loss for Phi-3 MLP layers.
    We target the 'down_proj' or 'gate_up_proj' weights.
    Phi-3 structure usually has: model.layers[i].mlp.down_proj.weight
    """
    ortho_loss = 0.0
    count = 0
    
    # Iterate through modules to find linear layers in MLP
    for name, module in model.named_modules():
        if "mlp" in name and "down_proj" in name and isinstance(module, nn.Linear):
            # Weight shape: (out_features, in_features)
            # We want to enforce orthogonality on chunks of the weight matrix
            # simulating SO(8) blocks.
            W = module.weight
            
            # Reshape to (N, 8, 8) blocks if possible, or just enforce on the whole matrix
            # For efficiency and "Mass Gap" effect, let's enforce on 8x8 blocks along the diagonal
            # or just random blocks.
            # Simplified approach: Enforce R^T R = I on the first 8x8 block for demonstration
            # or a sum of 8x8 blocks.
            
            # Let's take the first 64x64 chunk and break it into 8x8 blocks
            if W.shape[0] >= 64 and W.shape[1] >= 64:
                w_chunk = W[:64, :64] # (64, 64)
                # Reshape to (8, 8, 8, 8) -> (Block_Row, Block_Col, 8, 8)
                # Actually, let's just treat it as 64 vectors of dim 64 and enforce orthogonality?
                # Too expensive.
                
                # AGIASI Approach: "Mass Gap" via SO(8).
                # We want the weights to approximate SO(8) rotations.
                # Let's take diagonal 8x8 blocks.
                for i in range(0, 64, 8):
                    block = W[i:i+8, i:i+8] # (8, 8)
                    # Loss = || W^T W - I ||^2
                    I = torch.eye(8, device=W.device)
                    # Use matrix multiplication
                    gram = torch.matmul(block.T, block)
                    # Normalize block to avoid exploding loss if weights are large
                    # But we want to force them to be orthonormal, so magnitude 1 is desired.
                    loss = torch.norm(gram - I)
                    ortho_loss += loss
                    count += 1
                    
    if count > 0:
        return ortho_loss / count
    return torch.tensor(0.0, device=model.device)

def train():
    parser = argparse.ArgumentParser(description="Phi-3.5 AGIASI Upgrade (GGUF Compatible)")
    parser.add_argument("--max-steps", type=int, default=100) # Short fine-tuning
    parser.add_argument("--annealing-warmup", type=int, default=10)
    parser.add_argument("--annealing-steps", type=int, default=80)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1) # Small batch for GPU memory
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    
    args = parser.parse_args()

    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    # --- 2. Data Loading (Borea-Phi3.5 Knowledge) ---
    print("📚 Loading Knowledge: TFMC/imatrix-dataset-for-japanese-llm...")
    dataset = load_dataset("TFMC/imatrix-dataset-for-japanese-llm", split="train", streaming=True)
    
    # Tokenizer (Phi-3.5)
    model_id = "microsoft/Phi-3.5-mini-instruct"
    print(f"🔡 Loading Tokenizer ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.seq_len, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size, collate_fn=data_collator)
    
    # --- 3. Model Initialization ---
    print(f"🧠 Loading Model ({model_id})...")
    # Load in 4bit or 8bit if possible to save memory, but for fine-tuning we usually need full precision or LoRA.
    # Since we are modifying weights directly for GGUF, we should load in float16 or bfloat16.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Enable gradients
    model.train()
    # We want to update all parameters or just MLP? 
    # To "bake in" AGIASI, we should probably update MLP weights primarily, 
    # but standard fine-tuning updates everything.
    # Let's use a standard optimizer for all params.
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # --- 4. Training Loop (Phase Transition) ---
    print(f"🔥 Ignition! Starting AGIASI Upgrade Sequence...")
    print(f"   Target Alpha: {1.6180339887} (Golden Ratio)")
    
    PHI = 1.6180339887
    START_ALPHA = -5.0
    
    step = 0
    iterator = iter(train_dataloader)
    progress_bar = tqdm(range(args.max_steps), desc="AGIASI Upgrade")

    for step in progress_bar:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_dataloader)
            batch = next(iterator)
            
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()

        # --- A. Alpha Annealing ---
        current_alpha_val = linear_annealing(
            step, args.annealing_warmup, args.annealing_steps, START_ALPHA, PHI
        )
        
        # --- B. Forward ---
        outputs = model(input_ids=input_ids, labels=labels)
        task_loss = outputs.loss
        
        # --- C. Orthogonality Loss (Physics) ---
        ortho_loss = calculate_orthogonality_loss(model)
        
        # Weight the ortho loss by Alpha (Gate)
        # As Alpha approaches Golden Ratio, we enforce structure more?
        # Or does Alpha represent the "opening" of the gate?
        # Let's say Alpha controls the balance.
        # Sigmoid(-5) ~ 0, Sigmoid(1.6) ~ 0.8
        # We want Orthogonality to be enforced as we transition.
        gate = torch.sigmoid(torch.tensor(current_alpha_val))
        
        # Total Loss
        # We add ortho_loss * gate * coefficient
        total_loss = task_loss + ortho_loss * gate * 0.1

        total_loss.backward()
        optimizer.step()

        # --- D. Logging ---
        if step % args.logging_steps == 0:
            status = "🔵 Stable"
            if step > args.annealing_warmup and step < (args.annealing_warmup + args.annealing_steps):
                status = "🟡 Transitioning"
            if abs(current_alpha_val - PHI) < 0.01:
                status = "🟢 Golden Ratio Reached"

            progress_bar.set_postfix({
                "Alpha": f"{current_alpha_val:.4f}",
                "Loss": f"{task_loss.item():.4f}",
                "Ortho": f"{ortho_loss.item():.4f}",
                "Status": status
            })

        # --- E. Save Checkpoint ---
        if step > 0 and step % args.save_steps == 0:
            save_dir = os.path.join(project_root, "checkpoints_agiasi")
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            
        if step >= args.max_steps:
            break

    print("\n✅ AGIASI Upgrade complete. Model is ready for GGUF conversion.")
    
    # Final Save
    final_save_dir = os.path.join(project_root, "agiasi_phi3_final")
    model.save_pretrained(final_save_dir)
    tokenizer.save_pretrained(final_save_dir)
    print(f"   💾 Model saved to: {final_save_dir}")

if __name__ == "__main__":
    train()
