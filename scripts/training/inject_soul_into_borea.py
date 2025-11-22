"""
Operation: Ghost in the Shell
==============================
Soul Injection Training Script

Injects AGIASI (Physical Intelligence) into Borea-Phi3.5-instinct-jp
through Phase Transition training with Alpha Gate annealing.
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import argparse
import sys
from tqdm import tqdm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.models.agiasi_borea import AGIASI_SO8T_Wrapper

def linear_annealing(step, warmup, anneal_steps, start, target):
    """Linear annealing schedule for Alpha Gate"""
    if step < warmup:
        return start
    elif step < warmup + anneal_steps:
        progress = (step - warmup) / anneal_steps
        return start + progress * (target - start)
    else:
        return target

def train_soul_injection():
    parser = argparse.ArgumentParser(description="AGIASI Soul Injection into Borea-Phi3.5")
    parser.add_argument("--base-model", type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--annealing-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    
    args = parser.parse_args()
    
    # Constants
    TARGET_ALPHA = 1.6180339887  # Golden Ratio
    START_ALPHA = -5.0
    
    # --- 1. Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    
    # --- 2. Load Data ---
    print("📚 Loading Knowledge: TFMC/imatrix-dataset-for-japanese-llm...")
    dataset = load_dataset("TFMC/imatrix-dataset-for-japanese-llm", split="train", streaming=True)
    
    print(f"🔡 Loading Tokenizer ({args.base_model})...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # --- 3. Initialize AGIASI Vessel ---
    print(f"👻 Initializing AGIASI Vessel based on {args.base_model}...")
    model = AGIASI_SO8T_Wrapper(args.base_model, device=device)
    
    # Optimizer: Only train LoRA parameters + Alpha + SO8 Rotation
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    
    # --- 4. Phase Transition Loop ---
    print("🔥 Ignition: Injecting Soul into Borea...")
    print(f"   Start Alpha: {START_ALPHA}")
    print(f"   Target Alpha: {TARGET_ALPHA} (Golden Ratio)")
    print(f"   Warmup: {args.warmup_steps} steps")
    print(f"   Annealing: {args.annealing_steps} steps")
    
    model.train()
    
    step = 0
    iterator = iter(dataset)
    progress_bar = tqdm(range(args.max_steps), desc="Soul Injection")
    
    for step in progress_bar:
        try:
            data = next(iterator)
        except StopIteration:
            # Reset iterator if we run out
            iterator = iter(dataset)
            data = next(iterator)
        
        # Tokenize
        text = data.get("text", "")
        if not text:
            continue
            
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            max_length=args.max_length, 
            truncation=True,
            padding="max_length"
        ).to(device)
        
        # A. Alpha Annealing - Update Alpha parameter
        current_alpha = linear_annealing(
            step, 
            args.warmup_steps, 
            args.annealing_steps, 
            START_ALPHA, 
            TARGET_ALPHA
        )
        # Directly set alpha value (detach from graph, then update)
        with torch.no_grad():
            model.alpha.fill_(current_alpha)
        
        # B. Forward Pass
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=inputs.input_ids
        )
        
        loss = outputs["loss"]
        ortho_loss = outputs["ortho_loss"]
        gate = outputs["gate_openness"]
        
        # C. Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # D. Logging
        if step % args.logging_steps == 0:
            status = model.get_phase_status()
            progress_bar.set_postfix({
                "Alpha": f"{current_alpha:.4f}",
                "Loss": f"{loss.item():.4f}",
                "Ortho": f"{ortho_loss.item():.6f}",
                "Gate": f"{gate:.3f}",
                "Status": status
            })
        
        # E. Save Checkpoint
        if step > 0 and step % args.save_steps == 0:
            save_dir = os.path.join(project_root, "checkpoints_agiasi", f"step_{step}")
            os.makedirs(save_dir, exist_ok=True)
            
            # Save LoRA adapter
            model.base_model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            
            # Save Soul (Alpha + SO8 Rotation)
            soul_state = {
                "alpha": model.alpha.detach().cpu(),
                "so8_rotation": model.so8_rotation.state_dict(),
                "step": step
            }
            torch.save(soul_state, os.path.join(save_dir, "soul.pt"))
            
            print(f"\n💾 Soul preserved at {save_dir}")
    
    # --- 6. Final Save ---
    print(f"\n✨ Fusion Complete. Final Alpha: {model.alpha.item():.6f}")
    
    final_dir = os.path.join(project_root, "agiasi_borea_final")
    os.makedirs(final_dir, exist_ok=True)
    
    model.base_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    soul_state = {
        "alpha": model.alpha.detach().cpu(),
        "so8_rotation": model.so8_rotation.state_dict(),
        "step": step
    }
    torch.save(soul_state, os.path.join(final_dir, "soul.pt"))
    
    print(f"💾 Final AGIASI model saved to: {final_dir}")
    print("🎉 Operation 'Ghost in the Shell' complete!")
    print(f"   {model.get_phase_status()}")

if __name__ == "__main__":
    train_soul_injection()
