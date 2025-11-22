import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import math

# Add project root to path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.models.nkat_so8t import NKAT_SO8T_ThinkingModel

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

def train():
    parser = argparse.ArgumentParser(description="NKAT-SO8T Phase Transition Training")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--annealing-warmup", type=int, default=50)
    parser.add_argument("--annealing-steps", type=int, default=400)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--enable-mass-gap-monitor", action="store_true")
    
    args = parser.parse_args()

    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    # Dummy Data (Batch size 8, Sequence length 128)
    # In a real scenario, use a DataLoader
    dummy_input = torch.randint(0, 32000, (8, 128)).to(device)
    dummy_targets = torch.randint(0, 32000, (8, 128)).to(device)
    
    # Model Initialization
    model = NKAT_SO8T_ThinkingModel(in_dim=32000, out_dim=32000).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # --- 2. Training Loop (Phase Transition Observation) ---
    print(f"🔥 Ignition! Starting Phase Transition Sequence...")
    print(f"   Target Alpha: {1.6180339887} (Golden Ratio)")
    print(f"   Schedule: Warmup={args.annealing_warmup}, Annealing={args.annealing_steps}")

    model.train()
    
    # Target Golden Ratio
    PHI = 1.6180339887
    START_ALPHA = -5.0

    for step in range(args.max_steps):
        optimizer.zero_grad()

        # --- A. Alpha Annealing ---
        current_alpha_val = linear_annealing(
            step, args.annealing_warmup, args.annealing_steps, START_ALPHA, PHI
        )
        
        # Force update Alpha (Parameter)
        model.alpha.data.fill_(current_alpha_val)

        # --- B. Forward ---
        outputs = model(dummy_input) # (B, Seq, Vocab)
        
        # Loss Calculation
        # Task Loss
        task_loss = criterion(outputs.reshape(-1, 32000), dummy_targets.view(-1))
        
        # Orthogonality Loss
        ortho_loss = getattr(model, "ortho_loss", 0.0) 
        
        # Total Loss
        loss = task_loss + ortho_loss

        loss.backward()
        optimizer.step()

        # --- C. Observation (Logging) ---
        if step % args.logging_steps == 0:
            status = "🔵 Stable"
            if step > args.annealing_warmup and step < (args.annealing_warmup + args.annealing_steps):
                status = "🟡 Transitioning"
            if abs(current_alpha_val - PHI) < 0.01:
                status = "🟢 Golden Ratio Reached"

            # Check for Mass Gap Spike (heuristic)
            if args.enable_mass_gap_monitor and step > 0:
                 # In a real monitor, we'd track loss history. 
                 # Here we just print the current state.
                 pass

            print(f"[Step {step:03d}/{args.max_steps}] "
                  f"Alpha: {current_alpha_val:.6f} | "
                  f"Loss: {loss.item():.6f} (Ortho: {ortho_loss:.6f}) | "
                  f"Status: {status}")

        # --- D. Save Checkpoint ---
        if step > 0 and step % args.save_steps == 0:
            save_dir = os.path.join(project_root, "checkpoints")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"so8t_step_{step}.pt")
            torch.save(model.state_dict(), save_path)
            # print(f"   💾 Checkpoint saved: {save_path}")

    print("✅ Training sequence complete. Alpha Gate is now fully open.")
    print(f"   Final Alpha: {model.alpha.item():.6f}")

if __name__ == "__main__":
    train()
