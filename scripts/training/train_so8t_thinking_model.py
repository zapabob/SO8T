import sys
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import math
from tqdm import tqdm

# Add project root to path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.models.nkat_so8t import NKAT_SO8T_ThinkingModel

# Try importing datasets and transformers
try:
    from datasets import load_dataset
    from transformers import AutoTokenizer, DataCollatorForLanguageModeling
    from torch.utils.data import DataLoader
except ImportError:
    print("[ERROR] Missing required libraries: `datasets` or `transformers`.")
    print("   Please install them via: `pip install datasets transformers`")
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

def train():
    parser = argparse.ArgumentParser(description="NKAT-SO8T Phase Transition Training (TinyStories)")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--annealing-warmup", type=int, default=100)
    parser.add_argument("--annealing-steps", type=int, default=800)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--enable-mass-gap-monitor", action="store_true")
    parser.add_argument("--monitor-interval", type=int, default=25)
    
    args = parser.parse_args()

    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[START] Device: {device}")

    # --- 2. Data Loading (Borea-Phi3.5-instinct-jp Knowledge) ---
    print("[DATA] Loading Knowledge: TFMC/imatrix-dataset-for-japanese-llm...")
    # Load streaming to avoid downloading the whole thing
    dataset = load_dataset("TFMC/imatrix-dataset-for-japanese-llm", split="train", streaming=True)
    
    # Tokenizer (Phi-3.5)
    print("[TOKEN] Loading Tokenizer (microsoft/Phi-3.5-mini-instruct)...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.seq_len, padding="max_length")

    # Create an iterable dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # DataLoader
    train_dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size, collate_fn=data_collator)
    
    # --- 3. Model Initialization ---
    vocab_size = len(tokenizer)
    print(f"[MODEL] Initializing Model (Vocab Size: {vocab_size})...")
    
    model = NKAT_SO8T_ThinkingModel(in_dim=vocab_size, out_dim=vocab_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # --- 4. Training Loop (Phase Transition Observation) ---
    print(f"[IGNITION] Starting Phase Transition Sequence...")
    print(f"   Target Alpha: {1.6180339887} (Golden Ratio)")
    print(f"   Schedule: Warmup={args.annealing_warmup}, Annealing={args.annealing_steps}")

    model.train()
    
    # Target Golden Ratio
    PHI = 1.6180339887
    START_ALPHA = -5.0
    
    step = 0
    iterator = iter(train_dataloader)
    
    progress_bar = tqdm(range(args.max_steps), desc="Training")

    for step in progress_bar:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_dataloader)
            batch = next(iterator)
            
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device) # Auto-shifted by DataCollator? No, usually labels=input_ids for CausalLM
        
        # DataCollatorForLanguageModeling with mlm=False sets labels = input_ids, 
        # and handles -100 for padding if configured, but let's check.
        # Standard practice: inputs = input_ids, targets = input_ids (shifted inside model or loss)
        # Here we do manual loss calc, so we need to shift.
        
        optimizer.zero_grad()

        # --- A. Alpha Annealing ---
        current_alpha_val = linear_annealing(
            step, args.annealing_warmup, args.annealing_steps, START_ALPHA, PHI
        )
        
        # Force update Alpha (Parameter)
        model.alpha.data.fill_(current_alpha_val)

        # --- B. Forward ---
        outputs = model(input_ids) # (B, Seq, Vocab)
        
        # Shift for Causal LM Loss
        # Logits: [..., :-1, :] -> Predict next token
        # Labels: [..., 1:] -> Next token
        shift_logits = outputs[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Task Loss
        task_loss = criterion(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        
        # Orthogonality Loss
        ortho_loss = getattr(model, "ortho_loss", 0.0) 
        
        # Total Loss
        loss = task_loss + ortho_loss

        loss.backward()
        optimizer.step()

        # --- C. Observation (Logging) ---
        if step % args.logging_steps == 0:
            status = "[STABLE] Stable"
            if step > args.annealing_warmup and step < (args.annealing_warmup + args.annealing_steps):
                status = "[TRANSITION] Transitioning"
            if abs(current_alpha_val - PHI) < 0.01:
                status = "[TARGET] Golden Ratio Reached"

            progress_bar.set_postfix({
                "Alpha": f"{current_alpha_val:.4f}",
                "Loss": f"{loss.item():.4f}",
                "Status": status
            })

        # --- D. Save Checkpoint ---
        if step > 0 and step % args.save_steps == 0:
            save_dir = os.path.join(project_root, "checkpoints")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"so8t_step_{step}.pt")
            torch.save(model.state_dict(), save_path)
            
        if step >= args.max_steps:
            break

    # Save final model
    final_save_path = os.path.join(project_root, "checkpoints", "so8t_final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'alpha': model.alpha.item(),
        'step': step,
        'final_loss': loss.item()
    }, final_save_path)

    print("\n[COMPLETE] Training sequence complete. Alpha Gate is now fully open.")
    print(f"   Final Alpha: {model.alpha.item():.6f}")
    print(f"   Final Loss: {loss.item():.4f}")
    print(f"   Model saved to: {final_save_path}")

if __name__ == "__main__":
    train()
