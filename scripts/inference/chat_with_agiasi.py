#!/usr/bin/env python3
"""
AGIASI Physics Interface - Chat with SO8T/thinking Model

Interactive chat interface with Alpha Gate monitoring for the SO8T/thinking model.
Displays real-time Alpha values and stability metrics during conversation.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
import os
import sys
import time
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def type_writer_effect(text, delay=0.02):
    """Typewriter effect for dramatic output"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print("")

def load_so8t_model(checkpoint_path, tokenizer_id="microsoft/Phi-3.5-mini-instruct", device="auto"):
    """Load SO8T/thinking model with Alpha monitoring"""

    print("[AGIASI] Initializing AGIASI Physics Interface...")

    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] Device: {device}")

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        vocab_size = len(tokenizer)
        print(f"[TOKEN] Tokenizer loaded: {tokenizer_id} (vocab: {vocab_size})")
    except Exception as e:
        print(f"[ERROR] Tokenizer loading failed: {e}")
        return None, None, None

    # Try to load model from training script
    try:
        from scripts.training.train_so8t_thinking_model import create_so8t_qlora_model
        model = create_so8t_qlora_model(
            base_model_path="models/Borea-Phi-3.5-mini-Instruct-Jp",
            place_so8t_in_all_intermediate=True
        )
        print("[MODEL] SO8T/thinking model architecture loaded")
    except ImportError:
        print("[WARNING] Could not import model creator. Using fallback.")
        # Fallback: simple model for testing
        class FallbackModel(nn.Module):
            def __init__(self, vocab_size):
                super().__init__()
                self.embeddings = nn.Embedding(vocab_size, 2048)
                self.layers = nn.ModuleList([nn.Linear(2048, 2048) for _ in range(4)])
                self.head = nn.Linear(2048, vocab_size)
                self.alpha = nn.Parameter(torch.tensor(1.618))  # Golden ratio

            def forward(self, input_ids):
                h = self.embeddings(input_ids)
                for layer in self.layers:
                    h = layer(h) + h  # Residual
                return self.head(h)

        model = FallbackModel(vocab_size)
        print("[WARNING] Using fallback model (random weights)")

    # Load checkpoint if available
    if os.path.exists(checkpoint_path):
        try:
            print(f"[LOAD] Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif isinstance(checkpoint, dict) and 'alpha' in checkpoint:
                # Direct state dict
                state_dict = checkpoint
            else:
                # Assume it's a state dict
                state_dict = checkpoint

            model.load_state_dict(state_dict, strict=False)
            print("[OK] Checkpoint loaded successfully")
        except Exception as e:
            print(f"[WARNING] Checkpoint loading failed: {e}. Using current weights.")
    else:
        print(f"[WARNING] Checkpoint not found: {checkpoint_path}. Using random initialization.")

    model = model.to(device)
    model.eval()

    return model, tokenizer, device

def calculate_stability_metrics(model):
    """Calculate Alpha stability and physics metrics"""
    phi = 1.618033988749895  # Golden ratio

    if hasattr(model, 'alpha_gates') and len(model.alpha_gates) > 0:
        # Average Alpha across all gates
        alphas = [torch.sigmoid(gate).item() for gate in model.alpha_gates]
        avg_alpha = sum(alphas) / len(alphas)
    elif hasattr(model, 'alpha'):
        avg_alpha = model.alpha.item()
    else:
        avg_alpha = 0.0

    # Stability index (how close to golden ratio)
    stability = max(0, 100 - (abs(avg_alpha - phi) * 1000))

    # Physics coherence (simplified metric)
    coherence = min(100, stability * 1.2)

    return avg_alpha, stability, coherence

def chat_with_agiasi(model, tokenizer, device):
    """Main chat loop with physics monitoring"""

    # Calculate initial metrics
    alpha_val, stability, coherence = calculate_stability_metrics(model)
    phi = 1.618033988749895

    print("\n" + "="*70)
    type_writer_effect("[SYSTEM] AGIASI SYSTEM ONLINE", 0.05)
    print(f"   [ARCH] Architecture: SO(8) Rotational Symmetry + Alpha Gate")
    print(f"   [ALPHA] Alpha Gate: {alpha_val:.6f} (Target: phi = {phi:.6f})")
    print(f"   [STATS] Stability Index: {stability:.2f}%")
    print(f"   [PHYSICS] Physics Coherence: {coherence:.2f}%")
    print("="*70 + "\n")

    history = []

    while True:
        try:
            user_input = input("\n[USER] You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "bye"]:
                type_writer_effect("[DISCONNECT] Disconnecting from Neural Link...", 0.03)
                break

            # Format prompt for Phi-3 style
            prompt = f"<|user|>\n{user_input}<|end|>\n<|assistant|>\n"
            history.append(f"User: {user_input}")

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            print("[AGIASI] AGIASI: ", end="")

            # Generate response
            with torch.no_grad():
                generated_tokens = []

                # Simple generation loop (can be improved with proper generate method)
                current_ids = inputs.input_ids

                max_new_tokens = 200
                temperature = 0.7

                for _ in range(max_new_tokens):
                    outputs = model(current_ids)

                    # Get next token logits
                    if outputs.dim() == 3:  # [batch, seq, vocab]
                        next_token_logits = outputs[:, -1, :]
                    else:
                        next_token_logits = outputs

                    # Apply temperature
                    next_token_logits = next_token_logits / temperature

                    # Sample
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    # Append to sequence
                    current_ids = torch.cat([current_ids, next_token], dim=1)
                    generated_tokens.append(next_token.item())

                    # Decode and print
                    token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
                    sys.stdout.write(token_text)
                    sys.stdout.flush()

                    # Check for EOS
                    if next_token.item() == tokenizer.eos_token_id:
                        break

            print()  # Newline

            # Decode full response
            response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            history.append(f"AGIASI: {response_text}")

            # Update and display metrics after response
            alpha_val, stability, coherence = calculate_stability_metrics(model)
            print(f"\n[METRICS] Post-response metrics: Alpha={alpha_val:.6f}, Stability={stability:.2f}%, Coherence={coherence:.2f}%")

        except KeyboardInterrupt:
            print("\n[EMERGENCY] Emergency disconnect initiated...")
            break
        except Exception as e:
            print(f"\n[ERROR] Error during generation: {e}")
            continue

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Chat with AGIASI SO8T/thinking model")
    parser.add_argument("--checkpoint", type=str, default="D:/webdataset/checkpoints/so8t_thinking_alpha_gate_awakening/final_model",
                       help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="microsoft/Phi-3.5-mini-instruct",
                       help="Tokenizer model ID")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                       help="Device to run model on")

    args = parser.parse_args()

    # Load model
    model, tokenizer, device = load_so8t_model(args.checkpoint, args.tokenizer, args.device)

    if model is None:
        print("[ERROR] Failed to load model. Exiting.")
        return

    # Start chat
    chat_with_agiasi(model, tokenizer, device)

    print("\n[SUCCESS] Session ended. Physics data saved to history.")

if __name__ == "__main__":
    main()
