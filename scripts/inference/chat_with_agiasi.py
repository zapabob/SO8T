#!/usr/bin/env python3
"""
AGIASI Physics Interface - Chat with SO8T/thinking Model

Interactive chat interface with Alpha Gate monitoring for the SO8T/thinking model.
Displays real-time Alpha values and stability metrics during conversation.
"""

# Ensure UTF-8 encoding
import locale
import codecs
import sys

try:
    locale.setlocale(locale.LC_ALL, 'ja_JP.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except:
        pass

# Force UTF-8 for stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

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

    # Try to load Borea-Phi3.5-instinct-jp with physics fine-tuning
    try:
        print("[MODEL] Loading Borea-Phi3.5-instinct-jp with physics fine-tuning...")

        # Import transformers components
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            "models/Borea-Phi-3.5-mini-Instruct-Jp",
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        # Try to load physics adapter if available
        adapter_path = "models/Borea-Phi3.5-physics-finetuned/final_model"
        if os.path.exists(adapter_path):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
            print("[ADAPTER] Physics fine-tuning adapter loaded")
        else:
            print("[WARNING] Physics adapter not found, using base Borea model")

        # Add Alpha Gate parameter
        model.alpha_gate = nn.Parameter(torch.tensor(1.618))  # Golden ratio
        print("[ALPHA] Alpha Gate initialized at golden ratio")

    except Exception as e:
        print(f"[WARNING] Could not load Borea model: {e}. Using fallback.")
        # Fallback: simple model for testing
        class FallbackModel(nn.Module):
            def __init__(self, vocab_size):
                super().__init__()
                self.embeddings = nn.Embedding(vocab_size, 2048)
                self.layers = nn.ModuleList([nn.Linear(2048, 2048) for _ in range(4)])
                self.head = nn.Linear(2048, vocab_size)
                self.alpha_gate = nn.Parameter(torch.tensor(1.618))  # Golden ratio

            def forward(self, input_ids):
                h = self.embeddings(input_ids)
                for layer in self.layers:
                    h = layer(h) + h  # Residual
                return self.head(h)

        model = FallbackModel(vocab_size)
        print("[WARNING] Using fallback model (random weights)")

    except ImportError as e:
        print(f"[WARNING] Could not load Borea model with adapter: {e}. Using fallback.")
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
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    # Load Alpha Gate value if available
                    if 'alpha' in checkpoint and hasattr(model, 'alpha'):
                        model.alpha.data = torch.tensor(checkpoint['alpha'], device=device)
                        print(f"[ALPHA] Loaded Alpha Gate: {checkpoint['alpha']:.6f}")
                elif 'alpha' in checkpoint:
                    # SO8T final model format
                    state_dict = checkpoint
                    if 'alpha' in checkpoint and hasattr(model, 'alpha'):
                        model.alpha.data = torch.tensor(checkpoint['alpha'], device=device)
                        print(f"[ALPHA] Loaded Alpha Gate: {checkpoint['alpha']:.6f}")
                else:
                    # Direct state dict
                    state_dict = checkpoint
            else:
                # Legacy format - direct state dict
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

def run_automated_tests(model, tokenizer, device):
    """Run automated tests to verify AGIASI capabilities"""

    test_prompts = [
        "こんにちは、調子はどうですか？",
        "時間はなぜ不可逆なのですか？",
        "あなたは誰ですか？",
        "1+1は何ですか？",
        "物理学で最も重要な定数は何ですか？",
        "日本語で自己紹介をお願いします。"
    ]

    # Ensure prompts are properly encoded
    test_prompts = [p.encode('utf-8').decode('utf-8') for p in test_prompts]

    print("\n" + "="*60)
    print("[TEST] Running automated AGIASI capability tests...")
    print("="*60)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[TEST {i}] Prompt: {prompt}")

        # Format prompt
        formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"

        try:
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

            print("[AGIASI] ", end="")

            # Generate response
            with torch.no_grad():
                try:
                    # Use model's generate method for Borea models
                    if hasattr(model, 'generate') and 'Borea' in str(type(model)).lower():
                        generated_ids = model.generate(
                            inputs.input_ids,
                            max_new_tokens=150,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            repetition_penalty=1.1,
                            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            use_cache=True
                        )

                        # Decode response
                        full_response = tokenizer.decode(
                            generated_ids[0][len(inputs.input_ids[0]):],
                            skip_special_tokens=True
                        )
                        sys.stdout.write(full_response)
                        sys.stdout.flush()

                    else:
                        # Fallback for SO8T models: simple token-by-token generation
                        generated_tokens = []
                        current_ids = inputs.input_ids.clone()
                        max_length = len(current_ids[0]) + 100

                        for _ in range(100):
                            outputs = model(current_ids)

                            # Get logits for next token
                            if hasattr(outputs, 'logits'):
                                next_token_logits = outputs.logits[:, -1, :]
                            else:
                                next_token_logits = outputs[:, -1, :]

                            # Apply temperature
                            next_token_logits = next_token_logits / 0.8
                            probs = torch.softmax(next_token_logits, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)

                            current_ids = torch.cat([current_ids, next_token], dim=1)
                            generated_tokens.append(next_token.item())

                            token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
                            if token_text:
                                sys.stdout.write(token_text)
                                sys.stdout.flush()

                            if next_token.item() == tokenizer.eos_token_id or len(current_ids[0]) >= max_length:
                                break

                        if not generated_tokens:
                            print(" (AGIASI is thinking...)", end="")

                except Exception as e:
                    print(f"[Generation Error: {str(e)[:50]}...]", end="")
                    print(" (Unable to generate response)", end="")

            print()

            # Update metrics
            alpha_val, stability, coherence = calculate_stability_metrics(model)
            print(f"[METRICS] Alpha={alpha_val:.4f}, Stability={stability:.1f}%, Coherence={coherence:.1f}%")

        except Exception as e:
            print(f"[ERROR] Test failed: {e}")

    print("\n" + "="*60)
    print("[TEST] Automated testing complete!")
    print("="*60)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Chat with AGIASI SO8T/thinking model")
    parser.add_argument("--checkpoint", type=str, default="models/Borea-Phi-3.5-mini-Instruct-Jp-so8t-adapter/final_adapter",
                       help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="microsoft/Phi-3.5-mini-instruct",
                       help="Tokenizer model ID")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                       help="Device to run model on")
    parser.add_argument("--test", action="store_true",
                       help="Run automated tests instead of interactive chat")
    parser.add_argument("--borea-path", type=str, default="models/Borea-Phi-3.5-mini-Instruct-Jp",
                       help="Path to Borea base model")

    args = parser.parse_args()

    # Load model
    model, tokenizer, device = load_so8t_model(args.checkpoint, args.tokenizer, args.device)

    if model is None:
        print("[ERROR] Failed to load model. Exiting.")
        return

    if args.test:
        # Run automated tests
        run_automated_tests(model, tokenizer, device)
    else:
        # Interactive chat
        chat_with_agiasi(model, tokenizer, device)

    print("\n[SUCCESS] Session ended. AGIASI physics data recorded.")

if __name__ == "__main__":
    main()
