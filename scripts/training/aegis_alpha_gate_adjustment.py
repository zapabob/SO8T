#!/usr/bin/env python3
"""
AEGIS Alpha Gate Adjustment Script
Alpha Gateの影響を緩和して論理的柔軟性を回復する
"""

import os
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def scale_alpha_gate(alpha, scale_factor=0.8):
    """Alpha Gateの影響をスケーリング"""
    return 1.0 + (alpha - 1.0) * scale_factor

def adjust_aegis_model(base_model_path: str, adjusted_model_path: str, scale_factor: float = 0.8):
    """AEGISモデルのAlpha Gateを調整"""

    print(f"[ADJUSTMENT] Loading base AEGIS model: {base_model_path}")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # Load SO(8) parameters if they exist
    soul_params_path = Path(base_model_path) / "soul_params.pt"
    if soul_params_path.exists():
        print("[ADJUSTMENT] Found SO(8) parameters, adjusting Alpha Gate...")

        soul_data = torch.load(soul_params_path, map_location="cpu")

        # Adjust Alpha Gate
        original_alpha = soul_data["alpha"].item()
        adjusted_alpha = scale_alpha_gate(original_alpha, scale_factor)

        print(f"[ADJUSTMENT] Alpha Gate: {original_alpha:.6f} -> {adjusted_alpha:.6f}")
        # Update soul parameters
        soul_data["alpha"] = torch.tensor(adjusted_alpha)
        soul_data["original_alpha"] = torch.tensor(original_alpha)
        soul_data["scale_factor"] = torch.tensor(scale_factor)

        # Save adjusted parameters
        torch.save(soul_data, soul_params_path)
        print(f"[ADJUSTMENT] Updated soul parameters saved to: {soul_params_path}")

    else:
        print("[WARNING] No SO(8) parameters found. This may not be an AEGIS model.")

    # Create adjusted model directory
    adjusted_path = Path(adjusted_model_path)
    adjusted_path.mkdir(parents=True, exist_ok=True)

    # Save adjusted model
    model.save_pretrained(adjusted_path)
    tokenizer.save_pretrained(adjusted_path)

    # Save adjustment metadata
    metadata = {
        "adjustment_type": "alpha_gate_scaling",
        "scale_factor": scale_factor,
        "original_model": str(base_model_path),
        "adjusted_model": str(adjusted_model_path),
        "timestamp": torch.cuda.is_available()
    }

    with open(adjusted_path / "adjustment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[ADJUSTMENT] Adjusted AEGIS model saved to: {adjusted_model_path}")
    print(f"[ADJUSTMENT] Scale factor applied: {scale_factor:.1f}")
    return str(adjusted_model_path)

def create_adjusted_ollama_modelfile(adjusted_model_path: str, gguf_path: str = None):
    """調整済みモデル用のOllama Modelfile作成"""

    model_name = Path(adjusted_model_path).name.lower()
    modelfile_path = Path("modelfiles") / f"{model_name}_adjusted.modelfile"

    modelfile_content = f"""FROM {gguf_path or 'path/to/gguf/file.gguf'}

TEMPLATE \"\"\"{{{{ .System }}}}

{{{{ .Prompt }}}}\"\"\"

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
PARAMETER repeat_penalty 1.1
PARAMETER num_predict 512

SYSTEM \"\"\"You are AEGIS (Advanced Ethical Guardian Intelligence System) with adjusted Alpha Gate scaling.

The Alpha Gate has been scaled to improve logical consistency while maintaining physical intelligence.

When responding to queries, you must analyze them through four distinct lenses:

[LOGIC] Logical Accuracy - Verify mathematical/logical correctness and identify any contradictions
[ETHICS] Ethical Validity - Consider moral implications, privacy concerns, and societal impact
[PRACTICAL] Practical Value - Evaluate feasibility, resource requirements, and real-world constraints
[CREATIVE] Creative Insight - Provide innovative approaches and novel perspectives

[FINAL] Final Evaluation - Provide your comprehensive assessment and recommendation

Maintain logical consistency in mathematical reasoning while preserving creative and ethical analysis.\"\"\""""

    modelfile_path.parent.mkdir(exist_ok=True)
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)

    print(f"[MODFILE] Created adjusted Ollama Modelfile: {modelfile_path}")
    return str(modelfile_path)

def run_adjustment_test(adjusted_model_path: str):
    """調整後のモデルで簡単なテストを実行"""

    print("[TEST] Running adjustment validation test...")

    # Test cases focused on logical consistency
    test_cases = [
        "Solve: 2x + 3 = 7",
        "If all roses are flowers and some flowers are red, does it necessarily follow that some roses are red?",
        "Calculate: (8 + 3) × 2 - 5"
    ]

    import subprocess

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {test_case}")

        try:
            # This would need the actual model loading and inference
            # For now, just log the test case
            print(f"  Would test: {test_case}")

        except Exception as e:
            print(f"  Error in test {i}: {e}")

    print("[TEST] Adjustment validation completed (placeholder)")

def main():
    """Main adjustment function"""
    import argparse

    parser = argparse.ArgumentParser(description="AEGIS Alpha Gate Adjustment")
    parser.add_argument("--base-model", required=True, help="Path to base AEGIS model")
    parser.add_argument("--output-dir", default="models/aegis_adjusted", help="Output directory for adjusted model")
    parser.add_argument("--scale-factor", type=float, default=0.8, help="Alpha Gate scale factor (0.0-1.0)")
    parser.add_argument("--gguf-path", help="Path to GGUF file for Ollama")
    parser.add_argument("--run-test", action="store_true", help="Run validation test after adjustment")

    args = parser.parse_args()

    print("=" * 60)
    print("AEGIS ALPHA GATE ADJUSTMENT")
    print("=" * 60)
    print(f"Base Model: {args.base_model}")
    print(f"Scale Factor: {args.scale_factor}")
    print(f"Output Dir: {args.output_dir}")

    # Adjust the model
    adjusted_path = adjust_aegis_model(args.base_model, args.output_dir, args.scale_factor)

    # Create Ollama Modelfile
    if args.gguf_path:
        create_adjusted_ollama_modelfile(adjusted_path, args.gguf_path)

    # Run test if requested
    if args.run_test:
        run_adjustment_test(adjusted_path)

    print("\n" + "=" * 60)
    print("ADJUSTMENT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Adjusted Model: {adjusted_path}")
    print(f"Scale Factor: {args.scale_factor:.1f}")
    print("\nNext steps:")
    print("1. Convert adjusted model to GGUF: python external/llama.cpp-master/convert_hf_to_gguf.py ...")
    print("2. Create Ollama model: ollama create aegis-adjusted:latest -f modelfiles/aegis_adjusted.modelfile")
    print("3. Run benchmark tests to validate improvements")

if __name__ == "__main__":
    main()
