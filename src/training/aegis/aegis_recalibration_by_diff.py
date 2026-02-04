import torch
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

def recalibrate_by_diff():
    parser = argparse.ArgumentParser(description="AEGIS Recalibration by Difference Scaling")
    parser.add_argument("--base-model", type=str, default="Borea/Phi-3.5-instinct-jp", help="Original Base Model ID")
    parser.add_argument("--source-model", type=str, required=True, help="Path to the existing fused model (e.g. aegis_0.8)")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to save the adjusted model")
    parser.add_argument("--scale-factor", type=float, default=0.6, help="Scaling factor for the perturbation")
    
    args = parser.parse_args()

    from pathlib import Path
    
    print(f"⚖️  Alpha Recalibration (Diff Mode): Scaling Factor = {args.scale_factor}")
    
    # Force string paths with forward slashes
    base_path = str(Path(args.base_model).resolve()).replace("\\", "/")
    source_path = str(Path(args.source_model).resolve()).replace("\\", "/")
    output_path = str(Path(args.output_dir).resolve()).replace("\\", "/")

    print(f"   CWD: {os.getcwd()}")
    print(f"   Base Path: {base_path} (Exists: {os.path.exists(base_path)})")
    print(f"   Source Path: {source_path} (Exists: {os.path.exists(source_path)})")
    
    # 1. Load Base Model (CPU to save memory initially)
    print(f"   Loading Base Vessel: {base_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path, 
        torch_dtype=torch.float16,
        device_map="cpu",
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)

    # 2. Load Source Model (CPU)
    print(f"   Loading Source Vessel: {source_path}")
    source_model = AutoModelForCausalLM.from_pretrained(
        source_path, 
        torch_dtype=torch.float16,
        device_map="cpu",
        local_files_only=True
    )

    # 3. Calculate Difference and Scale
    print("   ⚗️  Calculating Perturbation and Recalibrating...")
    
    # We only need to adjust the LM Head if that's where the fusion happened.
    # The original script modified `base_model.lm_head.weight`.
    # So we assume only lm_head is different.
    
    with torch.no_grad():
        base_head = base_model.lm_head.weight.data
        source_head = source_model.lm_head.weight.data
        
        # Check if shapes match
        if base_head.shape != source_head.shape:
            raise ValueError(f"Shape mismatch: Base {base_head.shape} vs Source {source_head.shape}")
            
        print("   Calculating Diff (Source - Base)...")
        diff = source_head - base_head
        
        print(f"   Scaling Diff by {args.scale_factor}...")
        scaled_diff = diff * args.scale_factor
        
        print("   Applying to Base...")
        new_head = base_head + scaled_diff
        
        # Update Base Model
        base_model.lm_head.weight.data = new_head

    # 4. Save
    print(f"   💾 Saving Adjusted AEGIS to {args.output_dir}...")
    base_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("   ✅ Recalibration Complete.")
    
    # Cleanup
    del source_model
    del base_model
    gc.collect()

if __name__ == "__main__":
    recalibrate_by_diff()
