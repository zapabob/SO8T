#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GGUF to Safetensors Converter for Phi-3.5
Maps GGUF tensor names to Hugging Face Safetensors names.
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict

import torch
from gguf import GGUFReader
from safetensors.torch import save_file
from tqdm import tqdm

# Mapping from GGUF to HF
# Note: Phi-3.5 uses standard Llama-like naming in many places but has specific layer names
PHI3_MAPPING = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
    "blk.{n}.attn_qkv.weight": "model.layers.{n}.self_attn.qkv_proj.weight",
    "blk.{n}.attn_output.weight": "model.layers.{n}.self_attn.o_proj.weight",
    "blk.{n}.ffn_up.weight": "model.layers.{n}.mlp.gate_up_proj.weight", # Phi-3 mlp combines gate/up
    "blk.{n}.ffn_down.weight": "model.layers.{n}.mlp.down_proj.weight",
    "blk.{n}.attn_norm.weight": "model.layers.{n}.input_layernorm.weight",
    "blk.{n}.ffn_norm.weight": "model.layers.{n}.post_attention_layernorm.weight",
}

def convert_gguf_to_st(gguf_path: Path, output_dir: Path, config_dir: Path):
    print(f"Loading GGUF: {gguf_path}")
    reader = GGUFReader(gguf_path)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tensors: Dict[str, torch.Tensor] = {}
    
    for tensor in tqdm(reader.tensors, desc="Converting tensors"):
        name = tensor.name
        data = tensor.data
        
        # Convert numpy to torch
        # GGUF weights are often F16 or BF16.reader.tensors[i].data is a numpy array.
        tensor_data = torch.from_numpy(data.copy())
        
        hf_name = None
        if name in PHI3_MAPPING:
            hf_name = PHI3_MAPPING[name]
        elif "blk." in name:
            parts = name.split(".")
            layer_idx = parts[1]
            suffix = ".".join(parts[2:])
            pattern = f"blk.{{n}}.{suffix}"
            if pattern in PHI3_MAPPING:
                hf_name = PHI3_MAPPING[pattern].format(n=layer_idx)
        
        if hf_name:
            tensors[hf_name] = tensor_data
        else:
            print(f"Warning: No mapping for tensor {name}")

    # Save safetensors
    st_path = output_dir / "model.safetensors"
    print(f"Saving Safetensors to {st_path}")
    save_file(tensors, str(st_path), metadata={"format": "pt"})
    
    # Copy config files
    print(f"Copying config files from {config_dir}")
    for f in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = config_dir / f
        if src.exists():
            shutil.copy(src, output_dir / f)
        else:
            print(f"Warning: {f} not found in {config_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    
    convert_gguf_to_st(args.gguf, args.out, args.config)

if __name__ == "__main__":
    main()
