import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import os
import sys

path = "c:/Users/downl/Desktop/SO8T/models/Borea-Phi-3.5-mini-Instruct-Jp"
print(f"Testing load from: {path}")

try:
    print("1. Loading Config...")
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    print("   Config loaded.")
except Exception as e:
    print(f"   FAIL Config: {e}")

try:
    print("2. Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    print("   Tokenizer loaded.")
except Exception as e:
    print(f"   FAIL Tokenizer: {e}")

try:
    print("3. Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, device_map="cpu", torch_dtype=torch.float16)
    print("   Model loaded.")
except Exception as e:
    print(f"   FAIL Model: {e}")
