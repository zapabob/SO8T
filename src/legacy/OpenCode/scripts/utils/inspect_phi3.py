import os
# Set env var BEFORE importing anything else
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from transformers import AutoModelForCausalLM, AutoConfig

model_id = "microsoft/Phi-3.5-mini-instruct"
print(f"Inspecting {model_id}...")

try:
    # Just printing modules from a small instantiated model
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        torch_dtype="auto", 
        device_map="auto"
    )
    print(model)
    
    print("\n--- Module Names ---")
    for name, module in model.named_modules():
        if "mlp" in name and "proj" in name:
            print(name)
            # Just print the first few to confirm structure
            break
            
except Exception as e:
    print(f"Error: {e}")
