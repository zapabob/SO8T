import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from transformers import AutoModelForCausalLM, AutoConfig

model_id = "microsoft/Phi-3.5-mini-instruct"
print(f"Inspecting {model_id}...")

try:
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    # We don't need to load the full weights to see the structure, usually config + dummy init is enough, 
    # but let's try loading with meta device or just print config to see layer names if possible.
    # Actually, loading with trust_remote_code=True might require actual code execution.
    # Let's just load the model on CPU (it's small enough, ~3.8B params, might be tight on 12GB if loaded fully, 
    # but we can load with torch_dtype=torch.float16 or device_map='auto').
    
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
