import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from transformers import AutoTokenizer

model_ids = [
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-3-mini-4k-instruct"
]

for mid in model_ids:
    print(f"Checking {mid}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
        print(f"✅ Found tokenizer: {mid}")
        print(f"   Vocab size: {len(tokenizer)}")
        break
    except Exception as e:
        print(f"❌ Failed: {e}")
