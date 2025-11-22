import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from datasets import load_dataset

name = "TFMC/imatrix-dataset-for-japanese-llm"
print(f"Checking {name}...")
try:
    ds = load_dataset(name, split="train", streaming=True)
    print(f"✅ Found dataset: {name}")
    print(next(iter(ds)))
except Exception as e:
    print(f"❌ Failed: {e}")
