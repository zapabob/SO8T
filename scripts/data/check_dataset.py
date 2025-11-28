import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from datasets import load_dataset

dataset_names = [
    "Borea/Phi-3.5-instinct-jp",
    "Borea/Phi-3.5-mini-Instruct-Jp",
    "Borea/Phi3.5-instinct-jp",
    "mmnga/HODACHI-Borea-Phi-3.5-mini-Instruct-Jp-gguf" # Unlikely a dataset
]

for name in dataset_names:
    print(f"Checking {name}...")
    try:
        ds = load_dataset(name, split="train", streaming=True)
        print(f"✅ Found dataset: {name}")
        # Try to read one item
        print(next(iter(ds)))
        break
    except Exception as e:
        print(f"❌ Failed: {e}")
