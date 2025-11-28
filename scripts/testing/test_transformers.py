import os
import sys

# Try to set env var to fix protobuf issues
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    import google.protobuf
    print(f"Protobuf version: {google.protobuf.__version__}")
except ImportError:
    print("Protobuf not found")

try:
    from transformers import AutoTokenizer
    print("Transformers imported successfully")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print("Tokenizer loaded successfully")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
