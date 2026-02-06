import sys
import os
from pathlib import Path

# Insert CWD to path to simulate running from root
sys.path.insert(0, os.getcwd())

print(f"CWD: {os.getcwd()}")
print(f"Sys Path[0]: {sys.path[0]}")

try:
    import src.training.train_unsloth_so8t
    print("SUCCESS: Import worked")
except Exception as e:
    print(f"FAILURE: {e}")
    import traceback
    print(traceback.format_exc())
