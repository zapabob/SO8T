#!/usr/bin/env python3
"""Check enable_input_require_grads import path in PEFT 0.18.0"""

import sys

print(f"Python version: {sys.version}")
print(f"Checking PEFT version and enable_input_require_grads import path...")

try:
    import peft
    print(f"PEFT version: {peft.__version__}")
except Exception as e:
    print(f"ERROR: Failed to import peft: {e}")
    sys.exit(1)

# Try different import paths
import_paths = [
    "peft.tuners.lora",
    "peft.utils",
    "peft",
    "peft.tuners",
]

for import_path in import_paths:
    try:
        if import_path == "peft.tuners.lora":
            from peft.tuners.lora import enable_input_require_grads
            print(f"[OK] Found enable_input_require_grads in {import_path}")
            break
        elif import_path == "peft.utils":
            from peft.utils import enable_input_require_grads
            print(f"[OK] Found enable_input_require_grads in {import_path}")
            break
        elif import_path == "peft":
            from peft import enable_input_require_grads
            print(f"[OK] Found enable_input_require_grads in {import_path}")
            break
        elif import_path == "peft.tuners":
            from peft.tuners import enable_input_require_grads
            print(f"[OK] Found enable_input_require_grads in {import_path}")
            break
    except ImportError as e:
        print(f"[SKIP] {import_path}: {e}")
        continue
    except Exception as e:
        print(f"[ERROR] {import_path}: {e}")
        continue
else:
    print("[ERROR] enable_input_require_grads not found in any import path")
    print("\nTrying to find it manually...")
    
    # Try to find it in peft module
    try:
        import peft.tuners.lora as lora_module
        if hasattr(lora_module, 'enable_input_require_grads'):
            print("[OK] Found enable_input_require_grads in peft.tuners.lora (direct access)")
        else:
            print("[SKIP] enable_input_require_grads not in peft.tuners.lora")
    except Exception as e:
        print(f"[ERROR] Failed to check peft.tuners.lora: {e}")
    
    try:
        import peft.utils as utils_module
        if hasattr(utils_module, 'enable_input_require_grads'):
            print("[OK] Found enable_input_require_grads in peft.utils (direct access)")
        else:
            print("[SKIP] enable_input_require_grads not in peft.utils")
    except Exception as e:
        print(f"[ERROR] Failed to check peft.utils: {e}")

