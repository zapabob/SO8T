#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PEFTのenable_input_require_gradsのインポートパスを確認"""

import peft
print(f"PEFT version: {peft.__version__}")

# 様々なインポートパスを試す
import_paths = [
    "peft.tuners.lora",
    "peft.utils",
    "peft",
]

for path in import_paths:
    try:
        if path == "peft.tuners.lora":
            from peft.tuners.lora import enable_input_require_grads
            print(f"[OK] Found in {path}")
            break
        elif path == "peft.utils":
            from peft.utils import enable_input_require_grads
            print(f"[OK] Found in {path}")
            break
        elif path == "peft":
            from peft import enable_input_require_grads
            print(f"[OK] Found in {path}")
            break
    except ImportError as e:
        print(f"[NG] Not in {path}: {e}")
        continue
else:
    print("[ERROR] enable_input_require_grads not found in any location")
    print("[INFO] Checking available functions in peft.tuners.lora...")
    try:
        import peft.tuners.lora as lora_module
        print(f"Available in peft.tuners.lora: {dir(lora_module)}")
    except Exception as e:
        print(f"Error checking peft.tuners.lora: {e}")
    
    print("[INFO] Checking available functions in peft.utils...")
    try:
        import peft.utils as utils_module
        print(f"Available in peft.utils: {[x for x in dir(utils_module) if 'grad' in x.lower() or 'input' in x.lower()]}")
    except Exception as e:
        print(f"Error checking peft.utils: {e}")

