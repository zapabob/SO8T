#!/usr/bin/env python3
"""
Import verification test script
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")

try:
    print("Testing imports...")

    # Basic modules
    import torch
    print(f"[OK] PyTorch: {torch.__version__}")

    import numpy as np
    print(f"[OK] NumPy: {np.__version__}")

    # Project core modules
    from src.utils.checkpoint_manager import RollingCheckpointManager, EmergencyCheckpointManager
    print("[OK] Checkpoint manager imports")

    from src.utils.vssi_template import render_thinking, normalize_prompt_text
    print("[OK] VSSI template imports")

    from src.utils.path_resolver import PathResolver
    print("[OK] Path resolver imports")

    from src.utils.config_loader import ConfigLoader
    print("[OK] Config loader imports")

    # SO8T core modules
    try:
        from src.core.so8t_core.triality_heads import TrialityHead
        print("[OK] Triality heads imports")
    except ImportError as e:
        print(f"[SKIP] Triality heads: {e}")

    try:
        from src.core.so8t_core.self_verification import SelfVerifier
        print("[OK] Self verification imports")
    except ImportError as e:
        print(f"[SKIP] Self verification: {e}")

    # Model modules
    try:
        from src.core.models.so8t_residual_adapter import SO8ResidualAdapter
        print("[OK] SO8T residual adapter imports")
    except ImportError as e:
        print(f"[SKIP] SO8T adapter: {e}")

    print("\nAll critical imports successful.")

except Exception as e:
    print(f"[NG] Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
