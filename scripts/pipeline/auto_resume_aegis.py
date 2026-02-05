#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS-v3.0 Power-on Auto-Resume Entry Point
Checks for latest rolling checkpoint and continues execution.
"""

import sys
import os
from pathlib import Path

# Fix path to resolve project root
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.infrastructure.pipeline.integrated_moonshot_pipeline_2025_2026 import IntegratedMoonshotPipeline2025_2026

def main():
    print("="*60)
    print("      AEGIS-v3.0 CONTINUOUS OPERATION SYSTEM (SO8T/Sakana)")
    print("="*60)
    print("[INIT] Searching for rolling checkpoints...")
    
    # Instantiate the pipeline
    pipeline = IntegratedMoonshotPipeline2025_2026()
    
    # The execute_full_pipeline method already contains auto-resume logic
    try:
        pipeline.execute_full_pipeline(use_existing_datasets=True)
    except KeyboardInterrupt:
        print("\n[STOP] Pipeline paused by user. Process state saved.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Pipeline crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
