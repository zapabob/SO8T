#!/usr/bin/env python3

import sys
sys.path.append('scripts/training')

try:
    from mass_gap_monitor import MassGapMonitor, SO8TMassGapCallback
    print("Import successful")

    # Test callback instantiation
    monitor = MassGapMonitor()
    callback = SO8TMassGapCallback(monitor)
    print("Callback instantiation successful")

    # Check if methods exist
    methods = [m for m in dir(callback) if not m.startswith('_')]
    print(f"Available methods: {methods}")

    # Check if on_init_end exists
    if hasattr(callback, 'on_init_end'):
        print("on_init_end method exists ✓")
    else:
        print("on_init_end method missing ✗")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
