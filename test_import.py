import sys
import os

# Add project root to path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

print(f"Project Root: {project_root}")
print(f"Sys Path: {sys.path}")

try:
    from src.models.nkat_so8t import NKAT_SO8T_ThinkingModel
    print("✅ Import successful!")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
