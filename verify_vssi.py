import sys
import os
sys.path.insert(0, os.getcwd())

try:
    print("Attempting to import src.utils.vssi_template...")
    import src.utils.vssi_template
    print("SUCCESS: src.utils.vssi_template imported")
    
    from src.utils.vssi_template import normalize_prompt_text
    print("SUCCESS: normalize_prompt_text imported")

except Exception as e:
    print(f"FAILURE: {e}")
    import traceback
    print(traceback.format_exc())
