"""
Quick test to verify AGIASI code structure without loading the full model
"""
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

print("Testing AGIASI Soul Injection code structure...")

# Test 1: Import wrapper class
print("\n1. Testing import of AGIASI_SO8T_Wrapper...")
try:
    from src.models.agiasi_borea import AGIASI_SO8T_Wrapper
    print("   ✅ Import successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Verify class structure
print("\n2. Verifying class methods...")
required_methods = ['forward', 'get_phase_status']
for method in required_methods:
    if hasattr(AGIASI_SO8T_Wrapper, method):
        print(f"   ✅ Method '{method}' found")
    else:
        print(f"   ❌ Method '{method}' missing")

# Test 3: Check training script structure
print("\n3. Testing training script structure...")
try:
    with open(os.path.join(project_root, 'scripts', 'training', 'inject_soul_into_borea.py'), 'r') as f:
        content = f.read()
        checks = [
            ('linear_annealing', 'Annealing function'),
            ('AGIASI_SO8T_Wrapper', 'Wrapper import'),
            ('TARGET_ALPHA', 'Golden ratio constant'),
            ('optimizer', 'Optimizer setup')
        ]
        for check_str, desc in checks:
            if check_str in content:
                print(f"   ✅ {desc} found")
            else:
                print(f"   ⚠️ {desc} not found")
except Exception as e:
    print(f"   ❌ Failed to read training script: {e}")

print("\n🎉 Code structure verification complete!")
print("Note: Full functionality test requires bitsandbytes installation.")
