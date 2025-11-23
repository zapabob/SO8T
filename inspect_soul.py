import torch
import sys

try:
    data = torch.load(sys.argv[1], map_location="cpu")
    print("---KEYS_START---")
    if isinstance(data, dict):
        for k, v in data.items():
            print(f"KEY: {k}, TYPE: {type(v)}")
            if k == "rotation":
                if isinstance(v, dict):
                    print("  ROTATION KEYS:", v.keys())
                elif isinstance(v, torch.Tensor):
                    print("  ROTATION TENSOR SHAPE:", v.shape)
    else:
        print("NOT_A_DICT")
    print("---KEYS_END---")
except Exception as e:
    print(f"ERROR: {e}")
