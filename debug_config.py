from src.utils.config_loader import ConfigLoader
from src.utils.path_resolver import PathResolver
import json
import logging

logging.basicConfig(level=logging.INFO)
root = PathResolver.get_project_root()
print(f"Project Root: {root}")

try:
    config = ConfigLoader.load_json('training.json', required=True)
    print("Loaded config successfully.")
    print(f"Type: {type(config)}")
    if isinstance(config, dict):
        print(f"Keys: {list(config.keys())}")
        opt = config.get('optimization')
        print(f"Optimization: {opt} (Type: {type(opt)})")
except Exception as e:
    print(f"Error loading config: {e}")
