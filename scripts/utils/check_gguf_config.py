#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check GGUF benchmark models configuration"""

import json
from pathlib import Path

config_path = Path("configs/gguf_benchmark_models.json")
config = json.loads(config_path.read_text(encoding="utf-8"))

print(f"Total models: {len(config['gguf_models'])}")
print()
for i, m in enumerate(config['gguf_models'], 1):
    print(f"{i}. {m['alias']}: {m['description']}")
    print(f"   GGUF: {m['gguf_path']}")
    print(f"   Ollama: {m.get('ollama_name', 'N/A')}")
    print()
























