#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HFモデルチェックスクリプト
"""

from pathlib import Path
import json

def check_hf_model():
    """HFモデルの構造を確認"""
    model_path = Path("H:/from_D/webdataset/models/final/aegis_v21_sft_hf")

    print("Checking HF model files...")
    for file in sorted(model_path.glob("*")):
        size = file.stat().st_size / (1024 * 1024)
        print(f"  {file.name}: {size:.1f} MB")
    print("\nModel architecture check...")
    config_file = model_path / "config.json"
    if config_file.exists():
        with open(config_file, "r") as f:
            config = json.load(f)
        print(f"Architecture: {config.get('architectures', 'unknown')}")
        print(f"Model type: {config.get('model_type', 'unknown')}")

        # auto_mapを確認
        auto_map = config.get('auto_map', {})
        print(f"Auto map: {auto_map}")

    # modelingファイルが必要か確認
    modeling_files = [
        'modeling_phi3.py',
        'configuration_phi3.py'
    ]

    for modeling_file in modeling_files:
        file_path = model_path / modeling_file
        if file_path.exists():
            print(f"Found {modeling_file}")
        else:
            print(f"Missing {modeling_file}")

    # モデル読み込みテスト
    print("\nTesting model loading...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        print("Tokenizer loaded successfully")

        # trust_remote_code=Falseで試す
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False
        )
        print("Model loaded successfully")

    except Exception as e:
        print(f"Model loading failed: {e}")

        # trust_remote_code=Trueで試す
        try:
            print("Trying with trust_remote_code=True...")
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                local_files_only=True,
                trust_remote_code=True
            )
            print("Model loaded successfully with trust_remote_code=True")
        except Exception as e2:
            print(f"Still failed: {e2}")

if __name__ == "__main__":
    check_hf_model()
