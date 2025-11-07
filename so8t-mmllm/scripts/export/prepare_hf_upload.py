#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuggingFace アップロード準備スクリプト
- SO8T技術秘匿（公開ファイルからSO8T言及削除）
- SHA256証明は内部保存（公開しない）
- EZOファインチューニングとして公開
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime


class HuggingFaceUploadPreparer:
    """HF公開準備"""
    
    def __init__(self, model_dir: Path, output_dir: Path):
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 内部証明保存場所（非公開）
        self.proof_dir = Path("internal_proofs")
        self.proof_dir.mkdir(parents=True, exist_ok=True)
    
    def sanitize_config(self):
        """
        config.json からSO8T言及削除
        """
        print("[SANITIZE] Removing SO8T references from config.json...")
        
        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            print("[WARNING] config.json not found")
            return
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # SO8T関連フィールド削除
        removed_fields = []
        
        # _so8t_proof を内部保存してから削除
        if "_so8t_proof" in config:
            proof_data = config.pop("_so8t_proof")
            removed_fields.append("_so8t_proof")
            
            # 内部証明保存
            proof_file = self.proof_dir / f"so8t_proof_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(proof_file, 'w') as f:
                json.dump(proof_data, f, indent=2)
            
            print(f"[PROOF] SO8T proof saved internally: {proof_file}")
        
        # その他SO8T関連フィールド
        so8t_fields = [k for k in config.keys() if 'so8t' in k.lower() or 'pet' in k.lower()]
        for field in so8t_fields:
            config.pop(field)
            removed_fields.append(field)
        
        # クリーンなconfig保存
        clean_config_path = self.output_dir / "config.json"
        with open(clean_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[OK] Sanitized config saved: {clean_config_path}")
        print(f"[REMOVED] Fields: {removed_fields}")
    
    def copy_model_files(self):
        """モデルファイルコピー（SO8T言及なし）"""
        print("\n[COPY] Copying model files...")
        
        # 公開ファイルリスト
        public_files = [
            "pytorch_model.bin",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json"
        ]
        
        for file_name in public_files:
            src = self.model_dir / file_name
            if src.exists():
                dst = self.output_dir / file_name
                shutil.copy2(src, dst)
                print(f"[OK] Copied: {file_name}")
    
    def create_public_readme(self):
        """公開用README作成（SO8T言及なし）"""
        print("\n[CREATE] Public README...")
        
        readme_content = """# EZO Japanese Business Finetuned Model

日本企業向け業務支援に特化したファインチューニングモデル

## 特徴

- ベースモデル: ELYZA Llama-3-JP-8B
- 対象ドメイン: 医療・金融・ビジネス・情報システム
- 日本語最適化済み
- 業務タスク性能向上

## 使用方法

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("YOUR_USERNAME/ezo-japanese-business-ft")
model = AutoModelForCausalLM.from_pretrained("YOUR_USERNAME/ezo-japanese-business-ft")

response = model.generate(...)
```

## ライセンス

Apache 2.0
"""
        
        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"[OK] Public README created: {readme_path}")
    
    def prepare_upload(self):
        """アップロード準備完了"""
        print(f"\n{'='*60}")
        print(f"[PREPARE] HuggingFace Upload Preparation")
        print(f"{'='*60}\n")
        
        self.sanitize_config()
        self.copy_model_files()
        self.create_public_readme()
        
        # アップロード手順書
        upload_guide = self.output_dir / "UPLOAD_GUIDE.txt"
        with open(upload_guide, 'w', encoding='utf-8') as f:
            f.write(f"""HuggingFace Upload Guide
========================

Directory: {self.output_dir}

Upload steps:
1. Create HuggingFace repo: https://huggingface.co/new
2. Clone repo: git clone https://huggingface.co/YOUR_USERNAME/ezo-japanese-business-ft
3. Copy files from {self.output_dir} to cloned repo
4. Commit: git add . && git commit -m "Initial upload"
5. Push: git push
6. Update model card on HuggingFace web interface

IMPORTANT:
- DO NOT upload files in {self.proof_dir}/ (internal proofs)
- DO NOT mention SO8T in public description
- This is EZO finetuning for PoC purposes

Internal proof location: {self.proof_dir}/
Keep these files SECRET for verification only.
""")
        
        print(f"[OK] Upload guide created: {upload_guide}")
        
        print(f"\n{'='*60}")
        print(f"[OK] HuggingFace upload preparation completed!")
        print(f"Public files: {self.output_dir}")
        print(f"Internal proofs: {self.proof_dir} (DO NOT UPLOAD)")
        print(f"{'='*60}\n")


def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, 
                        default=Path("outputs/ezo_stealth_finetuned/final_model"))
    parser.add_argument("--output_dir", type=Path,
                        default=Path("outputs/hf_upload_ready"))
    args = parser.parse_args()
    
    preparer = HuggingFaceUploadPreparer(args.model_dir, args.output_dir)
    preparer.prepare_upload()


if __name__ == "__main__":
    main()




