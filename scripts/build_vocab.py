#!/usr/bin/env python3
"""
Build Vocabulary from Training Data
訓練データから語彙を構築するスクリプト
"""

import argparse
import json
from pathlib import Path
from shared.data import build_vocab_from_files
from shared.vocab import Vocabulary


def main():
    parser = argparse.ArgumentParser(description="Build vocabulary from training data")
    parser.add_argument("--data_dir", type=Path, default=Path("data"),
                       help="Data directory")
    parser.add_argument("--output_file", type=Path, default=Path("data/vocab.json"),
                       help="Output vocabulary file")
    parser.add_argument("--min_freq", type=int, default=1,
                       help="Minimum frequency for tokens")
    
    args = parser.parse_args()
    
    print("Building vocabulary from training data...")
    print(f"   Data directory: {args.data_dir}")
    print(f"   Output file: {args.output_file}")
    print(f"   Min frequency: {args.min_freq}")
    
    # データファイルのパス
    data_files = [
        args.data_dir / "train.jsonl",
        args.data_dir / "val.jsonl", 
        args.data_dir / "test.jsonl"
    ]
    
    # 存在確認
    missing_files = [f for f in data_files if not f.exists()]
    if missing_files:
        print(f"ERROR: Missing data files: {missing_files}")
        return 1
    
    try:
        # 語彙を構築
        vocab = build_vocab_from_files(data_files, min_freq=args.min_freq)
        
        # 語彙を保存
        args.output_file.parent.mkdir(exist_ok=True)
        vocab.to_file(args.output_file)
        
        print(f"SUCCESS: Vocabulary built successfully!")
        print(f"   Total tokens: {len(vocab)}")
        print(f"   Saved to: {args.output_file}")
        
        # 語彙の統計を表示
        print(f"\nVocabulary Statistics:")
        all_tokens = vocab._itos
        special_tokens = [t for t in all_tokens if t.startswith('<')]
        regular_tokens = [t for t in all_tokens if not t.startswith('<')]
        print(f"   Special tokens: {len(special_tokens)}")
        print(f"   Regular tokens: {len(regular_tokens)}")
        
        # 最初の10個のトークンを表示
        print(f"\nSample tokens:")
        for i, token in enumerate(all_tokens[:10]):
            print(f"   {i}: {token}")
        if len(all_tokens) > 10:
            print(f"   ... and {len(all_tokens) - 10} more")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: Failed to build vocabulary: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
