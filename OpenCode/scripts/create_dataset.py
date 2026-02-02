# create_dataset.py
"""Split processed JSONL into train/val/test.

Usage:
    python create_dataset.py --input D:\\dataset\\processed.jsonl --output-dir D:\\dataset\\final
"""
import argparse
from pathlib import Path
import json
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    random.shuffle(lines)
    
    total = len(lines)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)
    
    train_data = lines[:train_end]
    val_data = lines[train_end:val_end]
    test_data = lines[val_end:]
    
    (args.output_dir / "train.jsonl").write_text("".join(train_data), encoding="utf-8")
    (args.output_dir / "val.jsonl").write_text("".join(val_data), encoding="utf-8")
    (args.output_dir / "test.jsonl").write_text("".join(test_data), encoding="utf-8")
    
    print(f"Created dataset: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

if __name__ == "__main__":
    main()
