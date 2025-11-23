# verify_dataset.py
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    
    files = ["train.jsonl", "val.jsonl", "test.jsonl"]
    valid_labels = {"Critical Systems", "Safety & Law", "Academic Knowledge", "General/Other"}
    
    for fname in files:
        fpath = args.data_dir / fname
        if not fpath.exists():
            print(f"Missing {fname}")
            continue
            
        print(f"Checking {fname}...")
        with open(fpath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                try:
                    item = json.loads(line)
                    if not item.get("text"):
                        print(f"  Line {i}: Empty text")
                    if item.get("label") not in valid_labels:
                        print(f"  Line {i}: Invalid label '{item.get('label')}'")
                except json.JSONDecodeError:
                    print(f"  Line {i}: Invalid JSON")
        print(f"Finished checking {fname}")

if __name__ == "__main__":
    main()
