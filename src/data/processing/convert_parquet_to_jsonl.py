import pandas as pd
import json
from pathlib import Path

def convert_parquet_to_jsonl(input_file: str, output_file: str):
    print(f"Reading {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} rows.")
    
    # We only need instruction/input/output or messages
    # AymanTarig/function-calling-v0.2-with-r1-cot usually has 'messages'
    
    with open(output_file, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    
    print(f"Converted to {output_file}")

if __name__ == "__main__":
    convert_parquet_to_jsonl(
        "c:/Users/downl/Desktop/SO8T/data/hf_datasets/R1-CoT/data/train-00000-of-00001.parquet",
        "c:/Users/downl/Desktop/SO8T/data/hf_datasets/R1-CoT/r1_cot_train.jsonl"
    )
