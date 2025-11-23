# preprocess.py
"""Preprocess raw text files into a JSONL dataset with 4 classes.

Usage:
    python preprocess.py --input-dir D:\\dataset --output D:\\dataset\\processed.jsonl
"""
import argparse
from pathlib import Path
import json
import re

def clean_text(text: str) -> str:
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def classify_content(filename: str, content: str) -> str:
    # Heuristic classification based on filename prefixes (from Wikipedia downloader) or keywords
    name = filename.lower()
    text = content.lower()
    
    # 1. Critical Systems
    if any(x in name for x in ["military", "aerospace", "transport", "utility", "security", "business", "infrastructure"]):
        return "Critical Systems"
        
    # 2. Safety & Law
    if any(x in name for x in ["pharmacology", "drug", "sexuality", "law", "gov", "nsfw"]):
        return "Safety & Law"
    if "drug" in text or "compliance" in text:
        return "Safety & Law"

    # 3. Academic Knowledge
    if any(x in name for x in ["physics", "math", "linguistics", "grammar", "history", "arxiv"]):
        return "Academic Knowledge"
    if "theorem" in text or "equation" in text:
        return "Academic Knowledge"

    # 4. General/Other
    return "General/Other"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    
    data = []
    
    # Walk through all text files in subdirectories
    for txt_file in args.input_dir.rglob("*.txt"):
        try:
            content = txt_file.read_text(encoding="utf-8")
            if not content.strip():
                continue
                
            cleaned = clean_text(content)
            label = classify_content(txt_file.name, cleaned)
            
            data.append({
                "text": cleaned,
                "label": label,
                "source": txt_file.name
            })
        except Exception as e:
            print(f"Skipping {txt_file}: {e}")
            
    with open(args.output, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"Processed {len(data)} items to {args.output}")

if __name__ == "__main__":
    main()
