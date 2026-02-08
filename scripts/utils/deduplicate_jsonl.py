import json
import argparse
from pathlib import Path
from typing import Set

def deduplicate_jsonl(input_file: Path, output_file: Path, id_field: str = "metadata.paper_id"):
    """Deduplicates a JSONL file keeping only the first occurrence of each ID."""
    seen_ids: Set[str] = set()
    deduplicated_records = []
    
    if not input_file.exists():
        print(f"File not found: {input_file}")
        return

    print(f"Deduplicating {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                
                # Extract ID based on dot notation
                parts = id_field.split('.')
                current = data
                p_id = None
                for part in parts:
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                        p_id = current
                    else:
                        p_id = None
                        break
                
                # Fallback for generic 'id' if specified dot notation fails
                if p_id is None and 'id' in data:
                    p_id = data['id']
                
                if p_id and p_id in seen_ids:
                    continue
                
                if p_id:
                    seen_ids.add(p_id)
                
                deduplicated_records.append(data)
            except Exception as e:
                print(f"Error parsing line: {e}")

    with open(output_file, 'w', encoding='utf-8') as f:
        for record in deduplicated_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print(f"Finished. Original: {len(deduplicated_records) + len(seen_ids) - len(seen_ids)}? No, wait.")
    print(f"Unique records: {len(deduplicated_records)}")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate JSONL files")
    parser.add_argument("input", type=str, help="Input JSONL file")
    parser.add_argument("--output", type=str, help="Output JSONL file (defaults to input_dedup.jsonl)")
    parser.add_argument("--id-field", type=str, default="metadata.paper_id", help="Field to use for ID (dot notation)")
    
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_dedup{input_path.suffix}")
    
    deduplicate_jsonl(input_path, output_path, args.id_field)
