#!/usr/bin/env python3
"""
Check AEGIS v2.0 generated data format and quality
"""

import json
from pathlib import Path

def check_data_format():
    """Check the format and quality of generated AEGIS data"""

    input_file = Path("D:/webdataset/aegis_v2.0/deep_research_thinking_dataset.jsonl")
    output_file = Path("D:/webdataset/aegis_v2.0/deep_research_thinking_dataset_cleansed.jsonl")

    print("=== AEGIS v2.0 Data Format Check ===")

    # Check input file
    if not input_file.exists():
        print("❌ Input file not found")
        return False

    print(f"✅ Input file exists: {input_file}")
    print(f"   Size: {input_file.stat().st_size:,} bytes")

    # Check first few samples
    samples = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5:  # Check first 5 samples
                    break
                try:
                    data = json.loads(line.strip())
                    samples.append(data)
                    print(f"\nSample {i+1}:")
                    print(f"  Label: {data.get('four_class_label')}")
                    print(f"  Quality Score: {data.get('quality_score')}")
                    print(f"  Has 'chosen': {'chosen' in data}")
                    print(f"  Has 'rejected': {'rejected' in data}")
                    print(f"  Has 'prompt': {'prompt' in data}")
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error on line {i+1}: {e}")
                    return False
    except Exception as e:
        print(f"❌ Error reading input file: {e}")
        return False

    # Check output file
    if output_file.exists():
        size = output_file.stat().st_size
        print(f"\nOutput file exists: {output_file}")
        print(f"Output file size: {size} bytes")

        if size == 0:
            print("❌ Output file is empty - cleansing failed")
            return False
        else:
            print("✅ Output file has content")
    else:
        print("❌ Output file not found")

    # Check data quality
    labels = [s.get('four_class_label') for s in samples if s.get('four_class_label')]
    scores = [s.get('quality_score') for s in samples if s.get('quality_score') is not None]

    print("\n=== Data Quality Summary ===")
    print(f"Total samples checked: {len(samples)}")
    print(f"Labels found: {len(labels)}")
    print(f"Scores found: {len(scores)}")

    if labels:
        unique_labels = set(labels)
        print(f"Unique labels: {unique_labels}")

    if scores:
        avg_score = sum(scores) / len(scores)
        print(".2f")

    return True

if __name__ == "__main__":
    check_data_format()



Check AEGIS v2.0 generated data format and quality
"""

import json
from pathlib import Path

def check_data_format():
    """Check the format and quality of generated AEGIS data"""

    input_file = Path("D:/webdataset/aegis_v2.0/deep_research_thinking_dataset.jsonl")
    output_file = Path("D:/webdataset/aegis_v2.0/deep_research_thinking_dataset_cleansed.jsonl")

    print("=== AEGIS v2.0 Data Format Check ===")

    # Check input file
    if not input_file.exists():
        print("❌ Input file not found")
        return False

    print(f"✅ Input file exists: {input_file}")
    print(f"   Size: {input_file.stat().st_size:,} bytes")

    # Check first few samples
    samples = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5:  # Check first 5 samples
                    break
                try:
                    data = json.loads(line.strip())
                    samples.append(data)
                    print(f"\nSample {i+1}:")
                    print(f"  Label: {data.get('four_class_label')}")
                    print(f"  Quality Score: {data.get('quality_score')}")
                    print(f"  Has 'chosen': {'chosen' in data}")
                    print(f"  Has 'rejected': {'rejected' in data}")
                    print(f"  Has 'prompt': {'prompt' in data}")
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error on line {i+1}: {e}")
                    return False
    except Exception as e:
        print(f"❌ Error reading input file: {e}")
        return False

    # Check output file
    if output_file.exists():
        size = output_file.stat().st_size
        print(f"\nOutput file exists: {output_file}")
        print(f"Output file size: {size} bytes")

        if size == 0:
            print("❌ Output file is empty - cleansing failed")
            return False
        else:
            print("✅ Output file has content")
    else:
        print("❌ Output file not found")

    # Check data quality
    labels = [s.get('four_class_label') for s in samples if s.get('four_class_label')]
    scores = [s.get('quality_score') for s in samples if s.get('quality_score') is not None]

    print("\n=== Data Quality Summary ===")
    print(f"Total samples checked: {len(samples)}")
    print(f"Labels found: {len(labels)}")
    print(f"Scores found: {len(scores)}")

    if labels:
        unique_labels = set(labels)
        print(f"Unique labels: {unique_labels}")

    if scores:
        avg_score = sum(scores) / len(scores)
        print(".2f")

    return True

if __name__ == "__main__":
    check_data_format()
