import os
from pathlib import Path
from collections import defaultdict

def check_datasets():
    base_path = Path("D:/webdataset/datasets")

    if not base_path.exists():
        print("Dataset directory not found")
        return

    datasets = defaultdict(list)

    # Find all json/jsonl files
    for json_file in base_path.rglob("*.json*"):
        if "state" in json_file.name.lower():
            continue

        dataset_name = json_file.parent.name
        size_mb = json_file.stat().st_size / (1024 * 1024)
        size_gb = size_mb / 1024

        datasets[dataset_name].append({
            'file': json_file.name,
            'size_mb': size_mb,
            'size_gb': size_gb,
            'path': str(json_file)
        })

    # Display results
    print("Available Datasets:")
    print("=" * 80)

    total_size_gb = 0
    for dataset_name, files in datasets.items():
        dataset_total_gb = sum(f['size_gb'] for f in files)
        total_size_gb += dataset_total_gb

        print(f"\n{dataset_name}:")
        for file_info in sorted(files, key=lambda x: x['size_mb'], reverse=True):
            print(".1f"
        print(".1f"
    print("\n" + "=" * 80)
    print(".1f"
    print("\nRecommendations for SO8T training:")
    print("- Need at least 1-2GB of diverse, high-quality data")
    print("- Current cleansed data: 12.1MB (insufficient)")
    print("- Should combine multiple datasets for better coverage")

if __name__ == "__main__":
    check_datasets()
