import json
from pathlib import Path
from collections import Counter

def analyze_dataset():
    file_path = Path('D:/webdataset/aegis_v2.0/deep_research_thinking_dataset_cleansed.jsonl')
    sample_count = 0
    labels = Counter()
    total_prompt_len = 0
    total_chosen_len = 0
    total_rejected_len = 0

    print(f"Analyzing dataset: {file_path}")
    print(f"File size: {file_path.stat().st_size:,} bytes ({file_path.stat().st_size / (1024*1024):.1f} MB)")
    print()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                label = data.get('four_class_label', 'UNKNOWN')
                labels[label] += 1

                prompt_len = len(data.get('prompt', ''))
                chosen_len = len(data.get('chosen', ''))
                rejected_len = len(data.get('rejected', ''))

                total_prompt_len += prompt_len
                total_chosen_len += chosen_len
                total_rejected_len += rejected_len

                sample_count += 1

                if sample_count <= 3:  # 最初の3つだけ詳細表示
                    print(f'Sample {sample_count}:')
                    print(f'  Label: {label}')
                    print(f'  Quality: {data.get("quality_score", "N/A")}')
                    print(f'  Prompt length: {prompt_len}')
                    print(f'  Chosen length: {chosen_len}')
                    print(f'  Rejected length: {rejected_len}')
                    print(f'  Source: {data.get("source", "N/A")}')
                    print()

            except json.JSONDecodeError as e:
                print(f'JSON decode error at line {sample_count + 1}: {e}')
                break

    print("=== ANALYSIS RESULTS ===")
    print(f'Total samples: {sample_count}')
    print(f'Label distribution: {dict(labels)}')
    if sample_count > 0:
        print(f'Average prompt length: {total_prompt_len / sample_count:.0f} chars')
        print(f'Average chosen length: {total_chosen_len / sample_count:.0f} chars')
        print(f'Average rejected length: {total_rejected_len / sample_count:.0f} chars')

    # SO8T学習に必要なデータ量の評価
    print("\n=== SO8T TRAINING SUFFICIENCY ANALYSIS ===")
    min_samples = 10000  # SO(8)学習の最低限
    recommended_samples = 50000  # 推奨

    if sample_count < min_samples:
        print(f'[CRITICAL] Insufficient data: {sample_count} < {min_samples} minimum samples')
        print('SO(8) rotation gates and PET regularization require much more diverse data')
    elif sample_count < recommended_samples:
        print(f'[WARNING] Limited data: {sample_count} < {recommended_samples} recommended samples')
        print('May be sufficient for basic training but limited generalization')
    else:
        print(f'[OK] Sufficient data: {sample_count} >= {recommended_samples} samples')

if __name__ == "__main__":
    analyze_dataset()
