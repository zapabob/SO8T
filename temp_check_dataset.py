import json
from pathlib import Path

# サンプルデータを確認
file_path = Path('D:/webdataset/aegis_v2.0/deep_research_thinking_dataset_cleansed.jsonl')
count = 0
labels = {}

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        if count >= 5:  # 最初の5サンプルのみ
            break
        try:
            sample = json.loads(line.strip())
            label = sample.get('four_class_label', 'NO_LABEL')
            if label not in labels:
                labels[label] = 0
            labels[label] += 1
            if count < 3:  # 最初の3サンプルを表示
                print(f'Sample {count+1}:')
                print(f'  Label: {label}')
                text_preview = sample.get('text', '')[:200]
                print(f'  Text preview: {text_preview}...')
                print(f'  Quality score: {sample.get("quality_score", "N/A")}')
                print()
            count += 1
        except json.JSONDecodeError:
            continue

print(f'Total samples checked: {count}')
print(f'Label distribution in first 5: {labels}')
