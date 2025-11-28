import json

def check_integrated_dataset():
    with open('D:/webdataset/integrated_dataset.jsonl', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3: break
            data = json.loads(line.strip())
            print(f'Sample {i+1}:')
            print(f'  Keys: {list(data.keys())}')
            print(f'  Dataset: {data.get("dataset", "N/A")}')
            print(f'  Language: {data.get("language", "N/A")}')
            if 'text' in data:
                text_preview = data['text'][:200] + '...' if len(data['text']) > 200 else data['text']
                print(f'  Text preview: {text_preview}')
            print()

if __name__ == "__main__":
    check_integrated_dataset()
