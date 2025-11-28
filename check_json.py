import json
from pathlib import Path

gsm8k_path = Path(r'D:\webdataset\benchmark_datasets\gsm8k\gsm8k_full.json')
print('Checking GSM8K file...')

# ファイルの存在確認
if gsm8k_path.exists():
    print(f'File exists: {gsm8k_path}')
    print(f'File size: {gsm8k_path.stat().st_size} bytes')

    # 先頭部分を確認
    with open(gsm8k_path, 'r', encoding='utf-8') as f:
        content = f.read(500)
        print('First 500 chars:')
        print(repr(content[:200]))

    # JSONとして読み込み
    try:
        with open(gsm8k_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f'Successfully loaded {len(data)} records')
            if data:
                print('First record keys:', list(data[0].keys()))
    except Exception as e:
        print(f'JSON load failed: {e}')
        import traceback
        traceback.print_exc()
else:
    print('File does not exist')























