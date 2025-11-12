#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Webスクレイピング状況確認スクリプト"""

from pathlib import Path
import json
from datetime import datetime
from collections import Counter

def main():
    base_dir = Path('D:/webdataset/processed')
    
    print("="*80)
    print("Webスクレイピング状況")
    print("="*80)
    
    # 並列インスタンスを検索
    instances = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('parallel_instance_')]
    print(f"\n並列インスタンス数: {len(instances)}")
    
    total_samples = 0
    latest_time = None
    instance_stats = []
    
    for inst in sorted(instances):
        jsonl_files = list(inst.glob('*.jsonl'))
        if jsonl_files:
            latest_file = max(jsonl_files, key=lambda x: x.stat().st_mtime)
            with open(latest_file, 'r', encoding='utf-8') as f:
                count = sum(1 for line in f if line.strip())
            total_samples += count
            mtime = datetime.fromtimestamp(latest_file.stat().st_mtime)
            if latest_time is None or mtime > latest_time:
                latest_time = mtime
            instance_stats.append({
                'name': inst.name,
                'samples': count,
                'last_update': mtime
            })
            print(f"{inst.name}: {count}サンプル (最終更新: {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
    
    print(f"\n総サンプル数: {total_samples}")
    if latest_time:
        elapsed_hours = (datetime.now() - latest_time).total_seconds() / 3600
        print(f"最新更新: {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"経過時間: {elapsed_hours:.1f}時間前")
        
        if elapsed_hours > 2:
            print(f"\n[警告] 最新更新から{elapsed_hours:.1f}時間経過しています。スクレイピングが停止している可能性があります。")
    
    # カテゴリ分布を確認
    print("\n" + "="*80)
    print("カテゴリ分布（最新データ）")
    print("="*80)
    
    categories = Counter()
    languages = Counter()
    
    for inst in sorted(instances):
        jsonl_files = list(inst.glob('*.jsonl'))
        if jsonl_files:
            latest_file = max(jsonl_files, key=lambda x: x.stat().st_mtime)
            with open(latest_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            sample = json.loads(line)
                            category = sample.get('category', 'unknown')
                            language = sample.get('language', 'unknown')
                            categories[category] += 1
                            languages[language] += 1
                        except:
                            pass
    
    print("\nカテゴリ分布（上位10）:")
    for cat, count in categories.most_common(10):
        print(f"  {cat}: {count}")
    
    print("\n言語分布:")
    for lang, count in sorted(languages.items()):
        print(f"  {lang}: {count}")

if __name__ == "__main__":
    main()

















