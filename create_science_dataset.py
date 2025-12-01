#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
科学推論データセット作成スクリプト
既存のデータセットを基に、domainとscoreフィールドを追加したデータセットを作成
"""

import json
import random
from pathlib import Path
from tqdm import tqdm

def create_science_dataset():
    """科学推論データセットを作成"""
    input_path = Path("data/so8t_cot_enhanced_training_dataset_70k.jsonl")
    output_path = Path("data/science_reasoning_dataset.jsonl")

    if not input_path.exists():
        print(f"入力ファイルが見つかりません: {input_path}")
        return

    print("科学推論データセットを作成します...")
    print(f"入力: {input_path}")
    print(f"出力: {output_path}")

    # ドメインの割合設定
    domain_ratios = {
        'math': 0.3,      # 数学: 30%
        'physics': 0.3,   # 物理: 30%
        'reasoning': 0.4  # 一般推論: 40%
    }

    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="データ読み込み"):
            if line.strip():
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError:
                    continue

    print(f"総データ数: {len(data)}")

    # domainとscoreフィールドを追加
    processed_data = []
    for item in tqdm(data, desc="データ処理"):

        # phi35_labelに基づいてdomainを決定
        phi35_label = item.get('metadata', {}).get('phi35_label', 'general')

        if phi35_label == 'math':
            domain = 'math'
        elif phi35_label == 'physics':
            domain = 'physics'
        else:
            # general, safetyなどは確率的に割り当て
            rand = random.random()
            if rand < domain_ratios['math']:
                domain = 'math'
            elif rand < domain_ratios['math'] + domain_ratios['physics']:
                domain = 'physics'
            else:
                domain = 'reasoning'

        # quality_scoreをscoreとして使用（0.0-1.0の範囲）
        # より現実的なスコア分布にするため、調整
        base_score = item.get('quality_score', 0.0)

        # domainに応じてスコアを調整
        if domain == 'math':
            # 数学は中間〜高難易度
            score = max(0.3, min(0.9, base_score + random.uniform(0.2, 0.6)))
        elif domain == 'physics':
            # 物理は高難易度
            score = max(0.5, min(0.95, base_score + random.uniform(0.4, 0.7)))
        else:  # reasoning
            # 一般推論は低〜中難易度
            score = max(0.1, min(0.7, base_score + random.uniform(0.0, 0.4)))

        # 小数点第2位で丸める
        score = round(score, 2)

        # 新しいアイテムを作成
        new_item = {
            'id': item.get('id', f'generated_{len(processed_data)}'),
            'text': item.get('text', ''),
            'domain': domain,
            'score': score,
            'language': item.get('language', 'en'),
            'source_dataset': item.get('source_dataset', 'generated'),
            'metadata': item.get('metadata', {})
        }

        processed_data.append(new_item)

    # 出力
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(processed_data, desc="ファイル書き込み"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"作成完了: {len(processed_data)}件")
    print(f"出力ファイル: {output_path}")

    # 統計表示
    domains = [item['domain'] for item in processed_data]
    scores = [item['score'] for item in processed_data]

    from collections import Counter
    domain_counts = Counter(domains)

    print("\n=== 作成データ統計 ===")
    print("ドメイン分布:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count}件 ({count/len(processed_data)*100:.1f}%)")

    print("\nスコア分布:")
    print(f"  平均: {sum(scores)/len(scores):.2f}")
    print(f"  中央値: {sorted(scores)[len(scores)//2]:.2f}")
    print(f"  最小: {min(scores):.2f}")
    print(f"  最大: {max(scores):.2f}")

if __name__ == "__main__":
    create_science_dataset()
