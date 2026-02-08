#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データセット分析スクリプト
既存のデータセット構造を分析し、ドメインとスコアの分布を確認する
"""

import json
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_dataset():
    """データセットの構造を分析"""
    dataset_path = Path("data/so8t_cot_enhanced_training_dataset_70k.jsonl")

    if not dataset_path.exists():
        print(f"データセットファイルが見つかりません: {dataset_path}")
        return

    print("データセット分析を開始します...")
    print(f"ファイル: {dataset_path}")

    # データ読み込み（最初の1000件で分析）
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 1000:  # 最初の1000件のみ分析
                break
            if line.strip():
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"JSON解析エラー (line {i+1}): {e}")
                    continue

    print(f"分析対象件数: {len(data)}")

    # phi35_labelの分布分析
    labels = []
    scores = []
    categories = []

    for item in data:
        # phi35_labelの取得
        phi35_label = item.get('metadata', {}).get('phi35_label', 'unknown')
        labels.append(phi35_label)

        # quality_scoreの取得
        quality_score = item.get('quality_score', 0.0)
        scores.append(quality_score)

        # categoryの取得
        category = item.get('category', 'unknown')
        categories.append(category)

    # 分布の表示
    print("\n=== phi35_label 分布 ===")
    label_counts = Counter(labels)
    for label, count in sorted(label_counts.items()):
        print(f"{label}: {count}件 ({count/len(data)*100:.1f}%)")

    print("\n=== quality_score 分布 ===")
    print(f"平均: {sum(scores)/len(scores):.2f}")
    print(f"中央値: {sorted(scores)[len(scores)//2]:.2f}")
    print(f"最小: {min(scores):.2f}")
    print(f"最大: {max(scores):.2f}")
    # ドメインのマッピング案を表示
    print("\n=== ドメイン マッピング案 ===")
    print("現在のphi35_label -> 要求ドメイン:")
    print("- 'math' -> math (直接マッチ)")
    print("- 'physics' -> physics (直接マッチ)")
    print("- 'reasoning' -> reasoning (直接マッチ)")
    print("- 'general' -> reasoning (汎用的な推論として)")
    print("- 'safety' -> reasoning (安全推論として)")
    print("- その他 -> reasoning (デフォルト)")

    # スコア分布のヒストグラム
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, alpha=0.7, edgecolor='black')
    plt.title('Quality Score Distribution')
    plt.xlabel('Quality Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('data_quality_score_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\nスコア分布グラフを保存しました: data_quality_score_distribution.png")

if __name__ == "__main__":
    analyze_dataset()
