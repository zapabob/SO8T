#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLMトレーニング用データセット分割スクリプト
SFT (Supervised Fine-Tuning) と PPO (Proximal Policy Optimization) 用に層化抽出で分割

要件:
- SFT: 20% (約14,000件) - 基礎知識とフォーマットの学習
- PPO: 80% (約56,000件) - 深さと正確性の強化学習
- Test/Eval: 2% - 評価用データ

分割ロジック:
1. 全体の2%を評価用としてランダム確保
2. 残りのデータをscoreでソートし、上位40%を難問としてPPOに割り当て
3. 残りのデータをSFT:PPO = 1:2の割合で層化分割
4. 最終的に全体がSFT:PPO ≈ 1:4になるよう調整
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

def load_dataset(file_path):
    """データセットを読み込み"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"JSON解析エラー: {e}")
                    continue
    return data

def save_dataset(data, file_path):
    """データセットを保存"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def analyze_distribution(data, title="データ分布"):
    """データの分布を分析"""
    domains = [item['domain'] for item in data]
    scores = [item['score'] for item in data]

    domain_counts = Counter(domains)

    print(f"\n=== {title} ===")
    print(f"総件数: {len(data)}")
    print("ドメイン内訳:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count}件 ({count/len(data)*100:.1f}%)")

    print("スコア統計:")
    print(f"  平均: {np.mean(scores):.3f}")
    print(f"  中央値: {np.median(scores):.3f}")
    print(f"  最小: {np.min(scores):.3f}")
    print(f"  最大: {np.max(scores):.3f}")

    return domain_counts, scores

def stratified_split_by_domain_and_score(data, test_size=0.02):
    """
    ドメインとスコアを考慮した層化抽出でテストデータを分割

    Args:
        data: データセット
        test_size: テストデータの割合

    Returns:
        train_data, test_data
    """
    print(f"\nテストデータ確保: {test_size*100:.1f}% ({int(len(data)*test_size)}件)")

    # ドメインに基づいて層化
    domains = [item['domain'] for item in data]

    # StratifiedShuffleSplitで層化抽出
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)

    # ドメインをstrataとして使用
    indices = list(range(len(data)))
    train_indices, test_indices = next(sss.split(indices, domains))

    train_data = [data[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]

    print(f"学習データ: {len(train_data)}件")
    print(f"テストデータ: {len(test_data)}件")

    return train_data, test_data

def split_sft_ppo_by_complexity(train_data, sft_ratio=0.20, ppo_ratio=0.80):
    """
    複雑度スコアに基づいてSFTとPPOを分割

    Args:
        train_data: 学習データ
        sft_ratio: SFTデータの目標割合
        ppo_ratio: PPOデータの目標割合

    Returns:
        sft_data, ppo_data
    """
    print(f"\nSFT/PPO分割開始")
    print(f"目標割合 - SFT: {sft_ratio*100:.1f}%, PPO: {ppo_ratio*100:.1f}%")

    # スコアでソート（降順: 高スコア=難問）
    sorted_data = sorted(train_data, key=lambda x: x['score'], reverse=True)

    total_count = len(sorted_data)
    sft_target = int(total_count * sft_ratio)
    ppo_target = int(total_count * ppo_ratio)

    print(f"目標件数 - SFT: {sft_target}件, PPO: {ppo_target}件")

    # 上位40%の難問をPPOに強制割り当て
    hard_threshold = int(total_count * 0.4)
    hard_questions = sorted_data[:hard_threshold]

    print(f"難問閾値: 上位{hard_threshold}件 (スコア >= {sorted_data[hard_threshold]['score']:.2f})")

    # 残りのデータをSFTとPPOに1:2の割合で分割
    remaining_data = sorted_data[hard_threshold:]
    remaining_count = len(remaining_data)

    # 1:2の割合 = SFT: 1/3, PPO: 2/3
    sft_from_remaining = int(remaining_count / 3)
    ppo_from_remaining = remaining_count - sft_from_remaining

    # 実際の割合を計算して調整
    current_sft_count = sft_from_remaining
    current_ppo_count = len(hard_questions) + ppo_from_remaining

    actual_sft_ratio = current_sft_count / total_count
    actual_ppo_ratio = current_ppo_count / total_count

    print("調整後:")
    print(f"  SFT: {current_sft_count}件 ({actual_sft_ratio*100:.1f}%)")
    print(f"  PPO: {current_ppo_count}件 ({actual_ppo_ratio*100:.1f}%)")

    # データ分割
    sft_data = remaining_data[:sft_from_remaining]
    ppo_data = hard_questions + remaining_data[sft_from_remaining:]

    return sft_data, ppo_data

def create_visualization(sft_data, ppo_data, test_data, output_dir="data"):
    """分割結果の可視化"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    datasets = [
        ("SFT", sft_data),
        ("PPO", ppo_data),
        ("Test", test_data)
    ]

    colors = ['lightblue', 'lightgreen', 'lightcoral']

    for i, (name, data) in enumerate(datasets):
        domains = [item['domain'] for item in data]
        scores = [item['score'] for item in data]

        domain_counts = Counter(domains)

        # ドメイン分布
        ax1 = axes[i, 0]
        ax1.bar(domain_counts.keys(), domain_counts.values(), color=colors[i], alpha=0.7)
        ax1.set_title(f'{name} Dataset - Domain Distribution')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)

        # スコア分布
        ax2 = axes[i, 1]
        ax2.hist(scores, bins=20, color=colors[i], alpha=0.7, edgecolor='black')
        ax2.set_title(f'{name} Dataset - Score Distribution')
        ax2.set_xlabel('Complexity Score')
        ax2.set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/dataset_split_visualization.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n可視化グラフを保存: {output_dir}/dataset_split_visualization.png")

def main():
    """メイン処理"""
    print("LLMトレーニング用データセット分割を開始します")
    print("=" * 60)

    # 設定
    input_file = "data/science_reasoning_dataset_final.jsonl"
    output_dir = "data"

    sft_output = f"{output_dir}/train_sft.jsonl"
    ppo_output = f"{output_dir}/train_ppo.jsonl"
    test_output = f"{output_dir}/test_eval.jsonl"

    # 1. データセット読み込み
    print(f"入力ファイル: {input_file}")
    data = load_dataset(input_file)
    print(f"総データ数: {len(data)}件")

    # 2. 全体統計の分析
    analyze_distribution(data, "全体データ統計")

    # 3. テストデータを層化抽出で確保 (2%)
    train_data, test_data = stratified_split_by_domain_and_score(data, test_size=0.02)

    # 4. 学習データをSFTとPPOに分割
    sft_data, ppo_data = split_sft_ppo_by_complexity(train_data, sft_ratio=0.20, ppo_ratio=0.80)

    # 5. 分割結果の分析
    analyze_distribution(sft_data, "SFTデータ統計")
    analyze_distribution(ppo_data, "PPOデータ統計")
    analyze_distribution(test_data, "テストデータ統計")

    # 6. 可視化
    create_visualization(sft_data, ppo_data, test_data, output_dir)

    # 7. ファイル保存
    print("\nファイルを保存中...")
    save_dataset(sft_data, sft_output)
    save_dataset(ppo_data, ppo_output)
    save_dataset(test_data, test_output)

    print("\n保存完了:")
    print(f"  SFT: {sft_output} ({len(sft_data)}件)")
    print(f"  PPO: {ppo_output} ({len(ppo_data)}件)")
    print(f"  Test: {test_output} ({len(test_data)}件)")

    # 8. 最終確認
    final_sft_ratio = len(sft_data) / len(data)
    final_ppo_ratio = len(ppo_data) / len(data)
    final_test_ratio = len(test_data) / len(data)

    print("\n最終割合確認:")
    print(f"  SFT: {final_sft_ratio*100:.1f}%")
    print(f"  PPO: {final_ppo_ratio*100:.1f}%")
    print(f"  Test: {final_test_ratio*100:.1f}%")
    print(f"  Total: {(final_sft_ratio + final_ppo_ratio + final_test_ratio)*100:.1f}%")

    print("\nデータセット分割が完了しました！")

if __name__ == "__main__":
    main()
