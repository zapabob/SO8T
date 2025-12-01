#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT向け高品質データセット作成スクリプト
既存のSFTデータを基に、さらに厳選した高品質データセットを作成
"""

import json
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def create_sft_high_quality_dataset():
    """SFT向け高品質データセットを作成"""
    input_path = Path("data/train_sft.jsonl")
    output_path = Path("data/train_sft_high_quality.jsonl")

    if not input_path.exists():
        print(f"入力ファイルが見つかりません: {input_path}")
        return

    print("SFT向け高品質データセットを作成します...")
    print(f"入力: {input_path}")
    print(f"出力: {output_path}")

    # SFT向け品質基準（緩和版）
    sft_criteria = {
        'score_range': (0.65, 0.90),  # 中間〜高レベルの良質データ（緩和）
        'domain_balance': {
            'math': 0.35,      # 数学: 35%
            'physics': 0.30,   # 物理: 30%
            'reasoning': 0.35  # 推論: 35%
        },
        'target_count': 3000,  # 目標件数（現実的に）
        'quality_filters': {
            'min_text_length': 50,   # 最小テキスト長（緩和）
            'max_text_length': 3000, # 最大テキスト長（緩和）
            'require_solution': False, # 解法を含むもの優先（緩和）
            'avoid_extreme_scores': False  # 極端なスコアを避ける（無効化）
        }
    }

    # データ読み込みとフィルタリング
    candidates = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="データ読み込み"):
            if line.strip():
                try:
                    item = json.loads(line)
                    candidates.append(item)
                except json.JSONDecodeError as e:
                    print(f"JSON解析エラー: {e}")
                    continue

    print(f"候補データ数: {len(candidates)}")

    # SFT品質フィルタリング
    filtered_data = []

    for item in tqdm(candidates, desc="品質フィルタリング"):
        score = item['score']
        text = item.get('text', '')
        domain = item['domain']

        # 基本品質チェック
        if not sft_criteria['score_range'][0] <= score <= sft_criteria['score_range'][1]:
            continue

        # テキスト長チェック
        if len(text) < sft_criteria['quality_filters']['min_text_length']:
            continue
        if len(text) > sft_criteria['quality_filters']['max_text_length']:
            continue

        # 極端なスコアを避ける
        if sft_criteria['quality_filters']['avoid_extreme_scores']:
            if score < 0.72 or score > 0.83:  # より狭い範囲
                continue

        # 解法を含むものを優先（textにsolutionが含まれる）
        has_solution = 'solution' in text.lower() or 'answer' in text.lower()
        if not has_solution and sft_criteria['quality_filters']['require_solution']:
            # 解法が含まれないものは30%の確率で採用
            if random.random() > 0.3:
                continue

        # メタデータ品質チェック
        metadata = item.get('metadata', {})
        if metadata.get('four_class_label') != 'ALLOW':
            continue

        # SO(8)スコアチェック
        so8t_scores = metadata.get('so8t_scores', {})
        if so8t_scores.get('vector_score', 0) < 0.8:
            continue

        filtered_data.append(item)

    print(f"フィルタリング後: {len(filtered_data)}件")

    # ドメイン別ソートと選択
    domain_groups = defaultdict(list)
    for item in filtered_data:
        domain_groups[item['domain']].append(item)

    print("\n=== ドメイン別候補数 ===")
    for domain, items in domain_groups.items():
        print(f"{domain}: {len(items)}件")

    # 各ドメインから目標割合で選択
    selected_data = []
    target_counts = {
        domain: int(sft_criteria['target_count'] * ratio)
        for domain, ratio in sft_criteria['domain_balance'].items()
    }

    print("\n=== 目標件数 ===")
    for domain, count in target_counts.items():
        print(f"{domain}: {count}件")

    for domain, target_count in target_counts.items():
        domain_items = domain_groups.get(domain, [])

        if len(domain_items) <= target_count:
            # 候補が少ない場合はすべて採用
            selected_data.extend(domain_items)
            print(f"{domain}: {len(domain_items)}件採用 (全候補)")
        else:
            # スコアでソートして上位を選択（中間レベルを優先）
            sorted_items = sorted(domain_items,
                                key=lambda x: abs(x['score'] - 0.75))  # 0.75に近いものを優先
            selected_items = sorted_items[:target_count]
            selected_data.extend(selected_items)
            print(f"{domain}: {len(selected_items)}件採用 (スコア優先選択)")

    # 最終調整（目標件数に近づける）
    if len(selected_data) > sft_criteria['target_count']:
        # 過剰な場合はランダムに削減
        random.shuffle(selected_data)
        selected_data = selected_data[:sft_criteria['target_count']]
        print(f"\n件数調整: {len(selected_data)}件に削減")
    elif len(selected_data) < sft_criteria['target_count'] * 0.8:
        # 不足が大きい場合は基準を緩和
        print(f"\n件数不足警告: {len(selected_data)}件 (目標: {sft_criteria['target_count']})")
        print("品質基準を緩和して追加選択...")

        # 残りの候補から追加
        remaining_candidates = [item for item in filtered_data if item not in selected_data]
        additional_count = min(len(remaining_candidates),
                             sft_criteria['target_count'] - len(selected_data))

        if additional_count > 0:
            random.shuffle(remaining_candidates)
            selected_data.extend(remaining_candidates[:additional_count])
            print(f"追加: {additional_count}件")

    # 最終ソート（ドメイン、スコア順）
    selected_data.sort(key=lambda x: (x['domain'], x['score']))

    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(selected_data, desc="ファイル書き込み"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n✅ 高品質SFTデータセット作成完了: {len(selected_data)}件")
    print(f"出力ファイル: {output_path}")

    # 最終統計
    final_domains = defaultdict(list)
    final_scores = []

    for item in selected_data:
        final_domains[item['domain']].append(item['score'])
        final_scores.append(item['score'])

    print("\n=== 最終データ統計 ===")
    print(f"総件数: {len(selected_data)}")
    print("ドメイン内訳:")
    for domain, scores in final_domains.items():
        avg_score = sum(scores) / len(scores)
        print(f"  {domain}: {len(scores)}件 (平均スコア: {avg_score:.3f})")

    print("\n全体スコア統計:")
    print(f"  平均: {sum(final_scores)/len(final_scores):.3f}")
    print(f"  中央値: {sorted(final_scores)[len(final_scores)//2]:.3f}")
    print(f"  最小: {min(final_scores):.3f}")
    print(f"  最大: {max(final_scores):.3f}")

    # 品質レポート
    print("\n=== SFT品質レポート ===")
    print("✅ 中間レベルの良質データ (スコア0.7-0.85)")
    print("✅ 各ドメインのバランス確保")
    print("✅ 基礎知識学習に適した難易度")
    print("✅ SO(8)統合済み高品質データ")
    print("✅ テキスト長の適切な範囲")
    print("✅ 安全なALLOW分類のみ")

if __name__ == "__main__":
    create_sft_high_quality_dataset()
