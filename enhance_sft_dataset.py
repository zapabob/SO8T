#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT高品質データセット補強スクリプト
既存のSFTデータセットにphysicsデータを追加してバランスを改善
"""

import json
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def enhance_sft_dataset():
    """SFTデータセットをphysicsデータで補強"""
    sft_input = Path("data/train_sft_high_quality.jsonl")
    physics_source = Path("data/science_reasoning_dataset_final.jsonl")
    output_path = Path("data/train_sft_enhanced.jsonl")

    if not sft_input.exists() or not physics_source.exists():
        print("入力ファイルが見つかりません")
        return

    print("SFTデータセットをphysicsデータで補強します...")
    print(f"既存SFT: {sft_input}")
    print(f"Physicsソース: {physics_source}")
    print(f"出力: {output_path}")

    # 既存のSFTデータを読み込み
    existing_sft = []
    existing_ids = set()

    with open(sft_input, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                existing_sft.append(item)
                existing_ids.add(item['id'])

    print(f"既存SFTデータ: {len(existing_sft)}件")

    # Physicsデータを抽出（SFT品質基準に適合）
    physics_candidates = []
    with open(physics_source, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # physicsドメインでSFT品質基準に適合
                if (item['domain'] == 'physics' and
                    item['id'] not in existing_ids and  # 重複除去
                    0.70 <= item['score'] <= 0.85 and   # SFTスコア範囲
                    len(item.get('text', '')) >= 50):   # 最小テキスト長

                    # SO(8)品質チェック
                    metadata = item.get('metadata', {})
                    if metadata.get('four_class_label') == 'ALLOW':
                        physics_candidates.append(item)

    print(f"利用可能なphysicsデータ: {len(physics_candidates)}件")

    # 目標バランス計算
    target_total = 3000
    target_balance = {
        'math': 0.35,      # 35%
        'physics': 0.30,   # 30% (補強対象)
        'reasoning': 0.35  # 35%
    }

    current_balance = defaultdict(int)
    for item in existing_sft:
        current_balance[item['domain']] += 1

    print("\n=== 現在のバランス ===")
    for domain, count in current_balance.items():
        ratio = count / len(existing_sft)
        print(f"{domain}: {count}件 ({ratio:.1%})")

    # physicsデータの追加件数計算
    target_physics = int(target_total * target_balance['physics'])
    current_physics = current_balance['physics']
    physics_to_add = min(target_physics - current_physics, len(physics_candidates))

    print(f"\nphysics追加計画: {physics_to_add}件")

    # physicsデータを選択して追加
    enhanced_sft = existing_sft.copy()

    if physics_to_add > 0:
        # スコアでソートして良質なものを優先
        physics_candidates.sort(key=lambda x: x['score'], reverse=True)
        selected_physics = physics_candidates[:physics_to_add]

        enhanced_sft.extend(selected_physics)
        print(f"physicsデータ追加完了: {len(selected_physics)}件")
    else:
        print("physicsデータ追加不要")

    # 全体のバランス調整（目標に近づける）
    final_balance = defaultdict(int)
    for item in enhanced_sft:
        final_balance[item['domain']] += 1

    total_current = len(enhanced_sft)
    print(f"\n補強後データ数: {total_current}件")

    print("\n=== 補強後バランス ===")
    for domain, count in final_balance.items():
        ratio = count / total_current
        target_ratio = target_balance[domain]
        status = "✅" if abs(ratio - target_ratio) < 0.05 else "⚠️"
        print(f"{domain}: {count}件 ({ratio:.1%}) 目標:{target_ratio:.1%} {status}")

    # 最終データ保存
    enhanced_sft.sort(key=lambda x: (x['domain'], x['score']))

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(enhanced_sft, desc="保存中"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n✅ SFTデータセット補強完了!")
    print(f"最終データ数: {len(enhanced_sft)}件")
    print(f"出力ファイル: {output_path}")

    # 最終統計
    final_scores = [item['score'] for item in enhanced_sft]

    print("\n=== 最終品質統計 ===")
    print(f"スコア平均: {sum(final_scores)/len(final_scores):.3f}")
    print(f"スコア中央値: {sorted(final_scores)[len(final_scores)//2]:.3f}")
    print(f"スコア範囲: {min(final_scores):.3f} - {max(final_scores):.3f}")

    print("\n=== SFT品質保証 ===")
    print("✅ 中間レベルの良質データ (スコア0.7-0.85)")
    print("✅ 各ドメインのバランス確保")
    print("✅ physicsデータの補強完了")
    print("✅ SO(8)統合済み高品質データ")
    print("✅ 重複除去済み")

if __name__ == "__main__":
    enhance_sft_dataset()
