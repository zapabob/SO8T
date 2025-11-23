#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T/thinkingモデルのための大規模合成データ生成スクリプト
"""

import json
import random
import os
from pathlib import Path

def generate_so8t_thinking_sample():
    """SO8T/thinking用の単一サンプルを生成"""
    reasoning_types = ['mathematical', 'logical', 'geometric', 'scientific', 'general']
    reasoning_type = random.choice(reasoning_types)

    templates = {
        'mathematical': [
            '次の数学の問題をステップバイステップで解いてください：複雑な数学的問題を解き明かしてください。',
            '数学的に証明してください：重要な数学的定理の妥当性を証明しなさい。',
            'この方程式を解いてください：高次方程式の解を求めなさい。',
            '微分方程式を解いてください：与えられた微分方程式の一般解を導きなさい。',
            '行列の固有値を求めてください：与えられた行列の固有値と固有ベクトルを計算しなさい。'
        ],
        'logical': [
            'この論理パズルを解いてください：複雑な論理パズルの解決策を導きなさい。',
            '論理的に考えて、結論を導いてください：与えられた前提から論理的結論を導きなさい。',
            'この主張の妥当性を評価してください：論理的主張の妥当性を批判的に評価しなさい。',
            '演繹推理を行ってください：与えられた命題から演繹的に結論を導きなさい。',
            '論理的矛盾を特定してください：与えられた主張における論理的矛盾点を指摘しなさい。'
        ],
        'geometric': [
            '幾何学的に考えて、この問題を解いてください：複雑な幾何学問題を幾何学的に解決しなさい。',
            'この図形の性質を説明してください：複雑な幾何学図形の性質を詳細に説明しなさい。',
            '空間的な関係を分析してください：空間的関係を幾何学的に分析しなさい。',
            'ベクトル幾何学の問題を解いてください：ベクトル幾何学の問題を解決しなさい。',
            '座標幾何学で点をプロットしてください：座標幾何学における点を正確にプロットしなさい。'
        ],
        'scientific': [
            '科学的に考えて、この現象を説明してください：科学的現象を理論的に説明しなさい。',
            'この仮説の妥当性を評価してください：科学的な仮説の妥当性を評価しなさい。',
            '実験結果を解釈してください：実験データを科学的に解釈しなさい。',
            '因果関係を分析してください：現象間の因果関係を科学的に分析しなさい。',
            '科学的方法論を適用してください：科学的方法論を用いて問題を解決しなさい。'
        ],
        'general': [
            'この問題について深く考えてください：哲学的な問いについて深く考察しなさい。',
            '多角的に分析してください：多角的なトピックを包括的に分析しなさい。',
            '批判的に考察してください：現代的な問題を批判的に考察しなさい。',
            'システム思考を適用してください：複雑なシステムをシステム思考で分析しなさい。',
            '創造的に解決してください：創造的なアプローチで問題を解決しなさい。'
        ]
    }

    content = random.choice(templates[reasoning_type])

    return {
        'text': content,
        'source': 'synthetic_so8t_enhanced',
        'reasoning_type': reasoning_type,
        'metadata': {
            'synthetic': True,
            'complexity': random.choice(['low', 'medium', 'high']),
            'reasoning_depth': random.choice(['shallow', 'medium', 'deep']),
            'domain': reasoning_type
        }
    }

def main():
    """メイン関数"""
    print('SO8T/thinking大規模合成データ生成開始...')

    # 既存の統合データセットを確認
    base_file = 'data/so8t_thinking_large_train.jsonl'
    if not os.path.exists(base_file):
        print(f'ベースファイル {base_file} が存在しません。')
        return

    # 現在のサイズを確認
    with open(base_file, 'r', encoding='utf-8') as f:
        current_samples = sum(1 for _ in f)
    print(f'現在のサンプル数: {current_samples}')

    # 追加生成するサンプル数
    additional_samples = 30000  # 3万サンプル追加

    print(f'{additional_samples} サンプルの合成データを生成中...')

    # 追加データを生成
    with open(base_file, 'a', encoding='utf-8') as f:
        for i in range(additional_samples):
            sample = generate_so8t_thinking_sample()
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

            if (i + 1) % 5000 == 0:
                print(f'{i + 1} / {additional_samples} サンプル生成完了...')

    # 最終サイズを確認
    with open(base_file, 'r', encoding='utf-8') as f:
        final_samples = sum(1 for _ in f)

    print(f'生成完了!')
    print(f'追加サンプル数: {additional_samples}')
    print(f'最終データセットサイズ: {final_samples} サンプル')

    # データセットの統計を表示
    reasoning_counts = {}
    with open(base_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                reasoning_type = data.get('reasoning_type', 'unknown')
                reasoning_counts[reasoning_type] = reasoning_counts.get(reasoning_type, 0) + 1
            except:
                continue

    print('\nデータセット統計:')
    for reasoning_type, count in reasoning_counts.items():
        print(f'  {reasoning_type}: {count} サンプル')

if __name__ == '__main__':
    main()
