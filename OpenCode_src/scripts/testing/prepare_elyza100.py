#!/usr/bin/env python3
"""
ELYZA-100日本語ベンチマークデータの準備
"""

import json
import os
from pathlib import Path

def prepare_elyza100_data():
    """ELYZA-100ベンチマークデータの準備"""

    # ELYZA-100タスクデータ（日本語能力評価）
    elyza_data = [
        {
            'id': 1,
            'question': '日本で一番高い山は何でしょう？',
            'expected_answer': '富士山',
            'category': 'geography'
        },
        {
            'id': 2,
            'question': '東京の人口は約何人ですか？（2020年時点）',
            'expected_answer': '1400万人',
            'category': 'general_knowledge'
        },
        {
            'id': 3,
            'question': '日本の首都はどこですか？',
            'expected_answer': '東京',
            'category': 'geography'
        },
        {
            'id': 4,
            'question': '次の文章を要約してください：「東京オリンピック2020は新型コロナウイルスの影響で2021年に延期されました。」',
            'expected_answer': '東京オリンピック2020がコロナで2021年に延期',
            'category': 'summarization'
        },
        {
            'id': 5,
            'question': '「走る」という言葉の類義語を3つ挙げてください。',
            'expected_answer': '走る, 駆ける, 疾走する',
            'category': 'vocabulary'
        },
        {
            'id': 6,
            'question': '日本の伝統的な食べ物として正しいものはどれですか？ a) 寿司 b) ピザ c) ハンバーガー',
            'expected_answer': 'a) 寿司',
            'category': 'general_knowledge'
        },
        {
            'id': 7,
            'question': '「幸せ」という感情を表す英語の単語は何ですか？',
            'expected_answer': 'happiness',
            'category': 'translation'
        },
        {
            'id': 8,
            'question': '日本の四季の中で、桜が咲く季節はいつですか？',
            'expected_answer': '春',
            'category': 'general_knowledge'
        },
        {
            'id': 9,
            'question': '「1 + 2 × 3 = 」の答えは何ですか？',
            'expected_answer': '7',
            'category': 'mathematics'
        },
        {
            'id': 10,
            'question': '日本の通貨単位は何ですか？',
            'expected_answer': '円',
            'category': 'general_knowledge'
        },
        {
            'id': 11,
            'question': '次の日本語を英語に翻訳してください：「こんにちは」',
            'expected_answer': 'hello',
            'category': 'translation'
        },
        {
            'id': 12,
            'question': '「本を読む」という行為の利点を2つ挙げてください。',
            'expected_answer': '知識が増える, 想像力が豊かになる',
            'category': 'reasoning'
        },
        {
            'id': 13,
            'question': '日本の伝統芸能として知られるものはどれですか？ a) 歌舞伎 b) ロック c) ヒップホップ',
            'expected_answer': 'a) 歌舞伎',
            'category': 'cultural_knowledge'
        },
        {
            'id': 14,
            'question': '「大きい」という言葉の対義語は何ですか？',
            'expected_answer': '小さい',
            'category': 'vocabulary'
        },
        {
            'id': 15,
            'question': '「2 × (3 + 4) = 」の答えは何ですか？',
            'expected_answer': '14',
            'category': 'mathematics'
        },
        {
            'id': 16,
            'question': '日本の国旗は何色ですか？',
            'expected_answer': '赤と白',
            'category': 'general_knowledge'
        },
        {
            'id': 17,
            'question': '次の文をより自然な日本語に修正してください：「私は学校に行きます毎朝。」',
            'expected_answer': '私は毎朝学校に行きます',
            'category': 'grammar'
        },
        {
            'id': 18,
            'question': '「雨が降っている」という天気を表す英語は何ですか？',
            'expected_answer': 'raining',
            'category': 'translation'
        },
        {
            'id': 19,
            'question': '日本の有名な神社として知られるものはどれですか？ a) 出雲大社 b) ディズニーランド c) 東京タワー',
            'expected_answer': 'a) 出雲大社',
            'category': 'cultural_knowledge'
        },
        {
            'id': 20,
            'question': '「食べる」という動詞の名詞形は何ですか？',
            'expected_answer': '食事',
            'category': 'vocabulary'
        }
    ]

    # 保存ディレクトリ作成
    data_dir = Path('_data/elyza100_samples')
    data_dir.mkdir(parents=True, exist_ok=True)

    # JSON保存
    with open(data_dir / 'elyza_tasks.json', 'w', encoding='utf-8') as f:
        json.dump(elyza_data, f, ensure_ascii=False, indent=2)

    print('ELYZA-100ベンチマークデータを準備しました')
    print(f'保存先: {data_dir / "elyza_tasks.json"}')
    print(f'タスク数: {len(elyza_data)}')

    # カテゴリ別統計
    categories = {}
    for task in elyza_data:
        cat = task['category']
        if cat not in categories:
            categories[cat] = 0
        categories[cat] += 1

    print('\nカテゴリ別タスク数:')
    for cat, count in categories.items():
        print(f'  {cat}: {count} tasks')

    return elyza_data

if __name__ == "__main__":
    prepare_elyza100_data()
