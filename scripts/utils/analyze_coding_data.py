#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""コーディング関連データ分析スクリプト"""

from pathlib import Path
import json
from collections import Counter

def main():
    fp = Path('D:/webdataset/processed/four_class/four_class_improved_20251109_101005.jsonl')
    
    if not fp.exists():
        print(f"ファイルが見つかりません: {fp}")
        return
    
    lines = fp.read_text(encoding='utf-8').strip().split('\n')
    samples = [json.loads(l) for l in lines if l.strip()]
    
    print("="*80)
    print("コーディング関連データ分析")
    print("="*80)
    
    # コーディング関連キーワード
    coding_keywords = [
        'code', 'programming', 'python', 'javascript', 'java', 'c++', 'coding',
        '開発', 'プログラミング', 'コード', '実装', '関数', 'クラス', 'アルゴリズム',
        'algorithm', 'function', 'class', 'variable', 'loop', 'if', 'else',
        '変数', 'ループ', '条件分岐', 'API', 'library', 'framework', 'ライブラリ',
        'フレームワーク', 'git', 'github', 'repository', 'リポジトリ', 'commit',
        'pull request', 'merge', 'branch', 'デバッグ', 'debug', 'test', 'テスト',
        'syntax', 'シンタックス', 'エラー', 'error', 'exception', '例外'
    ]
    
    # コーディング関連サンプルを抽出
    coding_samples = []
    for s in samples:
        text = s.get('text', '').lower()
        keyword = s.get('keyword', '').lower()
        category = s.get('category', '').lower()
        
        # キーワードマッチング
        if any(kw.lower() in text or kw.lower() in keyword or kw.lower() in category for kw in coding_keywords):
            coding_samples.append(s)
    
    print(f"\n総サンプル数: {len(samples)}")
    print(f"コーディング関連サンプル: {len(coding_samples)} ({len(coding_samples)/len(samples)*100:.1f}%)")
    
    # カテゴリ分布
    categories = Counter()
    for s in coding_samples:
        cat = s.get('category', 'unknown')
        categories[cat] += 1
    
    print(f"\nコーディング関連カテゴリ分布（上位10）:")
    for cat, count in categories.most_common(10):
        print(f"  {cat}: {count} ({count/len(coding_samples)*100:.1f}%)")
    
    # 言語分布
    languages = Counter()
    for s in coding_samples:
        lang = s.get('language', 'unknown')
        languages[lang] += 1
    
    print(f"\nコーディング関連言語分布:")
    for lang, count in sorted(languages.items()):
        print(f"  {lang}: {count} ({count/len(coding_samples)*100:.1f}%)")
    
    # テキスト長統計
    text_lengths = [len(s.get('text', '')) for s in coding_samples]
    if text_lengths:
        print(f"\nコーディング関連テキスト長統計:")
        print(f"  最小: {min(text_lengths)}文字")
        print(f"  最大: {max(text_lengths)}文字")
        print(f"  平均: {sum(text_lengths)/len(text_lengths):.1f}文字")
        print(f"  中央値: {sorted(text_lengths)[len(text_lengths)//2]}文字")
    
    # 分類分布
    classifications = Counter()
    for s in coding_samples:
        quad_class = s.get('quadruple_classification', {})
        four_class_label = quad_class.get('four_class_label', 'unknown')
        classifications[four_class_label] += 1
    
    print(f"\nコーディング関連分類分布:")
    for label, count in sorted(classifications.items()):
        print(f"  {label}: {count} ({count/len(coding_samples)*100:.1f}%)")

if __name__ == "__main__":
    main()

