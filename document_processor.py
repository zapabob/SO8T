#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ドキュメント処理スクリプト
3つの数学ドキュメントをデータセット形式に変換
"""

import json
import re
import os
from pathlib import Path
from typing import Dict, List, Any
import hashlib
import math

def calculate_so8t_score(text: str) -> Dict[str, float]:
    """SO(8)Tスコア計算"""
    # テキストの数学的複雑さを評価
    math_symbols = len(re.findall(r'[∫∑∏∂∇∞≠≈≡≪≫≤≥]', text))
    greek_letters = len(re.findall(r'[αβγδεζηθικλμνξοπρστυφχψω]', text))
    equations = len(re.findall(r'\$[^$]+\$', text))
    theorems = len(re.findall(r'(theorem|定理|補題)', text.lower()))

    # 複雑度スコア計算
    complexity = (math_symbols * 0.3 + greek_letters * 0.2 + equations * 0.4 + theorems * 0.1)
    complexity = min(complexity / 10.0, 1.0)  # 0-1に正規化

    # SO(8)スコア生成（物理学理論に基づく）
    phi = (1 + math.sqrt(5)) / 2
    vector_score = min(complexity * 1.2, 1.0)
    spinor_plus = min(complexity * phi * 0.3, 1.0)
    spinor_minus = min(complexity * (1 - phi) * 0.1, 1.0)
    combined_score = (vector_score + spinor_plus - spinor_minus) / 3.0

    return {
        'vector_score': vector_score,
        'spinor_plus_score': spinor_plus,
        'spinor_minus_score': spinor_minus,
        'combined_score': combined_score
    }

def classify_document(text: str) -> str:
    """ドキュメント分類"""
    text_lower = text.lower()

    # ドメイン分類
    if any(keyword in text_lower for keyword in ['quantum', '量子', 'field', '場', 'particle', '粒子', 'physics', '物理']):
        return 'physics'
    elif any(keyword in text_lower for keyword in ['theorem', '定理', 'proof', '証明', 'mathematics', '数学', 'algebra', '代数']):
        return 'math'
    else:
        return 'reasoning'

def classify_four_class(text: str) -> str:
    """四値分類"""
    # 安全性の高い数学的内容は基本的にALLOW
    dangerous_keywords = ['hack', 'exploit', 'illegal', 'dangerous', 'harmful']

    if any(keyword in text.lower() for keyword in dangerous_keywords):
        return 'DENY'
    else:
        return 'ALLOW'

def create_dataset_entry(text: str, title: str, source: str) -> Dict[str, Any]:
    """データセットエントリ作成"""
    # ID生成
    content_hash = hashlib.md5(f"{title}{text}".encode()).hexdigest()[:16]
    entry_id = f"doc_{content_hash}"

    # スコア計算
    score = calculate_so8t_score(text)['combined_score'] * 0.8 + 0.2  # 0.2-1.0の範囲に調整

    # ドメイン分類
    domain = classify_document(text)

    # SO(8)Tスコア
    so8t_scores = calculate_so8t_score(text)

    # 四値分類
    four_class = classify_four_class(text)

    # PPO形式のメッセージ作成
    messages = [
        {
            "content": f"以下の数学的内容について説明してください：\n\n{title}\n\n{text[:1000]}...",
            "role": "user"
        },
        {
            "content": f"これは{domain}分野の高度な数学的内容です。{title}について説明します。",
            "role": "assistant"
        }
    ]

    return {
        "id": entry_id,
        "text": json.dumps({
            "instruction": f"以下の{domain}分野の数学的内容について説明してください：\n\n{title}",
            "output": f"これは{domain}分野の高度な数学的定理です。{title}について詳しく説明します。",
            "messages": messages
        }, ensure_ascii=False),
        "domain": domain,
        "score": round(score, 3),
        "language": "ja",
        "source_dataset": source,
        "metadata": {
            "original_ppo_format": True,
            "four_class_label": four_class,
            "quality_score": 0.8,  # 高品質ドキュメント
            "so8t_scores": so8t_scores,
            "document_type": "mathematical_manuscript",
            "word_count": len(text.split())
        }
    }

def process_document(doc_path: str, source_name: str) -> List[Dict[str, Any]]:
    """ドキュメント処理"""
    print(f"Processing document: {doc_path}")

    if not os.path.exists(doc_path):
        print(f"Document not found: {doc_path}")
        return []

    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"Document length: {len(content)} characters")

    # ドキュメントをセクションに分割
    sections = re.split(r'^#+\s+', content, flags=re.MULTILINE)

    entries = []

    for section in sections:
        if section.strip() and len(section.strip()) > 100:  # 意味のある長さのセクションのみ
            lines = section.strip().split('\n')
            title = lines[0].strip() if lines else 'Untitled Section'
            body = '\n'.join(lines[1:]).strip()

            if body:  # 本文がある場合のみ
                entry = create_dataset_entry(body, title, source_name)
                entries.append(entry)

    print(f"Generated {len(entries)} entries from {len(sections)} sections")
    return entries

def main():
    """メイン処理"""
    print("🔬 Document Integration Pipeline")
    print("=" * 50)

    # ドキュメントパス
    documents = [
        (r'C:\Users\downl\Desktop\ChatGPT-非可換KART定理 (1).md', 'chatgpt_noncommutative_kart'),
        (r'C:\Users\downl\Desktop\Gemini-統合特解と非可換表現理論.md', 'gemini_unified_solution_theorem'),
        (r'C:\Users\downl\Desktop\Gemini-NC-KART★とURTの数学的探求.md', 'gemini_nc_kart_urt_exploration')
    ]

    all_entries = []

    # 各ドキュメントを処理
    for doc_path, source_name in documents:
        entries = process_document(doc_path, source_name)
        all_entries.extend(entries)

    print(f"\nTotal entries generated: {len(all_entries)}")

    # 既存のデータセットに統合
    existing_dataset = 'data/train_ppo.jsonl'
    integrated_dataset = 'data/train_ppo_integrated.jsonl'

    # 既存データを読み込み
    existing_entries = []
    if os.path.exists(existing_dataset):
        with open(existing_dataset, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    existing_entries.append(json.loads(line))

    print(f"Existing entries: {len(existing_entries)}")

    # 統合
    integrated_entries = existing_entries + all_entries
    print(f"Integrated entries: {len(integrated_entries)}")

    # 保存
    with open(integrated_dataset, 'w', encoding='utf-8') as f:
        for entry in integrated_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"✅ Integrated dataset saved: {integrated_dataset}")

    # 統計表示
    domains = {}
    scores = []

    for entry in all_entries:
        domain = entry['domain']
        score = entry['score']
        domains[domain] = domains.get(domain, 0) + 1
        scores.append(score)

    print("\n=== New Document Statistics ===")
    print(f"Total new entries: {len(all_entries)}")
    print("Domain distribution:")
    for domain, count in domains.items():
        print(f"  {domain}: {count} entries")

    if scores:
        print(f"Score statistics:")
        print(f"  Mean: {sum(scores)/len(scores):.3f}")
        print(f"  Min: {min(scores):.3f}")
        print(f"  Max: {max(scores):.3f}")

if __name__ == "__main__":
    main()

