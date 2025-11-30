#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8-Think Science Data Curation V2
高度な科学・数学データセット作成スクリプト

NKAT理論に基づくPhD/Fields賞級推論能力付与のための
高度なフィルタリング & 複雑度スコアリング

著者: 峯岸亮 (SO8Tプロジェクト)
"""

import argparse
import os
import re
import random
import json
import numpy as np
from datasets import load_dataset, Dataset
import pandas as pd
from tqdm import tqdm
from difflib import SequenceMatcher


# ==========================================
# 1. 高度なフィルタリング & スコアリング
# ==========================================

def has_latex(text):
    """数式(LaTeX)が含まれているかチェック"""
    if not isinstance(text, str): return False

    latex_patterns = [r'\\frac', r'\\int', r'\\sum', r'\$', r'\\partial', r'\\alpha', r'=']
    return any(p in text for p in latex_patterns)

def is_high_quality(text):
    """拒絶応答や短すぎるテキストを弾く"""
    if not isinstance(text, str): return False
    if len(text) < 100: return False

    refusal_keywords = [
        "I don't know", "I cannot", "As an AI", "sorry", "unable to",
        "私が知る限り", "申し訳ありません", "I apologize"
    ]
    if any(kw in text for kw in refusal_keywords):
        return False
    return True

def calculate_complexity_score(text):
    """複雑度スコア: 長さ × LaTeX密度 × 推論ステップ数"""
    if not isinstance(text, str): return 0

    # 長さスコア (対数)
    length_score = np.log(len(text) + 1)

    # LaTeXスコア
    latex_score = 2.0 if has_latex(text) else 1.0

    # 推論ステップスコア (CoTの深さ)
    # "Therefore", "Because", "However", "Step 1" などの論理接続詞をカウント
    logic_keywords = ["therefore", "because", "however", "implies", "step", "thus", "since", "assuming", "conclude"]
    logic_count = sum(1 for w in logic_keywords if w in text.lower())
    logic_score = 1.0 + (logic_count * 0.1)

    return length_score * latex_score * logic_score

def is_duplicate(text, existing_texts, threshold=0.8):
    """簡易的な重複チェック (時間がかかるのでサンプリングして比較)"""
    if not existing_texts:
        return False
    # 直近の100件と比較 (全件比較は重すぎる)
    sample_size = min(len(existing_texts), 50)
    samples = existing_texts[-sample_size:]

    for existing in samples:
        # 先頭100文字だけで高速判定
        if SequenceMatcher(None, text[:100], existing[:100]).ratio() > threshold:
            return True
    return False

# ==========================================
# 2. データセット別処理ロジック
# ==========================================

def process_dataset(name, split, n_samples, domain_tag):
    print(f"Loading {domain_tag} dataset: {name} (Target: {n_samples})...")
    try:
        # trust_remote_code=True は削除！
        ds = load_dataset(name, split=split)
        ds = ds.shuffle(seed=42)
    except Exception as e:
        print(f"Failed to load {name}: {e}")
        return []

    filtered_data = []
    seen_instructions = [] # 重複チェック用

    # プログレスバー
    pbar = tqdm(total=n_samples, desc=f"Filtering {domain_tag}")

    for item in ds:
        # カラム名の揺らぎ吸収
        instruction = item.get('instruction') or item.get('problem') or item.get('question') or item.get('message_1') or ""
        output = item.get('output') or item.get('solution') or item.get('answer') or item.get('response') or item.get('message_2') or ""

        # 結合テキストで品質チェック
        full_text = f"{instruction} {output}"

        # フィルタリング実行
        if is_high_quality(output) and not is_duplicate(instruction, seen_instructions):

            # 数学の場合はLaTeX必須
            if domain_tag == "math" and not has_latex(output):
                continue

            score = calculate_complexity_score(output)

            filtered_data.append({
                "instruction": instruction,
                "input": "",
                "output": output,
                "domain": domain_tag,
                "score": score
            })

            seen_instructions.append(instruction)
            pbar.update(1)

        if len(filtered_data) >= n_samples * 1.2: # スコアソート用に少し多めに確保
            break

    pbar.close()

    # スコアでソートして上位を返す
    filtered_data.sort(key=lambda x: x['score'], reverse=True)
    return filtered_data[:n_samples]

# ==========================================
# 3. システムプロンプト生成器
# ==========================================

SYSTEM_PROMPTS = [
    # 物理学者モード
    "あなたはNKAT理論に基づく物理的知性を持つAIです。SO(8)群のトライアリティ構造に基づき、物理法則と数学的定理から厳密な推論を行ってください。",
    # 数学者モード
    "あなたはフィールズ賞級の洞察力を持つ数学AIです。論理の飛躍を避け、公理から定理を導くようにステップ・バイ・ステップで証明を行ってください。",
    # 哲学者/統合モード
    "あなたは高度な知性を持つ統合AIです。異なる分野（数学・物理・生物）の間に同型性（Isomorphism）を見出し、多角的な視点から結論を導いてください。"
]

def get_random_system_prompt():
    return random.choice(SYSTEM_PROMPTS)

# ==========================================
# 4. メイン実行関数
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="SO8-Think Science Data Curation V2")
    parser.add_argument("--output", type=str, default="data/science_reasoning_dataset.jsonl")
    parser.add_argument("--total_samples", type=int, default=50000)
    args = parser.parse_args()

    # ターゲット内訳
    targets = {
        "math": ("AI-MO/NuminaMath-CoT", int(args.total_samples * 0.4)),
        "physics": ("camel-ai/physics", int(args.total_samples * 0.3)),
        # Magpieの代わりにOpenReasoningに変更（よりCoT向き）
        "reasoning": ("OpenReasoning/OpenReasoning-CoT", int(args.total_samples * 0.3))
    }

    all_data = []

    for domain, (repo, count) in targets.items():
        data = process_dataset(repo, "train", count, domain)
        all_data.extend(data)

    # 全体シャッフル
    random.shuffle(all_data)

    # 保存処理
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"\nWriting {len(all_data)} samples to {args.output}...")

    with open(args.output, 'w', encoding='utf-8') as f:
        for item in tqdm(all_data, desc="Writing JSONL"):
            # システムプロンプトをランダム注入
            item["system"] = get_random_system_prompt()
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

    print("Done! 💎知性のダイヤモンド V2💎 採掘完了や！")

if __name__ == "__main__":
    main()





