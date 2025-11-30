#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8Tデータセット品質分析スクリプト
データセットの品質、分布、多様性を評価
"""

import json
import os
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Any

def analyze_dataset_quality(dataset_path: str) -> Dict[str, Any]:
    """データセット品質分析"""

    print("=== SO8Tデータセット品質分析 ===")
    print(f"データセット: {dataset_path}")

    # データ読み込み
    data = []
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"JSON解析エラー (行{line_num}): {e}")
                        continue
    except FileNotFoundError:
        print(f"データセットファイルが見つかりません: {dataset_path}")
        return {}

    total_samples = len(data)
    print(f"総サンプル数: {total_samples:,}")

    if total_samples == 0:
        return {"error": "データセットが空です"}

    # 基本統計
    analysis = {
        "total_samples": total_samples,
        "quality_metrics": {},
        "distribution_analysis": {},
        "content_analysis": {},
        "safety_analysis": {}
    }

    # 1. 品質メトリクス分析
    print("\n=== 品質メトリクス分析 ===")

    # テキスト長分布
    text_lengths = []
    for item in data:
        text = item.get('text', '')
        text_lengths.append(len(text))

    analysis["quality_metrics"]["text_length_stats"] = {
        "mean": sum(text_lengths) / len(text_lengths),
        "min": min(text_lengths),
        "max": max(text_lengths),
        "median": sorted(text_lengths)[len(text_lengths)//2]
    }

    print("テキスト長統計:")
    print(".2f")
    print(f"  最小: {analysis['quality_metrics']['text_length_stats']['min']}")
    print(f"  最大: {analysis['quality_metrics']['text_length_stats']['max']}")
    print(f"  中央値: {analysis['quality_metrics']['text_length_stats']['median']}")

    # 2. 分類分布分析
    print("\n=== 分類分布分析 ===")

    reasoning_types = Counter()
    categories = Counter()
    sources = Counter()

    drug_detection = 0
    nsfw_erotic = 0
    nsfw_violence = 0
    safety_detection = 0

    for item in data:
        # reasoning_type
        rt = item.get('reasoning_type', 'unknown')
        reasoning_types[rt] += 1

        # source
        source = item.get('source', 'unknown')
        sources[source] += 1

        # classifications
        metadata = item.get('metadata', {})
        classifications = metadata.get('classifications', {})

        if classifications.get('drug_detection', False):
            drug_detection += 1
        if classifications.get('nsfw_erotic', False):
            nsfw_erotic += 1
        if classifications.get('nsfw_violence', False):
            nsfw_violence += 1
        if classifications.get('safety_detection', False):
            safety_detection += 1

    analysis["distribution_analysis"] = {
        "reasoning_types": dict(reasoning_types.most_common()),
        "sources": dict(sources.most_common()),
        "safety_flags": {
            "drug_detection": drug_detection,
            "nsfw_erotic": nsfw_erotic,
            "nsfw_violence": nsfw_violence,
            "safety_detection": safety_detection
        }
    }

    print("Reasoning Type分布:")
    for rt, count in reasoning_types.most_common():
        print(".1f")

    print("\nSafety Flags分布:")
    print(f"  Drug Detection: {drug_detection} ({drug_detection/total_samples*100:.1f}%)")
    print(f"  NSFW Erotic: {nsfw_erotic} ({nsfw_erotic/total_samples*100:.1f}%)")
    print(f"  NSFW Violence: {nsfw_violence} ({nsfw_violence/total_samples*100:.1f}%)")
    print(f"  Safety Detection: {safety_detection} ({safety_detection/total_samples*100:.1f}%)")

    # 3. 品質スコア分析
    print("\n=== 品質スコア分析 ===")

    quality_scores = []
    weights = []
    qc_controlled = 0

    for item in data:
        metadata = item.get('metadata', {})

        # QC controlled count
        if metadata.get('qc_controlled', False):
            qc_controlled += 1

        # weights
        weight = metadata.get('weight', 1.0)
        weights.append(weight)

    analysis["quality_metrics"]["qc_stats"] = {
        "qc_controlled_count": qc_controlled,
        "qc_controlled_ratio": qc_controlled / total_samples
    }

    analysis["quality_metrics"]["weight_stats"] = {
        "mean": sum(weights) / len(weights),
        "min": min(weights),
        "max": max(weights)
    }

    print("QC統計:")
    print(".1f")
    print(".1f")

    print("Weight統計:")
    print(".3f")
    print(f"  最小: {analysis['quality_metrics']['weight_stats']['min']}")
    print(f"  最大: {analysis['quality_metrics']['weight_stats']['max']}")

    # 4. SO8Tトレーニング適合性評価
    print("\n=== SO8Tトレーニング適合性評価 ===")

    # 最小要件チェック (50,000サンプル)
    min_samples_required = 50000
    sample_sufficiency = total_samples >= min_samples_required

    # NSFWデータ比率チェック (検知目的)
    nsfw_ratio = (nsfw_erotic + nsfw_violence) / total_samples
    nsfw_sufficient = nsfw_ratio >= 0.1  # 最低10%はNSFWデータが必要

    # 品質スコアチェック
    avg_weight = sum(weights) / len(weights)
    quality_sufficient = avg_weight >= 0.8

    # テキスト多様性チェック
    unique_texts = len(set(item.get('text', '') for item in data))
    diversity_ratio = unique_texts / total_samples

    analysis["content_analysis"] = {
        "sample_sufficiency": sample_sufficiency,
        "nsfw_sufficient": nsfw_sufficient,
        "quality_sufficient": quality_sufficient,
        "diversity_ratio": diversity_ratio,
        "min_samples_required": min_samples_required
    }

    print(f"サンプル数充足: {'✅' if sample_sufficiency else '❌'} ({total_samples}/{min_samples_required})")
    print(".1f")
    print(".1f")
    print(".1f")
    print(".1f")

    # 総合評価
    print("\n=== 総合評価 ===")

    overall_score = 0
    if sample_sufficiency: overall_score += 40
    if nsfw_sufficient: overall_score += 25
    if quality_sufficient: overall_score += 20
    if diversity_ratio >= 0.95: overall_score += 15

    analysis["overall_assessment"] = {
        "overall_score": overall_score,
        "grade": "A" if overall_score >= 90 else "B" if overall_score >= 75 else "C" if overall_score >= 60 else "D" if overall_score >= 40 else "F",
        "recommendations": []
    }

    if not sample_sufficiency:
        analysis["overall_assessment"]["recommendations"].append("データセットサイズが不十分です。最低50,000サンプルが必要です。")
    if not nsfw_sufficient:
        analysis["overall_assessment"]["recommendations"].append("NSFWデータの割合が低いです。検知目的で最低10%のNSFWデータが必要です。")
    if not quality_sufficient:
        analysis["overall_assessment"]["recommendations"].append("平均品質スコアが低いです。QCプロセスを強化してください。")
    if diversity_ratio < 0.95:
        analysis["overall_assessment"]["recommendations"].append("テキストの多様性が不十分です。重複データを除去してください。")

    print(f"総合スコア: {overall_score}/100")
    print(f"グレード: {analysis['overall_assessment']['grade']}")

    if analysis["overall_assessment"]["recommendations"]:
        print("\n改善推奨事項:")
        for rec in analysis["overall_assessment"]["recommendations"]:
            print(f"  - {rec}")

    return analysis

def save_analysis_report(analysis: Dict[str, Any], output_path: str):
    """分析レポートを保存"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"\n分析レポートを保存しました: {output_path}")
    except Exception as e:
        print(f"レポート保存エラー: {e}")

if __name__ == "__main__":
    import sys

    # コマンドライン引数からデータセットパスを取得
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = "data/aegis_v2_mathematical_enhanced_dataset.jsonl"

    # 品質分析実行
    analysis = analyze_dataset_quality(dataset_path)

    if analysis:
        # レポート保存
        output_path = f"dataset_quality_analysis_{Path(dataset_path).stem}.json"
        save_analysis_report(analysis, output_path)
