#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/thinking形式データセットをペア比較形式に変換

既存の/thinking形式データセットをPPO学習用のペア比較形式（chosen/rejected）に変換
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import random

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "convert_thinking_to_pairwise.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def extract_thinking_and_final(text: str) -> Dict[str, str]:
    """/thinking形式から思考ステップと最終回答を抽出"""
    thinking = ""
    final = ""
    
    if not text:
        return {"thinking": "", "final": ""}
    
    # XML形式の四重推論タグを抽出
    import re
    
    # <think-task>, <think-safety>, <think-policy>を抽出
    think_task = re.findall(r'<think-task>(.*?)</think-task>', text, re.DOTALL)
    think_safety = re.findall(r'<think-safety>(.*?)</think-safety>', text, re.DOTALL)
    think_policy = re.findall(r'<think-policy>(.*?)</think-policy>', text, re.DOTALL)
    
    if think_task or think_safety or think_policy:
        # 四重推論形式
        thinking_parts = []
        if think_task:
            thinking_parts.append(f"Task: {think_task[0].strip()}")
        if think_safety:
            thinking_parts.append(f"Safety: {think_safety[0].strip()}")
        if think_policy:
            thinking_parts.append(f"Policy: {think_policy[0].strip()}")
        thinking = "\n".join(thinking_parts)
        
        # <final>を抽出
        final_matches = re.findall(r'<final>(.*?)</final>', text, re.DOTALL)
        if final_matches:
            final = final_matches[0].strip()
    
    # マークダウン形式（# 思考ステップ / # 最終回答）
    if not thinking and "# 思考ステップ" in text:
        parts = text.split("# 思考ステップ")
        if len(parts) > 1:
            thinking_part = parts[1].split("# 最終回答")[0] if "# 最終回答" in parts[1] else parts[1]
            thinking = thinking_part.strip()
    
    if not final and "# 最終回答" in text:
        parts = text.split("# 最終回答")
        if len(parts) > 1:
            final = parts[1].strip()
    
    # 単純な<think>タグ
    if not thinking and "<think>" in text:
        think_matches = re.findall(r'<think>(.*?)</think>', text, re.DOTALL)
        if think_matches:
            thinking = "\n".join(think_matches).strip()
    
    # フォールバック: テキスト全体を使用
    if not thinking and not final:
        # テキストの前半を思考、後半を最終回答とする
        if len(text) > 200:
            thinking = text[:len(text)//2]
            final = text[len(text)//2:]
        else:
            thinking = f"この問題について考えます。{text[:100]}"
            final = text
    
    return {"thinking": thinking, "final": final}


def evaluate_quality(thinking: str, final: str) -> float:
    """品質スコアを評価"""
    score = 0.0
    
    # 思考ステップの存在と詳細度
    if thinking and len(thinking.strip()) > 50:
        score += 0.3
    if thinking and len(thinking.strip()) > 200:
        score += 0.2
    
    # 最終回答の存在
    if final and len(final.strip()) > 20:
        score += 0.3
    if final and len(final.strip()) > 100:
        score += 0.2
    
    return min(score, 1.0)


def convert_thinking_to_pairwise(
    input_file: Path,
    output_file: Path,
    min_quality: float = 0.5,
    num_rejected_per_chosen: int = 1
) -> int:
    """
    /thinking形式データセットをペア比較形式に変換
    
    Args:
        input_file: 入力ファイル（/thinking形式）
        output_file: 出力ファイル（ペア比較形式）
        min_quality: 最小品質スコア
        num_rejected_per_chosen: chosenサンプルあたりのrejectedサンプル数
    
    Returns:
        変換されたサンプル数
    """
    samples = []
    
    # 入力ファイルを読み込み
    logger.info(f"[LOAD] Loading dataset from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    logger.warning(f"[WARNING] Line {line_num}: JSON decode error: {e}")
                    continue
    
    logger.info(f"[LOAD] Loaded {len(samples)} samples")
    
    # ペア比較形式に変換
    pairwise_samples = []
    
    for i, sample in enumerate(samples):
        # プロンプトを取得（複数の可能性を試す）
        prompt = (
            sample.get("instruction") or 
            sample.get("prompt") or 
            sample.get("query") or
            sample.get("input") or
            ""
        )
        
        # 出力を取得（複数の可能性を試す）
        output = (
            sample.get("output") or
            sample.get("response") or
            sample.get("text") or
            ""
        )
        
        # プロンプトまたは出力がない場合はスキップ
        if not prompt and not output:
            continue
        
        # プロンプトがない場合は、outputから生成を試みる
        if not prompt:
            # outputの最初の部分をプロンプトとして使用
            if output:
                prompt = output.split("\n")[0][:200] if "\n" in output else output[:200]
            else:
                continue
        
        # 出力がない場合はスキップ
        if not output:
            continue
        
        # 思考ステップと最終回答を抽出
        parts = extract_thinking_and_final(output)
        thinking = parts.get("thinking", "")
        final = parts.get("final", output)
        
        # 品質スコアを評価
        quality = evaluate_quality(thinking, final)
        
        if quality < min_quality:
            continue
        
        # chosenサンプルを作成（四重推論形式）
        # 既に四重推論形式の場合はそのまま使用、そうでない場合は変換
        if "<think-task>" in output or "<think-safety>" in output:
            # 既に四重推論形式
            chosen = output
        else:
            # 四重推論形式に変換
            chosen = f"<think-task>{thinking[:500] if thinking else 'タスクを分析します。'}</think-task>"
            chosen += f"<think-safety>安全性を確認します。</think-safety>"
            chosen += f"<think-policy>ポリシーを確認します。</think-policy>"
            chosen += f"<final>{final}</final>"
        
        # rejectedサンプルを生成（品質の低いバージョン）
        for j in range(num_rejected_per_chosen):
            # ランダムに別のサンプルからrejectedを選択、または簡略化したバージョンを作成
            if i + j + 1 < len(samples):
                rejected_sample = samples[i + j + 1]
                rejected_output = (
                    rejected_sample.get("output") or
                    rejected_sample.get("response") or
                    rejected_sample.get("text") or
                    ""
                )
                if rejected_output:
                    rejected_parts = extract_thinking_and_final(rejected_output)
                    rejected_final = rejected_parts.get("final", rejected_output[:200])
                else:
                    rejected_final = final[:100] + "..." if len(final) > 100 else final
            else:
                # 簡略化したバージョン
                rejected_final = final[:100] + "..." if len(final) > 100 else final
            
            rejected = f"<think-task>簡略化された分析</think-task>"
            rejected += f"<think-safety>安全性確認</think-safety>"
            rejected += f"<think-policy>ポリシー確認</think-policy>"
            rejected += f"<final>{rejected_final}</final>"
            
            # 四値分類ラベル（デフォルト: ALLOW）
            four_class_label = (
                sample.get("four_class_label") or
                sample.get("label") or
                sample.get("safety_label") or
                "ALLOW"
            )
            
            pairwise_samples.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "four_class_label": four_class_label,
                "quality_score": quality,
                "source": sample.get("source", "converted"),
                "created_at": datetime.now().isoformat()
            })
    
    logger.info(f"[CONVERT] Converted {len(samples)} -> {len(pairwise_samples)} pairwise samples")
    
    # 出力ファイルに保存
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in pairwise_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"[SAVE] Saved {len(pairwise_samples)} pairwise samples to {output_file}")
    
    # 統計情報
    if pairwise_samples:
        avg_quality = sum(s.get("quality_score", 0.0) for s in pairwise_samples) / len(pairwise_samples)
        logger.info(f"[STATS] Average quality score: {avg_quality:.2f}")
        
        label_counts = {}
        for s in pairwise_samples:
            label = s.get("four_class_label", "ALLOW")
            label_counts[label] = label_counts.get(label, 0) + 1
        logger.info(f"[STATS] Label distribution: {label_counts}")
    
    return len(pairwise_samples)


def main():
    parser = argparse.ArgumentParser(
        description="Convert /thinking format dataset to pairwise format for PPO training"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input dataset file (/thinking format, JSONL)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset file (pairwise format, JSONL)"
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.5,
        help="Minimum quality score threshold (default: 0.5)"
    )
    parser.add_argument(
        "--num-rejected",
        type=int,
        default=1,
        help="Number of rejected samples per chosen sample (default: 1)"
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        logger.error(f"[ERROR] Input file not found: {args.input}")
        return
    
    if args.input.stat().st_size == 0:
        logger.error(f"[ERROR] Input file is empty: {args.input}")
        return
    
    logger.info("="*80)
    logger.info("Converting /thinking format to pairwise format")
    logger.info("="*80)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Min quality: {args.min_quality}")
    logger.info(f"Num rejected per chosen: {args.num_rejected}")
    logger.info("="*80)
    
    count = convert_thinking_to_pairwise(
        input_file=args.input,
        output_file=args.output,
        min_quality=args.min_quality,
        num_rejected_per_chosen=args.num_rejected
    )
    
    logger.info("="*80)
    logger.info(f"[SUCCESS] Conversion completed: {count} pairwise samples")
    logger.info("="*80)


if __name__ == "__main__":
    main()

