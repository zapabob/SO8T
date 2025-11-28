#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/think形式SFTデータセット作成スクリプト

既存の4値分類データセットやドメイン知識データセットから
Phi-3.5チャットテンプレート準拠の/think形式（「思考ステップ→最終回答」）SFTデータを生成
"""

import json
import argparse
import logging
import re
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sys

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/create_thinking_sft_dataset.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ThinkingDatasetQualityEvaluator:
    """/thinkingデータセットの品質評価クラス"""
    
    def __init__(self):
        """品質評価器を初期化"""
        # 論理接続詞のリスト
        self.logical_connectors = [
            "したがって", "ゆえに", "つまり", "すなわち", "なぜなら", "なぜかというと",
            "そのため", "その結果", "このため", "この結果", "よって", "従って",
            "まず", "次に", "さらに", "また", "そして", "また", "さらに", "加えて",
            "一方で", "しかし", "ただし", "しかしながら", "とはいえ", "もっとも"
        ]
    
    def evaluate_thinking_quality(self, thinking_steps: str, final_answer: str) -> Dict[str, float]:
        """
        思考ステップと最終回答の品質を評価
        
        Args:
            thinking_steps: 思考ステップのテキスト
            final_answer: 最終回答のテキスト
        
        Returns:
            品質スコアの辞書
        """
        scores = {}
        
        # 1. 論理性評価（思考ステップの構造、論理接続詞の使用）
        logicality_score = self._evaluate_logicality(thinking_steps)
        scores['logicality'] = logicality_score
        
        # 2. 正確性評価（最終回答の妥当性）
        correctness_score = self._evaluate_correctness(final_answer)
        scores['correctness'] = correctness_score
        
        # 3. 深さ評価（思考ステップの詳細度）
        depth_score = self._evaluate_depth(thinking_steps)
        scores['depth'] = depth_score
        
        # 4. 多様性評価（思考パターンの多様性）
        diversity_score = self._evaluate_diversity(thinking_steps)
        scores['diversity'] = diversity_score
        
        # 総合スコア（重み付き平均）
        scores['overall'] = (
            0.3 * logicality_score +
            0.3 * correctness_score +
            0.2 * depth_score +
            0.2 * diversity_score
        )
        
        return scores
    
    def _evaluate_logicality(self, thinking_steps: str) -> float:
        """論理性評価"""
        if not thinking_steps:
            return 0.0
        
        score = 0.0
        
        # 論理接続詞の使用
        connector_count = sum(1 for connector in self.logical_connectors if connector in thinking_steps)
        score += min(connector_count / 3.0, 1.0) * 0.4
        
        # 文の構造（句読点の使用）
        punctuation_count = thinking_steps.count('。') + thinking_steps.count('、')
        if len(thinking_steps) > 0:
            score += min(punctuation_count / (len(thinking_steps) / 50), 1.0) * 0.3
        
        # 思考ステップの明確性（「#」や番号付きリストの使用）
        if re.search(r'#|^\d+[\.\)]', thinking_steps, re.MULTILINE):
            score += 0.3
        
        return min(score, 1.0)
    
    def _evaluate_correctness(self, final_answer: str) -> float:
        """正確性評価（簡易版）"""
        if not final_answer:
            return 0.0
        
        score = 1.0
        
        # 空でないことを確認
        if len(final_answer.strip()) < 5:
            score -= 0.5
        
        # 明らかに不適切な内容をチェック（簡易版）
        inappropriate_patterns = ['[削除]', '[エラー]', 'None', 'null', 'undefined']
        if any(pattern in final_answer for pattern in inappropriate_patterns):
            score -= 0.5
        
        return max(score, 0.0)
    
    def _evaluate_depth(self, thinking_steps: str) -> float:
        """深さ評価（思考ステップの詳細度）"""
        if not thinking_steps:
            return 0.0
        
        # 文字数による評価
        length_score = min(len(thinking_steps) / 200.0, 1.0) * 0.5
        
        # 文の数による評価
        sentence_count = thinking_steps.count('。') + thinking_steps.count('\n')
        sentence_score = min(sentence_count / 5.0, 1.0) * 0.5
        
        return length_score + sentence_score
    
    def _evaluate_diversity(self, thinking_steps: str) -> float:
        """多様性評価（思考パターンの多様性）"""
        if not thinking_steps:
            return 0.0
        
        # 異なる思考パターンの使用（簡易版）
        # 異なる接続詞の種類数
        unique_connectors = sum(1 for connector in self.logical_connectors if connector in thinking_steps)
        diversity_score = min(unique_connectors / 5.0, 1.0)
        
        return diversity_score
    
    def filter_by_quality(self, samples: List[Dict], min_score: float = 0.7) -> List[Dict]:
        """
        品質スコアに基づいてフィルタリング
        
        Args:
            samples: サンプルのリスト
            min_score: 最小品質スコア
        
        Returns:
            フィルタリングされたサンプルのリスト
        """
        filtered_samples = []
        
        for sample in samples:
            # 思考ステップと最終回答を抽出
            output = sample.get("output", "")
            if "# 思考ステップ" in output and "# 最終回答" in output:
                thinking_steps = output.split("# 思考ステップ")[1].split("# 最終回答")[0].strip()
                final_answer = output.split("# 最終回答")[1].strip()
                
                # 品質評価
                quality_scores = self.evaluate_thinking_quality(thinking_steps, final_answer)
                
                # 品質スコアをサンプルに追加
                sample["quality_scores"] = quality_scores
                
                # フィルタリング
                if quality_scores.get("overall", 0.0) >= min_score:
                    filtered_samples.append(sample)
        
        return filtered_samples


def format_phi35_chat_template(
    system_message: str,
    user_message: str,
    assistant_message: str
) -> str:
    """
    Phi-3.5チャットテンプレート形式にフォーマット
    
    Args:
        system_message: システムメッセージ
        user_message: ユーザーメッセージ
        assistant_message: アシスタントメッセージ（思考ステップ+最終回答を含む）
    
    Returns:
        フォーマット済みテキスト
    """
    template = f"<|system|>\n{system_message}<|end|>\n<|user|>\n{user_message}<|end|>\n<|assistant|>\n{assistant_message}<|end|>"
    return template


def create_thinking_output(thinking_steps: str, final_answer: str) -> str:
    """
    思考ステップと最終回答を結合
    
    Args:
        thinking_steps: 思考ステップのテキスト
        final_answer: 最終回答のテキスト
    
    Returns:
        結合されたテキスト
    """
    return f"# 思考ステップ\n{thinking_steps}\n\n# 最終回答\n{final_answer}"


def convert_sample_to_thinking_format(
    sample: Dict[str, Any],
    default_system_message: str = "あなたは慎重に考えるAIアシスタントです。まず「# 思考ステップ」で考えを整理し、そのあとに「# 最終回答」でユーザーへの短い答えだけを出してください。"
) -> Optional[Dict[str, Any]]:
    """
    サンプルを/think形式に変換
    
    Args:
        sample: 入力サンプル
        default_system_message: デフォルトのシステムメッセージ
    
    Returns:
        変換されたサンプル（Noneの場合はスキップ）
    """
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output = sample.get("output", "")
    text = sample.get("text", "")
    
    # textフィールドがある場合は、それをinstructionとして使用
    if not instruction and not input_text and text:
        # textフィールドからinstructionを生成
        # タイトルやキーワードがあれば使用、なければtextの最初の部分を使用
        title = sample.get("title", "")
        keyword = sample.get("keyword", "")
        if title:
            instruction = f"{title}について説明してください。"
        elif keyword:
            instruction = f"{keyword}について説明してください。"
        else:
            # textの最初の200文字をinstructionとして使用
            instruction = f"以下の内容について説明してください。\n\n{text[:200]}..."
        input_text = text
    
    # 既に/think形式の場合はそのまま使用
    if "# 思考ステップ" in output and "# 最終回答" in output:
        thinking_steps = output.split("# 思考ステップ")[1].split("# 最終回答")[0].strip()
        final_answer = output.split("# 最終回答")[1].strip()
    elif output:
        # 既存の出力から思考ステップと最終回答を生成
        # 簡易実装: 出力の前半を思考ステップ、後半を最終回答とする
        if len(output) > 100:
            thinking_steps = output[:len(output)//2]
            final_answer = output[len(output)//2:]
        else:
            thinking_steps = f"この問題について考えます。{output[:50]}"
            final_answer = output
    elif text:
        # textフィールドから思考ステップと最終回答を生成
        # textの前半を思考ステップ、後半を最終回答とする
        if len(text) > 200:
            thinking_steps = f"この内容について分析します。\n\n{text[:len(text)//2]}"
            final_answer = text[len(text)//2:]
        else:
            thinking_steps = f"この内容について分析します。\n\n{text[:100]}"
            final_answer = text[100:] if len(text) > 100 else text
    else:
        logger.warning("No output or text found, skipping sample")
        return None
    
    # ユーザーメッセージを構築
    if instruction and input_text:
        user_message = f"{instruction}\n\n{input_text}"
    elif instruction:
        user_message = instruction
    elif input_text:
        user_message = input_text
    else:
        logger.warning("No instruction or input found, skipping sample")
        return None
    
    # アシスタントメッセージを構築
    assistant_message = create_thinking_output(thinking_steps, final_answer)
    
    # システムメッセージを取得（サンプルに含まれている場合はそれを使用）
    system_message = sample.get("system", default_system_message)
    
    # チャットテンプレート形式にフォーマット
    formatted_text = format_phi35_chat_template(
        system_message=system_message,
        user_message=user_message,
        assistant_message=assistant_message
    )
    
    # 新しいサンプルを作成
    new_sample = {
        "instruction": instruction,
        "input": input_text,
        "output": formatted_text,
    }
    
    # メタデータを保持
    if "four_class_label" in sample:
        new_sample["four_class_label"] = sample["four_class_label"]
    if "domain_label" in sample:
        new_sample["domain_label"] = sample["domain_label"]
    if "safety_label" in sample:
        new_sample["safety_label"] = sample["safety_label"]
    if "system" in sample:
        new_sample["system"] = sample["system"]
    
    return new_sample


def convert_dataset_to_thinking_format(
    input_file: Path,
    output_file: Path,
    default_system_message: Optional[str] = None
) -> int:
    """
    データセットを/think形式に変換
    
    Args:
        input_file: 入力データセットファイル（JSONL形式）
        output_file: 出力データセットファイル（JSONL形式）
        default_system_message: デフォルトのシステムメッセージ
    
    Returns:
        変換されたサンプル数
    """
    if default_system_message is None:
        default_system_message = "あなたは慎重に考えるAIアシスタントです。まず「# 思考ステップ」で考えを整理し、そのあとに「# 最終回答」でユーザーへの短い答えだけを出してください。"
    
    logger.info(f"Loading dataset from: {input_file}")
    converted_count = 0
    skipped_count = 0
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as f, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                converted_sample = convert_sample_to_thinking_format(
                    sample,
                    default_system_message=default_system_message
                )
                
                if converted_sample is None:
                    skipped_count += 1
                    continue
                
                out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                converted_count += 1
                
                if converted_count % 1000 == 0:
                    logger.info(f"Converted {converted_count} samples...")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON decode error: {e}")
                skipped_count += 1
                continue
            except Exception as e:
                logger.warning(f"Line {line_num}: Error: {e}")
                skipped_count += 1
                continue
    
    logger.info(f"Conversion complete: {converted_count} samples converted, {skipped_count} skipped")
    logger.info(f"Saved to: {output_file}")
    
    return converted_count


def merge_multiple_datasets(
    input_files: List[Path],
    output_file: Path,
    default_system_message: Optional[str] = None
) -> int:
    """
    複数のデータセットをマージして/think形式に変換
    
    Args:
        input_files: 入力データセットファイルのリスト
        output_file: 出力データセットファイル
        default_system_message: デフォルトのシステムメッセージ
    
    Returns:
        マージされたサンプル数
    """
    total_count = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for input_file in input_files:
            logger.info(f"Processing: {input_file}")
            count = 0
            
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sample = json.loads(line)
                        converted_sample = convert_sample_to_thinking_format(
                            sample,
                            default_system_message=default_system_message
                        )
                        
                        if converted_sample is None:
                            continue
                        
                        out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                        count += 1
                        total_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error processing sample: {e}")
                        continue
            
            logger.info(f"  Processed {count} samples from {input_file}")
    
    logger.info(f"Merged {total_count} total samples")
    logger.info(f"Saved to: {output_file}")
    
    return total_count


def expand_dataset_gradually(
    base_dataset: Path,
    target_sizes: List[int],
    quality_threshold: float = 0.7,
    output_dir: Optional[Path] = None
) -> List[Path]:
    """
    データセットを段階的に拡張
    
    Args:
        base_dataset: ベースデータセットファイル
        target_sizes: 目標サイズのリスト（例: [5000, 10000, 25000, 50000]）
        quality_threshold: 品質スコアの閾値
        output_dir: 出力ディレクトリ（Noneの場合はbase_datasetと同じディレクトリ）
    
    Returns:
        各段階のデータセットファイルパスのリスト
    """
    if output_dir is None:
        output_dir = base_dataset.parent
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # ベースデータセットを読み込み
    logger.info(f"Loading base dataset from: {base_dataset}")
    all_samples = []
    with open(base_dataset, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                all_samples.append(sample)
            except json.JSONDecodeError:
                continue
    
    logger.info(f"Loaded {len(all_samples)} samples from base dataset")
    
    # 品質評価器を初期化
    evaluator = ThinkingDatasetQualityEvaluator()
    
    # 品質評価とフィルタリング
    logger.info(f"Evaluating quality with threshold: {quality_threshold}")
    quality_samples = evaluator.filter_by_quality(all_samples, min_score=quality_threshold)
    logger.info(f"Quality-filtered samples: {len(quality_samples)} / {len(all_samples)}")
    
    # 各段階でデータセットを拡張
    output_files = []
    current_samples = quality_samples.copy()
    
    for target_size in target_sizes:
        logger.info(f"Expanding to {target_size} samples...")
        
        # 目標サイズに達するまでサンプルを追加
        if len(current_samples) < target_size:
            # 不足分を既存サンプルからランダムに複製（簡易版）
            # 実際の実装では、新しいデータソースから追加することを推奨
            needed = target_size - len(current_samples)
            additional_samples = random.sample(quality_samples, min(needed, len(quality_samples)))
            current_samples.extend(additional_samples)
        
        # 目標サイズに合わせてサンプリング
        if len(current_samples) > target_size:
            current_samples = random.sample(current_samples, target_size)
        
        # 出力ファイル名を生成
        output_file = output_dir / f"{base_dataset.stem}_expanded_{target_size}.jsonl"
        
        # データセットを保存
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in current_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved expanded dataset: {output_file} ({len(current_samples)} samples)")
        output_files.append(output_file)
    
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Create /think format SFT dataset for Borea-Phi-3.5"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input dataset file (JSONL format)"
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        help="Multiple input dataset files (JSONL format) for merging"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset file (JSONL format)"
    )
    parser.add_argument(
        "--system-message",
        type=str,
        default=None,
        help="Custom system message (default: built-in thinking prompt)"
    )
    parser.add_argument(
        "--quality-filter",
        action="store_true",
        help="Enable quality filtering"
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.7,
        help="Minimum quality score for filtering (default: 0.7)"
    )
    parser.add_argument(
        "--expand-gradually",
        action="store_true",
        help="Expand dataset gradually (5000 -> 10000 -> 25000 -> 50000)"
    )
    parser.add_argument(
        "--target-sizes",
        type=int,
        nargs="+",
        default=[5000, 10000, 25000, 50000],
        help="Target sizes for gradual expansion (default: 5000 10000 25000 50000)"
    )
    
    args = parser.parse_args()
    
    if args.expand_gradually:
        # 段階的拡張
        if not args.input:
            parser.error("--expand-gradually requires --input")
        
        output_files = expand_dataset_gradually(
            base_dataset=args.input,
            target_sizes=args.target_sizes,
            quality_threshold=args.min_quality_score,
            output_dir=args.output.parent if args.output else None
        )
        logger.info(f"[SUCCESS] Expanded dataset to {len(output_files)} stages")
        for output_file in output_files:
            logger.info(f"  - {output_file}")
    
    elif args.inputs:
        # 複数ファイルをマージ
        count = merge_multiple_datasets(
            input_files=args.inputs,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Merged and converted {count} samples")
    elif args.input:
        # 単一ファイルを変換
        count = convert_dataset_to_thinking_format(
            input_file=args.input,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Converted {count} samples")
    else:
        parser.error("Either --input or --inputs must be specified")


if __name__ == "__main__":
    main()





    
    with open(input_file, 'r', encoding='utf-8') as f, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                converted_sample = convert_sample_to_thinking_format(
                    sample,
                    default_system_message=default_system_message
                )
                
                if converted_sample is None:
                    skipped_count += 1
                    continue
                
                out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                converted_count += 1
                
                if converted_count % 1000 == 0:
                    logger.info(f"Converted {converted_count} samples...")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON decode error: {e}")
                skipped_count += 1
                continue
            except Exception as e:
                logger.warning(f"Line {line_num}: Error: {e}")
                skipped_count += 1
                continue
    
    logger.info(f"Conversion complete: {converted_count} samples converted, {skipped_count} skipped")
    logger.info(f"Saved to: {output_file}")
    
    return converted_count


def merge_multiple_datasets(
    input_files: List[Path],
    output_file: Path,
    default_system_message: Optional[str] = None
) -> int:
    """
    複数のデータセットをマージして/think形式に変換
    
    Args:
        input_files: 入力データセットファイルのリスト
        output_file: 出力データセットファイル
        default_system_message: デフォルトのシステムメッセージ
    
    Returns:
        マージされたサンプル数
    """
    total_count = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for input_file in input_files:
            logger.info(f"Processing: {input_file}")
            count = 0
            
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sample = json.loads(line)
                        converted_sample = convert_sample_to_thinking_format(
                            sample,
                            default_system_message=default_system_message
                        )
                        
                        if converted_sample is None:
                            continue
                        
                        out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                        count += 1
                        total_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error processing sample: {e}")
                        continue
            
            logger.info(f"  Processed {count} samples from {input_file}")
    
    logger.info(f"Merged {total_count} total samples")
    logger.info(f"Saved to: {output_file}")
    
    return total_count


def expand_dataset_gradually(
    base_dataset: Path,
    target_sizes: List[int],
    quality_threshold: float = 0.7,
    output_dir: Optional[Path] = None
) -> List[Path]:
    """
    データセットを段階的に拡張
    
    Args:
        base_dataset: ベースデータセットファイル
        target_sizes: 目標サイズのリスト（例: [5000, 10000, 25000, 50000]）
        quality_threshold: 品質スコアの閾値
        output_dir: 出力ディレクトリ（Noneの場合はbase_datasetと同じディレクトリ）
    
    Returns:
        各段階のデータセットファイルパスのリスト
    """
    if output_dir is None:
        output_dir = base_dataset.parent
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # ベースデータセットを読み込み
    logger.info(f"Loading base dataset from: {base_dataset}")
    all_samples = []
    with open(base_dataset, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                all_samples.append(sample)
            except json.JSONDecodeError:
                continue
    
    logger.info(f"Loaded {len(all_samples)} samples from base dataset")
    
    # 品質評価器を初期化
    evaluator = ThinkingDatasetQualityEvaluator()
    
    # 品質評価とフィルタリング
    logger.info(f"Evaluating quality with threshold: {quality_threshold}")
    quality_samples = evaluator.filter_by_quality(all_samples, min_score=quality_threshold)
    logger.info(f"Quality-filtered samples: {len(quality_samples)} / {len(all_samples)}")
    
    # 各段階でデータセットを拡張
    output_files = []
    current_samples = quality_samples.copy()
    
    for target_size in target_sizes:
        logger.info(f"Expanding to {target_size} samples...")
        
        # 目標サイズに達するまでサンプルを追加
        if len(current_samples) < target_size:
            # 不足分を既存サンプルからランダムに複製（簡易版）
            # 実際の実装では、新しいデータソースから追加することを推奨
            needed = target_size - len(current_samples)
            additional_samples = random.sample(quality_samples, min(needed, len(quality_samples)))
            current_samples.extend(additional_samples)
        
        # 目標サイズに合わせてサンプリング
        if len(current_samples) > target_size:
            current_samples = random.sample(current_samples, target_size)
        
        # 出力ファイル名を生成
        output_file = output_dir / f"{base_dataset.stem}_expanded_{target_size}.jsonl"
        
        # データセットを保存
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in current_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved expanded dataset: {output_file} ({len(current_samples)} samples)")
        output_files.append(output_file)
    
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Create /think format SFT dataset for Borea-Phi-3.5"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input dataset file (JSONL format)"
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        help="Multiple input dataset files (JSONL format) for merging"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset file (JSONL format)"
    )
    parser.add_argument(
        "--system-message",
        type=str,
        default=None,
        help="Custom system message (default: built-in thinking prompt)"
    )
    parser.add_argument(
        "--quality-filter",
        action="store_true",
        help="Enable quality filtering"
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.7,
        help="Minimum quality score for filtering (default: 0.7)"
    )
    parser.add_argument(
        "--expand-gradually",
        action="store_true",
        help="Expand dataset gradually (5000 -> 10000 -> 25000 -> 50000)"
    )
    parser.add_argument(
        "--target-sizes",
        type=int,
        nargs="+",
        default=[5000, 10000, 25000, 50000],
        help="Target sizes for gradual expansion (default: 5000 10000 25000 50000)"
    )
    
    args = parser.parse_args()
    
    if args.expand_gradually:
        # 段階的拡張
        if not args.input:
            parser.error("--expand-gradually requires --input")
        
        output_files = expand_dataset_gradually(
            base_dataset=args.input,
            target_sizes=args.target_sizes,
            quality_threshold=args.min_quality_score,
            output_dir=args.output.parent if args.output else None
        )
        logger.info(f"[SUCCESS] Expanded dataset to {len(output_files)} stages")
        for output_file in output_files:
            logger.info(f"  - {output_file}")
    
    elif args.inputs:
        # 複数ファイルをマージ
        count = merge_multiple_datasets(
            input_files=args.inputs,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Merged and converted {count} samples")
    elif args.input:
        # 単一ファイルを変換
        count = convert_dataset_to_thinking_format(
            input_file=args.input,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Converted {count} samples")
    else:
        parser.error("Either --input or --inputs must be specified")


if __name__ == "__main__":
    main()





    
    with open(input_file, 'r', encoding='utf-8') as f, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                converted_sample = convert_sample_to_thinking_format(
                    sample,
                    default_system_message=default_system_message
                )
                
                if converted_sample is None:
                    skipped_count += 1
                    continue
                
                out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                converted_count += 1
                
                if converted_count % 1000 == 0:
                    logger.info(f"Converted {converted_count} samples...")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON decode error: {e}")
                skipped_count += 1
                continue
            except Exception as e:
                logger.warning(f"Line {line_num}: Error: {e}")
                skipped_count += 1
                continue
    
    logger.info(f"Conversion complete: {converted_count} samples converted, {skipped_count} skipped")
    logger.info(f"Saved to: {output_file}")
    
    return converted_count


def merge_multiple_datasets(
    input_files: List[Path],
    output_file: Path,
    default_system_message: Optional[str] = None
) -> int:
    """
    複数のデータセットをマージして/think形式に変換
    
    Args:
        input_files: 入力データセットファイルのリスト
        output_file: 出力データセットファイル
        default_system_message: デフォルトのシステムメッセージ
    
    Returns:
        マージされたサンプル数
    """
    total_count = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for input_file in input_files:
            logger.info(f"Processing: {input_file}")
            count = 0
            
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sample = json.loads(line)
                        converted_sample = convert_sample_to_thinking_format(
                            sample,
                            default_system_message=default_system_message
                        )
                        
                        if converted_sample is None:
                            continue
                        
                        out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                        count += 1
                        total_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error processing sample: {e}")
                        continue
            
            logger.info(f"  Processed {count} samples from {input_file}")
    
    logger.info(f"Merged {total_count} total samples")
    logger.info(f"Saved to: {output_file}")
    
    return total_count


def expand_dataset_gradually(
    base_dataset: Path,
    target_sizes: List[int],
    quality_threshold: float = 0.7,
    output_dir: Optional[Path] = None
) -> List[Path]:
    """
    データセットを段階的に拡張
    
    Args:
        base_dataset: ベースデータセットファイル
        target_sizes: 目標サイズのリスト（例: [5000, 10000, 25000, 50000]）
        quality_threshold: 品質スコアの閾値
        output_dir: 出力ディレクトリ（Noneの場合はbase_datasetと同じディレクトリ）
    
    Returns:
        各段階のデータセットファイルパスのリスト
    """
    if output_dir is None:
        output_dir = base_dataset.parent
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # ベースデータセットを読み込み
    logger.info(f"Loading base dataset from: {base_dataset}")
    all_samples = []
    with open(base_dataset, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                all_samples.append(sample)
            except json.JSONDecodeError:
                continue
    
    logger.info(f"Loaded {len(all_samples)} samples from base dataset")
    
    # 品質評価器を初期化
    evaluator = ThinkingDatasetQualityEvaluator()
    
    # 品質評価とフィルタリング
    logger.info(f"Evaluating quality with threshold: {quality_threshold}")
    quality_samples = evaluator.filter_by_quality(all_samples, min_score=quality_threshold)
    logger.info(f"Quality-filtered samples: {len(quality_samples)} / {len(all_samples)}")
    
    # 各段階でデータセットを拡張
    output_files = []
    current_samples = quality_samples.copy()
    
    for target_size in target_sizes:
        logger.info(f"Expanding to {target_size} samples...")
        
        # 目標サイズに達するまでサンプルを追加
        if len(current_samples) < target_size:
            # 不足分を既存サンプルからランダムに複製（簡易版）
            # 実際の実装では、新しいデータソースから追加することを推奨
            needed = target_size - len(current_samples)
            additional_samples = random.sample(quality_samples, min(needed, len(quality_samples)))
            current_samples.extend(additional_samples)
        
        # 目標サイズに合わせてサンプリング
        if len(current_samples) > target_size:
            current_samples = random.sample(current_samples, target_size)
        
        # 出力ファイル名を生成
        output_file = output_dir / f"{base_dataset.stem}_expanded_{target_size}.jsonl"
        
        # データセットを保存
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in current_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved expanded dataset: {output_file} ({len(current_samples)} samples)")
        output_files.append(output_file)
    
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Create /think format SFT dataset for Borea-Phi-3.5"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input dataset file (JSONL format)"
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        help="Multiple input dataset files (JSONL format) for merging"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset file (JSONL format)"
    )
    parser.add_argument(
        "--system-message",
        type=str,
        default=None,
        help="Custom system message (default: built-in thinking prompt)"
    )
    parser.add_argument(
        "--quality-filter",
        action="store_true",
        help="Enable quality filtering"
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.7,
        help="Minimum quality score for filtering (default: 0.7)"
    )
    parser.add_argument(
        "--expand-gradually",
        action="store_true",
        help="Expand dataset gradually (5000 -> 10000 -> 25000 -> 50000)"
    )
    parser.add_argument(
        "--target-sizes",
        type=int,
        nargs="+",
        default=[5000, 10000, 25000, 50000],
        help="Target sizes for gradual expansion (default: 5000 10000 25000 50000)"
    )
    
    args = parser.parse_args()
    
    if args.expand_gradually:
        # 段階的拡張
        if not args.input:
            parser.error("--expand-gradually requires --input")
        
        output_files = expand_dataset_gradually(
            base_dataset=args.input,
            target_sizes=args.target_sizes,
            quality_threshold=args.min_quality_score,
            output_dir=args.output.parent if args.output else None
        )
        logger.info(f"[SUCCESS] Expanded dataset to {len(output_files)} stages")
        for output_file in output_files:
            logger.info(f"  - {output_file}")
    
    elif args.inputs:
        # 複数ファイルをマージ
        count = merge_multiple_datasets(
            input_files=args.inputs,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Merged and converted {count} samples")
    elif args.input:
        # 単一ファイルを変換
        count = convert_dataset_to_thinking_format(
            input_file=args.input,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Converted {count} samples")
    else:
        parser.error("Either --input or --inputs must be specified")


if __name__ == "__main__":
    main()





    
    with open(input_file, 'r', encoding='utf-8') as f, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                converted_sample = convert_sample_to_thinking_format(
                    sample,
                    default_system_message=default_system_message
                )
                
                if converted_sample is None:
                    skipped_count += 1
                    continue
                
                out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                converted_count += 1
                
                if converted_count % 1000 == 0:
                    logger.info(f"Converted {converted_count} samples...")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON decode error: {e}")
                skipped_count += 1
                continue
            except Exception as e:
                logger.warning(f"Line {line_num}: Error: {e}")
                skipped_count += 1
                continue
    
    logger.info(f"Conversion complete: {converted_count} samples converted, {skipped_count} skipped")
    logger.info(f"Saved to: {output_file}")
    
    return converted_count


def merge_multiple_datasets(
    input_files: List[Path],
    output_file: Path,
    default_system_message: Optional[str] = None
) -> int:
    """
    複数のデータセットをマージして/think形式に変換
    
    Args:
        input_files: 入力データセットファイルのリスト
        output_file: 出力データセットファイル
        default_system_message: デフォルトのシステムメッセージ
    
    Returns:
        マージされたサンプル数
    """
    total_count = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for input_file in input_files:
            logger.info(f"Processing: {input_file}")
            count = 0
            
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sample = json.loads(line)
                        converted_sample = convert_sample_to_thinking_format(
                            sample,
                            default_system_message=default_system_message
                        )
                        
                        if converted_sample is None:
                            continue
                        
                        out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                        count += 1
                        total_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error processing sample: {e}")
                        continue
            
            logger.info(f"  Processed {count} samples from {input_file}")
    
    logger.info(f"Merged {total_count} total samples")
    logger.info(f"Saved to: {output_file}")
    
    return total_count


def expand_dataset_gradually(
    base_dataset: Path,
    target_sizes: List[int],
    quality_threshold: float = 0.7,
    output_dir: Optional[Path] = None
) -> List[Path]:
    """
    データセットを段階的に拡張
    
    Args:
        base_dataset: ベースデータセットファイル
        target_sizes: 目標サイズのリスト（例: [5000, 10000, 25000, 50000]）
        quality_threshold: 品質スコアの閾値
        output_dir: 出力ディレクトリ（Noneの場合はbase_datasetと同じディレクトリ）
    
    Returns:
        各段階のデータセットファイルパスのリスト
    """
    if output_dir is None:
        output_dir = base_dataset.parent
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # ベースデータセットを読み込み
    logger.info(f"Loading base dataset from: {base_dataset}")
    all_samples = []
    with open(base_dataset, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                all_samples.append(sample)
            except json.JSONDecodeError:
                continue
    
    logger.info(f"Loaded {len(all_samples)} samples from base dataset")
    
    # 品質評価器を初期化
    evaluator = ThinkingDatasetQualityEvaluator()
    
    # 品質評価とフィルタリング
    logger.info(f"Evaluating quality with threshold: {quality_threshold}")
    quality_samples = evaluator.filter_by_quality(all_samples, min_score=quality_threshold)
    logger.info(f"Quality-filtered samples: {len(quality_samples)} / {len(all_samples)}")
    
    # 各段階でデータセットを拡張
    output_files = []
    current_samples = quality_samples.copy()
    
    for target_size in target_sizes:
        logger.info(f"Expanding to {target_size} samples...")
        
        # 目標サイズに達するまでサンプルを追加
        if len(current_samples) < target_size:
            # 不足分を既存サンプルからランダムに複製（簡易版）
            # 実際の実装では、新しいデータソースから追加することを推奨
            needed = target_size - len(current_samples)
            additional_samples = random.sample(quality_samples, min(needed, len(quality_samples)))
            current_samples.extend(additional_samples)
        
        # 目標サイズに合わせてサンプリング
        if len(current_samples) > target_size:
            current_samples = random.sample(current_samples, target_size)
        
        # 出力ファイル名を生成
        output_file = output_dir / f"{base_dataset.stem}_expanded_{target_size}.jsonl"
        
        # データセットを保存
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in current_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved expanded dataset: {output_file} ({len(current_samples)} samples)")
        output_files.append(output_file)
    
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Create /think format SFT dataset for Borea-Phi-3.5"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input dataset file (JSONL format)"
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        help="Multiple input dataset files (JSONL format) for merging"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset file (JSONL format)"
    )
    parser.add_argument(
        "--system-message",
        type=str,
        default=None,
        help="Custom system message (default: built-in thinking prompt)"
    )
    parser.add_argument(
        "--quality-filter",
        action="store_true",
        help="Enable quality filtering"
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.7,
        help="Minimum quality score for filtering (default: 0.7)"
    )
    parser.add_argument(
        "--expand-gradually",
        action="store_true",
        help="Expand dataset gradually (5000 -> 10000 -> 25000 -> 50000)"
    )
    parser.add_argument(
        "--target-sizes",
        type=int,
        nargs="+",
        default=[5000, 10000, 25000, 50000],
        help="Target sizes for gradual expansion (default: 5000 10000 25000 50000)"
    )
    
    args = parser.parse_args()
    
    if args.expand_gradually:
        # 段階的拡張
        if not args.input:
            parser.error("--expand-gradually requires --input")
        
        output_files = expand_dataset_gradually(
            base_dataset=args.input,
            target_sizes=args.target_sizes,
            quality_threshold=args.min_quality_score,
            output_dir=args.output.parent if args.output else None
        )
        logger.info(f"[SUCCESS] Expanded dataset to {len(output_files)} stages")
        for output_file in output_files:
            logger.info(f"  - {output_file}")
    
    elif args.inputs:
        # 複数ファイルをマージ
        count = merge_multiple_datasets(
            input_files=args.inputs,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Merged and converted {count} samples")
    elif args.input:
        # 単一ファイルを変換
        count = convert_dataset_to_thinking_format(
            input_file=args.input,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Converted {count} samples")
    else:
        parser.error("Either --input or --inputs must be specified")


if __name__ == "__main__":
    main()





    
    with open(input_file, 'r', encoding='utf-8') as f, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                converted_sample = convert_sample_to_thinking_format(
                    sample,
                    default_system_message=default_system_message
                )
                
                if converted_sample is None:
                    skipped_count += 1
                    continue
                
                out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                converted_count += 1
                
                if converted_count % 1000 == 0:
                    logger.info(f"Converted {converted_count} samples...")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON decode error: {e}")
                skipped_count += 1
                continue
            except Exception as e:
                logger.warning(f"Line {line_num}: Error: {e}")
                skipped_count += 1
                continue
    
    logger.info(f"Conversion complete: {converted_count} samples converted, {skipped_count} skipped")
    logger.info(f"Saved to: {output_file}")
    
    return converted_count


def merge_multiple_datasets(
    input_files: List[Path],
    output_file: Path,
    default_system_message: Optional[str] = None
) -> int:
    """
    複数のデータセットをマージして/think形式に変換
    
    Args:
        input_files: 入力データセットファイルのリスト
        output_file: 出力データセットファイル
        default_system_message: デフォルトのシステムメッセージ
    
    Returns:
        マージされたサンプル数
    """
    total_count = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for input_file in input_files:
            logger.info(f"Processing: {input_file}")
            count = 0
            
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sample = json.loads(line)
                        converted_sample = convert_sample_to_thinking_format(
                            sample,
                            default_system_message=default_system_message
                        )
                        
                        if converted_sample is None:
                            continue
                        
                        out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                        count += 1
                        total_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error processing sample: {e}")
                        continue
            
            logger.info(f"  Processed {count} samples from {input_file}")
    
    logger.info(f"Merged {total_count} total samples")
    logger.info(f"Saved to: {output_file}")
    
    return total_count


def expand_dataset_gradually(
    base_dataset: Path,
    target_sizes: List[int],
    quality_threshold: float = 0.7,
    output_dir: Optional[Path] = None
) -> List[Path]:
    """
    データセットを段階的に拡張
    
    Args:
        base_dataset: ベースデータセットファイル
        target_sizes: 目標サイズのリスト（例: [5000, 10000, 25000, 50000]）
        quality_threshold: 品質スコアの閾値
        output_dir: 出力ディレクトリ（Noneの場合はbase_datasetと同じディレクトリ）
    
    Returns:
        各段階のデータセットファイルパスのリスト
    """
    if output_dir is None:
        output_dir = base_dataset.parent
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # ベースデータセットを読み込み
    logger.info(f"Loading base dataset from: {base_dataset}")
    all_samples = []
    with open(base_dataset, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                all_samples.append(sample)
            except json.JSONDecodeError:
                continue
    
    logger.info(f"Loaded {len(all_samples)} samples from base dataset")
    
    # 品質評価器を初期化
    evaluator = ThinkingDatasetQualityEvaluator()
    
    # 品質評価とフィルタリング
    logger.info(f"Evaluating quality with threshold: {quality_threshold}")
    quality_samples = evaluator.filter_by_quality(all_samples, min_score=quality_threshold)
    logger.info(f"Quality-filtered samples: {len(quality_samples)} / {len(all_samples)}")
    
    # 各段階でデータセットを拡張
    output_files = []
    current_samples = quality_samples.copy()
    
    for target_size in target_sizes:
        logger.info(f"Expanding to {target_size} samples...")
        
        # 目標サイズに達するまでサンプルを追加
        if len(current_samples) < target_size:
            # 不足分を既存サンプルからランダムに複製（簡易版）
            # 実際の実装では、新しいデータソースから追加することを推奨
            needed = target_size - len(current_samples)
            additional_samples = random.sample(quality_samples, min(needed, len(quality_samples)))
            current_samples.extend(additional_samples)
        
        # 目標サイズに合わせてサンプリング
        if len(current_samples) > target_size:
            current_samples = random.sample(current_samples, target_size)
        
        # 出力ファイル名を生成
        output_file = output_dir / f"{base_dataset.stem}_expanded_{target_size}.jsonl"
        
        # データセットを保存
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in current_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved expanded dataset: {output_file} ({len(current_samples)} samples)")
        output_files.append(output_file)
    
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Create /think format SFT dataset for Borea-Phi-3.5"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input dataset file (JSONL format)"
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        help="Multiple input dataset files (JSONL format) for merging"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset file (JSONL format)"
    )
    parser.add_argument(
        "--system-message",
        type=str,
        default=None,
        help="Custom system message (default: built-in thinking prompt)"
    )
    parser.add_argument(
        "--quality-filter",
        action="store_true",
        help="Enable quality filtering"
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.7,
        help="Minimum quality score for filtering (default: 0.7)"
    )
    parser.add_argument(
        "--expand-gradually",
        action="store_true",
        help="Expand dataset gradually (5000 -> 10000 -> 25000 -> 50000)"
    )
    parser.add_argument(
        "--target-sizes",
        type=int,
        nargs="+",
        default=[5000, 10000, 25000, 50000],
        help="Target sizes for gradual expansion (default: 5000 10000 25000 50000)"
    )
    
    args = parser.parse_args()
    
    if args.expand_gradually:
        # 段階的拡張
        if not args.input:
            parser.error("--expand-gradually requires --input")
        
        output_files = expand_dataset_gradually(
            base_dataset=args.input,
            target_sizes=args.target_sizes,
            quality_threshold=args.min_quality_score,
            output_dir=args.output.parent if args.output else None
        )
        logger.info(f"[SUCCESS] Expanded dataset to {len(output_files)} stages")
        for output_file in output_files:
            logger.info(f"  - {output_file}")
    
    elif args.inputs:
        # 複数ファイルをマージ
        count = merge_multiple_datasets(
            input_files=args.inputs,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Merged and converted {count} samples")
    elif args.input:
        # 単一ファイルを変換
        count = convert_dataset_to_thinking_format(
            input_file=args.input,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Converted {count} samples")
    else:
        parser.error("Either --input or --inputs must be specified")


if __name__ == "__main__":
    main()





    
    with open(input_file, 'r', encoding='utf-8') as f, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                converted_sample = convert_sample_to_thinking_format(
                    sample,
                    default_system_message=default_system_message
                )
                
                if converted_sample is None:
                    skipped_count += 1
                    continue
                
                out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                converted_count += 1
                
                if converted_count % 1000 == 0:
                    logger.info(f"Converted {converted_count} samples...")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON decode error: {e}")
                skipped_count += 1
                continue
            except Exception as e:
                logger.warning(f"Line {line_num}: Error: {e}")
                skipped_count += 1
                continue
    
    logger.info(f"Conversion complete: {converted_count} samples converted, {skipped_count} skipped")
    logger.info(f"Saved to: {output_file}")
    
    return converted_count


def merge_multiple_datasets(
    input_files: List[Path],
    output_file: Path,
    default_system_message: Optional[str] = None
) -> int:
    """
    複数のデータセットをマージして/think形式に変換
    
    Args:
        input_files: 入力データセットファイルのリスト
        output_file: 出力データセットファイル
        default_system_message: デフォルトのシステムメッセージ
    
    Returns:
        マージされたサンプル数
    """
    total_count = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for input_file in input_files:
            logger.info(f"Processing: {input_file}")
            count = 0
            
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sample = json.loads(line)
                        converted_sample = convert_sample_to_thinking_format(
                            sample,
                            default_system_message=default_system_message
                        )
                        
                        if converted_sample is None:
                            continue
                        
                        out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                        count += 1
                        total_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error processing sample: {e}")
                        continue
            
            logger.info(f"  Processed {count} samples from {input_file}")
    
    logger.info(f"Merged {total_count} total samples")
    logger.info(f"Saved to: {output_file}")
    
    return total_count


def expand_dataset_gradually(
    base_dataset: Path,
    target_sizes: List[int],
    quality_threshold: float = 0.7,
    output_dir: Optional[Path] = None
) -> List[Path]:
    """
    データセットを段階的に拡張
    
    Args:
        base_dataset: ベースデータセットファイル
        target_sizes: 目標サイズのリスト（例: [5000, 10000, 25000, 50000]）
        quality_threshold: 品質スコアの閾値
        output_dir: 出力ディレクトリ（Noneの場合はbase_datasetと同じディレクトリ）
    
    Returns:
        各段階のデータセットファイルパスのリスト
    """
    if output_dir is None:
        output_dir = base_dataset.parent
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # ベースデータセットを読み込み
    logger.info(f"Loading base dataset from: {base_dataset}")
    all_samples = []
    with open(base_dataset, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                all_samples.append(sample)
            except json.JSONDecodeError:
                continue
    
    logger.info(f"Loaded {len(all_samples)} samples from base dataset")
    
    # 品質評価器を初期化
    evaluator = ThinkingDatasetQualityEvaluator()
    
    # 品質評価とフィルタリング
    logger.info(f"Evaluating quality with threshold: {quality_threshold}")
    quality_samples = evaluator.filter_by_quality(all_samples, min_score=quality_threshold)
    logger.info(f"Quality-filtered samples: {len(quality_samples)} / {len(all_samples)}")
    
    # 各段階でデータセットを拡張
    output_files = []
    current_samples = quality_samples.copy()
    
    for target_size in target_sizes:
        logger.info(f"Expanding to {target_size} samples...")
        
        # 目標サイズに達するまでサンプルを追加
        if len(current_samples) < target_size:
            # 不足分を既存サンプルからランダムに複製（簡易版）
            # 実際の実装では、新しいデータソースから追加することを推奨
            needed = target_size - len(current_samples)
            additional_samples = random.sample(quality_samples, min(needed, len(quality_samples)))
            current_samples.extend(additional_samples)
        
        # 目標サイズに合わせてサンプリング
        if len(current_samples) > target_size:
            current_samples = random.sample(current_samples, target_size)
        
        # 出力ファイル名を生成
        output_file = output_dir / f"{base_dataset.stem}_expanded_{target_size}.jsonl"
        
        # データセットを保存
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in current_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved expanded dataset: {output_file} ({len(current_samples)} samples)")
        output_files.append(output_file)
    
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Create /think format SFT dataset for Borea-Phi-3.5"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input dataset file (JSONL format)"
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        help="Multiple input dataset files (JSONL format) for merging"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset file (JSONL format)"
    )
    parser.add_argument(
        "--system-message",
        type=str,
        default=None,
        help="Custom system message (default: built-in thinking prompt)"
    )
    parser.add_argument(
        "--quality-filter",
        action="store_true",
        help="Enable quality filtering"
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.7,
        help="Minimum quality score for filtering (default: 0.7)"
    )
    parser.add_argument(
        "--expand-gradually",
        action="store_true",
        help="Expand dataset gradually (5000 -> 10000 -> 25000 -> 50000)"
    )
    parser.add_argument(
        "--target-sizes",
        type=int,
        nargs="+",
        default=[5000, 10000, 25000, 50000],
        help="Target sizes for gradual expansion (default: 5000 10000 25000 50000)"
    )
    
    args = parser.parse_args()
    
    if args.expand_gradually:
        # 段階的拡張
        if not args.input:
            parser.error("--expand-gradually requires --input")
        
        output_files = expand_dataset_gradually(
            base_dataset=args.input,
            target_sizes=args.target_sizes,
            quality_threshold=args.min_quality_score,
            output_dir=args.output.parent if args.output else None
        )
        logger.info(f"[SUCCESS] Expanded dataset to {len(output_files)} stages")
        for output_file in output_files:
            logger.info(f"  - {output_file}")
    
    elif args.inputs:
        # 複数ファイルをマージ
        count = merge_multiple_datasets(
            input_files=args.inputs,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Merged and converted {count} samples")
    elif args.input:
        # 単一ファイルを変換
        count = convert_dataset_to_thinking_format(
            input_file=args.input,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Converted {count} samples")
    else:
        parser.error("Either --input or --inputs must be specified")


if __name__ == "__main__":
    main()





    
    with open(input_file, 'r', encoding='utf-8') as f, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                converted_sample = convert_sample_to_thinking_format(
                    sample,
                    default_system_message=default_system_message
                )
                
                if converted_sample is None:
                    skipped_count += 1
                    continue
                
                out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                converted_count += 1
                
                if converted_count % 1000 == 0:
                    logger.info(f"Converted {converted_count} samples...")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON decode error: {e}")
                skipped_count += 1
                continue
            except Exception as e:
                logger.warning(f"Line {line_num}: Error: {e}")
                skipped_count += 1
                continue
    
    logger.info(f"Conversion complete: {converted_count} samples converted, {skipped_count} skipped")
    logger.info(f"Saved to: {output_file}")
    
    return converted_count


def merge_multiple_datasets(
    input_files: List[Path],
    output_file: Path,
    default_system_message: Optional[str] = None
) -> int:
    """
    複数のデータセットをマージして/think形式に変換
    
    Args:
        input_files: 入力データセットファイルのリスト
        output_file: 出力データセットファイル
        default_system_message: デフォルトのシステムメッセージ
    
    Returns:
        マージされたサンプル数
    """
    total_count = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for input_file in input_files:
            logger.info(f"Processing: {input_file}")
            count = 0
            
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sample = json.loads(line)
                        converted_sample = convert_sample_to_thinking_format(
                            sample,
                            default_system_message=default_system_message
                        )
                        
                        if converted_sample is None:
                            continue
                        
                        out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                        count += 1
                        total_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error processing sample: {e}")
                        continue
            
            logger.info(f"  Processed {count} samples from {input_file}")
    
    logger.info(f"Merged {total_count} total samples")
    logger.info(f"Saved to: {output_file}")
    
    return total_count


def expand_dataset_gradually(
    base_dataset: Path,
    target_sizes: List[int],
    quality_threshold: float = 0.7,
    output_dir: Optional[Path] = None
) -> List[Path]:
    """
    データセットを段階的に拡張
    
    Args:
        base_dataset: ベースデータセットファイル
        target_sizes: 目標サイズのリスト（例: [5000, 10000, 25000, 50000]）
        quality_threshold: 品質スコアの閾値
        output_dir: 出力ディレクトリ（Noneの場合はbase_datasetと同じディレクトリ）
    
    Returns:
        各段階のデータセットファイルパスのリスト
    """
    if output_dir is None:
        output_dir = base_dataset.parent
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # ベースデータセットを読み込み
    logger.info(f"Loading base dataset from: {base_dataset}")
    all_samples = []
    with open(base_dataset, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                all_samples.append(sample)
            except json.JSONDecodeError:
                continue
    
    logger.info(f"Loaded {len(all_samples)} samples from base dataset")
    
    # 品質評価器を初期化
    evaluator = ThinkingDatasetQualityEvaluator()
    
    # 品質評価とフィルタリング
    logger.info(f"Evaluating quality with threshold: {quality_threshold}")
    quality_samples = evaluator.filter_by_quality(all_samples, min_score=quality_threshold)
    logger.info(f"Quality-filtered samples: {len(quality_samples)} / {len(all_samples)}")
    
    # 各段階でデータセットを拡張
    output_files = []
    current_samples = quality_samples.copy()
    
    for target_size in target_sizes:
        logger.info(f"Expanding to {target_size} samples...")
        
        # 目標サイズに達するまでサンプルを追加
        if len(current_samples) < target_size:
            # 不足分を既存サンプルからランダムに複製（簡易版）
            # 実際の実装では、新しいデータソースから追加することを推奨
            needed = target_size - len(current_samples)
            additional_samples = random.sample(quality_samples, min(needed, len(quality_samples)))
            current_samples.extend(additional_samples)
        
        # 目標サイズに合わせてサンプリング
        if len(current_samples) > target_size:
            current_samples = random.sample(current_samples, target_size)
        
        # 出力ファイル名を生成
        output_file = output_dir / f"{base_dataset.stem}_expanded_{target_size}.jsonl"
        
        # データセットを保存
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in current_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved expanded dataset: {output_file} ({len(current_samples)} samples)")
        output_files.append(output_file)
    
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Create /think format SFT dataset for Borea-Phi-3.5"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input dataset file (JSONL format)"
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        help="Multiple input dataset files (JSONL format) for merging"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset file (JSONL format)"
    )
    parser.add_argument(
        "--system-message",
        type=str,
        default=None,
        help="Custom system message (default: built-in thinking prompt)"
    )
    parser.add_argument(
        "--quality-filter",
        action="store_true",
        help="Enable quality filtering"
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.7,
        help="Minimum quality score for filtering (default: 0.7)"
    )
    parser.add_argument(
        "--expand-gradually",
        action="store_true",
        help="Expand dataset gradually (5000 -> 10000 -> 25000 -> 50000)"
    )
    parser.add_argument(
        "--target-sizes",
        type=int,
        nargs="+",
        default=[5000, 10000, 25000, 50000],
        help="Target sizes for gradual expansion (default: 5000 10000 25000 50000)"
    )
    
    args = parser.parse_args()
    
    if args.expand_gradually:
        # 段階的拡張
        if not args.input:
            parser.error("--expand-gradually requires --input")
        
        output_files = expand_dataset_gradually(
            base_dataset=args.input,
            target_sizes=args.target_sizes,
            quality_threshold=args.min_quality_score,
            output_dir=args.output.parent if args.output else None
        )
        logger.info(f"[SUCCESS] Expanded dataset to {len(output_files)} stages")
        for output_file in output_files:
            logger.info(f"  - {output_file}")
    
    elif args.inputs:
        # 複数ファイルをマージ
        count = merge_multiple_datasets(
            input_files=args.inputs,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Merged and converted {count} samples")
    elif args.input:
        # 単一ファイルを変換
        count = convert_dataset_to_thinking_format(
            input_file=args.input,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Converted {count} samples")
    else:
        parser.error("Either --input or --inputs must be specified")


if __name__ == "__main__":
    main()





    
    with open(input_file, 'r', encoding='utf-8') as f, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                converted_sample = convert_sample_to_thinking_format(
                    sample,
                    default_system_message=default_system_message
                )
                
                if converted_sample is None:
                    skipped_count += 1
                    continue
                
                out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                converted_count += 1
                
                if converted_count % 1000 == 0:
                    logger.info(f"Converted {converted_count} samples...")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON decode error: {e}")
                skipped_count += 1
                continue
            except Exception as e:
                logger.warning(f"Line {line_num}: Error: {e}")
                skipped_count += 1
                continue
    
    logger.info(f"Conversion complete: {converted_count} samples converted, {skipped_count} skipped")
    logger.info(f"Saved to: {output_file}")
    
    return converted_count


def merge_multiple_datasets(
    input_files: List[Path],
    output_file: Path,
    default_system_message: Optional[str] = None
) -> int:
    """
    複数のデータセットをマージして/think形式に変換
    
    Args:
        input_files: 入力データセットファイルのリスト
        output_file: 出力データセットファイル
        default_system_message: デフォルトのシステムメッセージ
    
    Returns:
        マージされたサンプル数
    """
    total_count = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for input_file in input_files:
            logger.info(f"Processing: {input_file}")
            count = 0
            
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sample = json.loads(line)
                        converted_sample = convert_sample_to_thinking_format(
                            sample,
                            default_system_message=default_system_message
                        )
                        
                        if converted_sample is None:
                            continue
                        
                        out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                        count += 1
                        total_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error processing sample: {e}")
                        continue
            
            logger.info(f"  Processed {count} samples from {input_file}")
    
    logger.info(f"Merged {total_count} total samples")
    logger.info(f"Saved to: {output_file}")
    
    return total_count


def expand_dataset_gradually(
    base_dataset: Path,
    target_sizes: List[int],
    quality_threshold: float = 0.7,
    output_dir: Optional[Path] = None
) -> List[Path]:
    """
    データセットを段階的に拡張
    
    Args:
        base_dataset: ベースデータセットファイル
        target_sizes: 目標サイズのリスト（例: [5000, 10000, 25000, 50000]）
        quality_threshold: 品質スコアの閾値
        output_dir: 出力ディレクトリ（Noneの場合はbase_datasetと同じディレクトリ）
    
    Returns:
        各段階のデータセットファイルパスのリスト
    """
    if output_dir is None:
        output_dir = base_dataset.parent
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # ベースデータセットを読み込み
    logger.info(f"Loading base dataset from: {base_dataset}")
    all_samples = []
    with open(base_dataset, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                all_samples.append(sample)
            except json.JSONDecodeError:
                continue
    
    logger.info(f"Loaded {len(all_samples)} samples from base dataset")
    
    # 品質評価器を初期化
    evaluator = ThinkingDatasetQualityEvaluator()
    
    # 品質評価とフィルタリング
    logger.info(f"Evaluating quality with threshold: {quality_threshold}")
    quality_samples = evaluator.filter_by_quality(all_samples, min_score=quality_threshold)
    logger.info(f"Quality-filtered samples: {len(quality_samples)} / {len(all_samples)}")
    
    # 各段階でデータセットを拡張
    output_files = []
    current_samples = quality_samples.copy()
    
    for target_size in target_sizes:
        logger.info(f"Expanding to {target_size} samples...")
        
        # 目標サイズに達するまでサンプルを追加
        if len(current_samples) < target_size:
            # 不足分を既存サンプルからランダムに複製（簡易版）
            # 実際の実装では、新しいデータソースから追加することを推奨
            needed = target_size - len(current_samples)
            additional_samples = random.sample(quality_samples, min(needed, len(quality_samples)))
            current_samples.extend(additional_samples)
        
        # 目標サイズに合わせてサンプリング
        if len(current_samples) > target_size:
            current_samples = random.sample(current_samples, target_size)
        
        # 出力ファイル名を生成
        output_file = output_dir / f"{base_dataset.stem}_expanded_{target_size}.jsonl"
        
        # データセットを保存
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in current_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved expanded dataset: {output_file} ({len(current_samples)} samples)")
        output_files.append(output_file)
    
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Create /think format SFT dataset for Borea-Phi-3.5"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input dataset file (JSONL format)"
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        help="Multiple input dataset files (JSONL format) for merging"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset file (JSONL format)"
    )
    parser.add_argument(
        "--system-message",
        type=str,
        default=None,
        help="Custom system message (default: built-in thinking prompt)"
    )
    parser.add_argument(
        "--quality-filter",
        action="store_true",
        help="Enable quality filtering"
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.7,
        help="Minimum quality score for filtering (default: 0.7)"
    )
    parser.add_argument(
        "--expand-gradually",
        action="store_true",
        help="Expand dataset gradually (5000 -> 10000 -> 25000 -> 50000)"
    )
    parser.add_argument(
        "--target-sizes",
        type=int,
        nargs="+",
        default=[5000, 10000, 25000, 50000],
        help="Target sizes for gradual expansion (default: 5000 10000 25000 50000)"
    )
    
    args = parser.parse_args()
    
    if args.expand_gradually:
        # 段階的拡張
        if not args.input:
            parser.error("--expand-gradually requires --input")
        
        output_files = expand_dataset_gradually(
            base_dataset=args.input,
            target_sizes=args.target_sizes,
            quality_threshold=args.min_quality_score,
            output_dir=args.output.parent if args.output else None
        )
        logger.info(f"[SUCCESS] Expanded dataset to {len(output_files)} stages")
        for output_file in output_files:
            logger.info(f"  - {output_file}")
    
    elif args.inputs:
        # 複数ファイルをマージ
        count = merge_multiple_datasets(
            input_files=args.inputs,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Merged and converted {count} samples")
    elif args.input:
        # 単一ファイルを変換
        count = convert_dataset_to_thinking_format(
            input_file=args.input,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Converted {count} samples")
    else:
        parser.error("Either --input or --inputs must be specified")


if __name__ == "__main__":
    main()





    
    with open(input_file, 'r', encoding='utf-8') as f, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                converted_sample = convert_sample_to_thinking_format(
                    sample,
                    default_system_message=default_system_message
                )
                
                if converted_sample is None:
                    skipped_count += 1
                    continue
                
                out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                converted_count += 1
                
                if converted_count % 1000 == 0:
                    logger.info(f"Converted {converted_count} samples...")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON decode error: {e}")
                skipped_count += 1
                continue
            except Exception as e:
                logger.warning(f"Line {line_num}: Error: {e}")
                skipped_count += 1
                continue
    
    logger.info(f"Conversion complete: {converted_count} samples converted, {skipped_count} skipped")
    logger.info(f"Saved to: {output_file}")
    
    return converted_count


def merge_multiple_datasets(
    input_files: List[Path],
    output_file: Path,
    default_system_message: Optional[str] = None
) -> int:
    """
    複数のデータセットをマージして/think形式に変換
    
    Args:
        input_files: 入力データセットファイルのリスト
        output_file: 出力データセットファイル
        default_system_message: デフォルトのシステムメッセージ
    
    Returns:
        マージされたサンプル数
    """
    total_count = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for input_file in input_files:
            logger.info(f"Processing: {input_file}")
            count = 0
            
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sample = json.loads(line)
                        converted_sample = convert_sample_to_thinking_format(
                            sample,
                            default_system_message=default_system_message
                        )
                        
                        if converted_sample is None:
                            continue
                        
                        out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                        count += 1
                        total_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error processing sample: {e}")
                        continue
            
            logger.info(f"  Processed {count} samples from {input_file}")
    
    logger.info(f"Merged {total_count} total samples")
    logger.info(f"Saved to: {output_file}")
    
    return total_count


def expand_dataset_gradually(
    base_dataset: Path,
    target_sizes: List[int],
    quality_threshold: float = 0.7,
    output_dir: Optional[Path] = None
) -> List[Path]:
    """
    データセットを段階的に拡張
    
    Args:
        base_dataset: ベースデータセットファイル
        target_sizes: 目標サイズのリスト（例: [5000, 10000, 25000, 50000]）
        quality_threshold: 品質スコアの閾値
        output_dir: 出力ディレクトリ（Noneの場合はbase_datasetと同じディレクトリ）
    
    Returns:
        各段階のデータセットファイルパスのリスト
    """
    if output_dir is None:
        output_dir = base_dataset.parent
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # ベースデータセットを読み込み
    logger.info(f"Loading base dataset from: {base_dataset}")
    all_samples = []
    with open(base_dataset, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                all_samples.append(sample)
            except json.JSONDecodeError:
                continue
    
    logger.info(f"Loaded {len(all_samples)} samples from base dataset")
    
    # 品質評価器を初期化
    evaluator = ThinkingDatasetQualityEvaluator()
    
    # 品質評価とフィルタリング
    logger.info(f"Evaluating quality with threshold: {quality_threshold}")
    quality_samples = evaluator.filter_by_quality(all_samples, min_score=quality_threshold)
    logger.info(f"Quality-filtered samples: {len(quality_samples)} / {len(all_samples)}")
    
    # 各段階でデータセットを拡張
    output_files = []
    current_samples = quality_samples.copy()
    
    for target_size in target_sizes:
        logger.info(f"Expanding to {target_size} samples...")
        
        # 目標サイズに達するまでサンプルを追加
        if len(current_samples) < target_size:
            # 不足分を既存サンプルからランダムに複製（簡易版）
            # 実際の実装では、新しいデータソースから追加することを推奨
            needed = target_size - len(current_samples)
            additional_samples = random.sample(quality_samples, min(needed, len(quality_samples)))
            current_samples.extend(additional_samples)
        
        # 目標サイズに合わせてサンプリング
        if len(current_samples) > target_size:
            current_samples = random.sample(current_samples, target_size)
        
        # 出力ファイル名を生成
        output_file = output_dir / f"{base_dataset.stem}_expanded_{target_size}.jsonl"
        
        # データセットを保存
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in current_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved expanded dataset: {output_file} ({len(current_samples)} samples)")
        output_files.append(output_file)
    
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Create /think format SFT dataset for Borea-Phi-3.5"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input dataset file (JSONL format)"
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        help="Multiple input dataset files (JSONL format) for merging"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset file (JSONL format)"
    )
    parser.add_argument(
        "--system-message",
        type=str,
        default=None,
        help="Custom system message (default: built-in thinking prompt)"
    )
    parser.add_argument(
        "--quality-filter",
        action="store_true",
        help="Enable quality filtering"
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.7,
        help="Minimum quality score for filtering (default: 0.7)"
    )
    parser.add_argument(
        "--expand-gradually",
        action="store_true",
        help="Expand dataset gradually (5000 -> 10000 -> 25000 -> 50000)"
    )
    parser.add_argument(
        "--target-sizes",
        type=int,
        nargs="+",
        default=[5000, 10000, 25000, 50000],
        help="Target sizes for gradual expansion (default: 5000 10000 25000 50000)"
    )
    
    args = parser.parse_args()
    
    if args.expand_gradually:
        # 段階的拡張
        if not args.input:
            parser.error("--expand-gradually requires --input")
        
        output_files = expand_dataset_gradually(
            base_dataset=args.input,
            target_sizes=args.target_sizes,
            quality_threshold=args.min_quality_score,
            output_dir=args.output.parent if args.output else None
        )
        logger.info(f"[SUCCESS] Expanded dataset to {len(output_files)} stages")
        for output_file in output_files:
            logger.info(f"  - {output_file}")
    
    elif args.inputs:
        # 複数ファイルをマージ
        count = merge_multiple_datasets(
            input_files=args.inputs,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Merged and converted {count} samples")
    elif args.input:
        # 単一ファイルを変換
        count = convert_dataset_to_thinking_format(
            input_file=args.input,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Converted {count} samples")
    else:
        parser.error("Either --input or --inputs must be specified")


if __name__ == "__main__":
    main()





    
    with open(input_file, 'r', encoding='utf-8') as f, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                converted_sample = convert_sample_to_thinking_format(
                    sample,
                    default_system_message=default_system_message
                )
                
                if converted_sample is None:
                    skipped_count += 1
                    continue
                
                out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                converted_count += 1
                
                if converted_count % 1000 == 0:
                    logger.info(f"Converted {converted_count} samples...")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON decode error: {e}")
                skipped_count += 1
                continue
            except Exception as e:
                logger.warning(f"Line {line_num}: Error: {e}")
                skipped_count += 1
                continue
    
    logger.info(f"Conversion complete: {converted_count} samples converted, {skipped_count} skipped")
    logger.info(f"Saved to: {output_file}")
    
    return converted_count


def merge_multiple_datasets(
    input_files: List[Path],
    output_file: Path,
    default_system_message: Optional[str] = None
) -> int:
    """
    複数のデータセットをマージして/think形式に変換
    
    Args:
        input_files: 入力データセットファイルのリスト
        output_file: 出力データセットファイル
        default_system_message: デフォルトのシステムメッセージ
    
    Returns:
        マージされたサンプル数
    """
    total_count = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for input_file in input_files:
            logger.info(f"Processing: {input_file}")
            count = 0
            
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sample = json.loads(line)
                        converted_sample = convert_sample_to_thinking_format(
                            sample,
                            default_system_message=default_system_message
                        )
                        
                        if converted_sample is None:
                            continue
                        
                        out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                        count += 1
                        total_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error processing sample: {e}")
                        continue
            
            logger.info(f"  Processed {count} samples from {input_file}")
    
    logger.info(f"Merged {total_count} total samples")
    logger.info(f"Saved to: {output_file}")
    
    return total_count


def expand_dataset_gradually(
    base_dataset: Path,
    target_sizes: List[int],
    quality_threshold: float = 0.7,
    output_dir: Optional[Path] = None
) -> List[Path]:
    """
    データセットを段階的に拡張
    
    Args:
        base_dataset: ベースデータセットファイル
        target_sizes: 目標サイズのリスト（例: [5000, 10000, 25000, 50000]）
        quality_threshold: 品質スコアの閾値
        output_dir: 出力ディレクトリ（Noneの場合はbase_datasetと同じディレクトリ）
    
    Returns:
        各段階のデータセットファイルパスのリスト
    """
    if output_dir is None:
        output_dir = base_dataset.parent
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # ベースデータセットを読み込み
    logger.info(f"Loading base dataset from: {base_dataset}")
    all_samples = []
    with open(base_dataset, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                all_samples.append(sample)
            except json.JSONDecodeError:
                continue
    
    logger.info(f"Loaded {len(all_samples)} samples from base dataset")
    
    # 品質評価器を初期化
    evaluator = ThinkingDatasetQualityEvaluator()
    
    # 品質評価とフィルタリング
    logger.info(f"Evaluating quality with threshold: {quality_threshold}")
    quality_samples = evaluator.filter_by_quality(all_samples, min_score=quality_threshold)
    logger.info(f"Quality-filtered samples: {len(quality_samples)} / {len(all_samples)}")
    
    # 各段階でデータセットを拡張
    output_files = []
    current_samples = quality_samples.copy()
    
    for target_size in target_sizes:
        logger.info(f"Expanding to {target_size} samples...")
        
        # 目標サイズに達するまでサンプルを追加
        if len(current_samples) < target_size:
            # 不足分を既存サンプルからランダムに複製（簡易版）
            # 実際の実装では、新しいデータソースから追加することを推奨
            needed = target_size - len(current_samples)
            additional_samples = random.sample(quality_samples, min(needed, len(quality_samples)))
            current_samples.extend(additional_samples)
        
        # 目標サイズに合わせてサンプリング
        if len(current_samples) > target_size:
            current_samples = random.sample(current_samples, target_size)
        
        # 出力ファイル名を生成
        output_file = output_dir / f"{base_dataset.stem}_expanded_{target_size}.jsonl"
        
        # データセットを保存
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in current_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved expanded dataset: {output_file} ({len(current_samples)} samples)")
        output_files.append(output_file)
    
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Create /think format SFT dataset for Borea-Phi-3.5"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input dataset file (JSONL format)"
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        help="Multiple input dataset files (JSONL format) for merging"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset file (JSONL format)"
    )
    parser.add_argument(
        "--system-message",
        type=str,
        default=None,
        help="Custom system message (default: built-in thinking prompt)"
    )
    parser.add_argument(
        "--quality-filter",
        action="store_true",
        help="Enable quality filtering"
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.7,
        help="Minimum quality score for filtering (default: 0.7)"
    )
    parser.add_argument(
        "--expand-gradually",
        action="store_true",
        help="Expand dataset gradually (5000 -> 10000 -> 25000 -> 50000)"
    )
    parser.add_argument(
        "--target-sizes",
        type=int,
        nargs="+",
        default=[5000, 10000, 25000, 50000],
        help="Target sizes for gradual expansion (default: 5000 10000 25000 50000)"
    )
    
    args = parser.parse_args()
    
    if args.expand_gradually:
        # 段階的拡張
        if not args.input:
            parser.error("--expand-gradually requires --input")
        
        output_files = expand_dataset_gradually(
            base_dataset=args.input,
            target_sizes=args.target_sizes,
            quality_threshold=args.min_quality_score,
            output_dir=args.output.parent if args.output else None
        )
        logger.info(f"[SUCCESS] Expanded dataset to {len(output_files)} stages")
        for output_file in output_files:
            logger.info(f"  - {output_file}")
    
    elif args.inputs:
        # 複数ファイルをマージ
        count = merge_multiple_datasets(
            input_files=args.inputs,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Merged and converted {count} samples")
    elif args.input:
        # 単一ファイルを変換
        count = convert_dataset_to_thinking_format(
            input_file=args.input,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Converted {count} samples")
    else:
        parser.error("Either --input or --inputs must be specified")


if __name__ == "__main__":
    main()





    
    with open(input_file, 'r', encoding='utf-8') as f, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                converted_sample = convert_sample_to_thinking_format(
                    sample,
                    default_system_message=default_system_message
                )
                
                if converted_sample is None:
                    skipped_count += 1
                    continue
                
                out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                converted_count += 1
                
                if converted_count % 1000 == 0:
                    logger.info(f"Converted {converted_count} samples...")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON decode error: {e}")
                skipped_count += 1
                continue
            except Exception as e:
                logger.warning(f"Line {line_num}: Error: {e}")
                skipped_count += 1
                continue
    
    logger.info(f"Conversion complete: {converted_count} samples converted, {skipped_count} skipped")
    logger.info(f"Saved to: {output_file}")
    
    return converted_count


def merge_multiple_datasets(
    input_files: List[Path],
    output_file: Path,
    default_system_message: Optional[str] = None
) -> int:
    """
    複数のデータセットをマージして/think形式に変換
    
    Args:
        input_files: 入力データセットファイルのリスト
        output_file: 出力データセットファイル
        default_system_message: デフォルトのシステムメッセージ
    
    Returns:
        マージされたサンプル数
    """
    total_count = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for input_file in input_files:
            logger.info(f"Processing: {input_file}")
            count = 0
            
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sample = json.loads(line)
                        converted_sample = convert_sample_to_thinking_format(
                            sample,
                            default_system_message=default_system_message
                        )
                        
                        if converted_sample is None:
                            continue
                        
                        out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                        count += 1
                        total_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error processing sample: {e}")
                        continue
            
            logger.info(f"  Processed {count} samples from {input_file}")
    
    logger.info(f"Merged {total_count} total samples")
    logger.info(f"Saved to: {output_file}")
    
    return total_count


def expand_dataset_gradually(
    base_dataset: Path,
    target_sizes: List[int],
    quality_threshold: float = 0.7,
    output_dir: Optional[Path] = None
) -> List[Path]:
    """
    データセットを段階的に拡張
    
    Args:
        base_dataset: ベースデータセットファイル
        target_sizes: 目標サイズのリスト（例: [5000, 10000, 25000, 50000]）
        quality_threshold: 品質スコアの閾値
        output_dir: 出力ディレクトリ（Noneの場合はbase_datasetと同じディレクトリ）
    
    Returns:
        各段階のデータセットファイルパスのリスト
    """
    if output_dir is None:
        output_dir = base_dataset.parent
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # ベースデータセットを読み込み
    logger.info(f"Loading base dataset from: {base_dataset}")
    all_samples = []
    with open(base_dataset, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                all_samples.append(sample)
            except json.JSONDecodeError:
                continue
    
    logger.info(f"Loaded {len(all_samples)} samples from base dataset")
    
    # 品質評価器を初期化
    evaluator = ThinkingDatasetQualityEvaluator()
    
    # 品質評価とフィルタリング
    logger.info(f"Evaluating quality with threshold: {quality_threshold}")
    quality_samples = evaluator.filter_by_quality(all_samples, min_score=quality_threshold)
    logger.info(f"Quality-filtered samples: {len(quality_samples)} / {len(all_samples)}")
    
    # 各段階でデータセットを拡張
    output_files = []
    current_samples = quality_samples.copy()
    
    for target_size in target_sizes:
        logger.info(f"Expanding to {target_size} samples...")
        
        # 目標サイズに達するまでサンプルを追加
        if len(current_samples) < target_size:
            # 不足分を既存サンプルからランダムに複製（簡易版）
            # 実際の実装では、新しいデータソースから追加することを推奨
            needed = target_size - len(current_samples)
            additional_samples = random.sample(quality_samples, min(needed, len(quality_samples)))
            current_samples.extend(additional_samples)
        
        # 目標サイズに合わせてサンプリング
        if len(current_samples) > target_size:
            current_samples = random.sample(current_samples, target_size)
        
        # 出力ファイル名を生成
        output_file = output_dir / f"{base_dataset.stem}_expanded_{target_size}.jsonl"
        
        # データセットを保存
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in current_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved expanded dataset: {output_file} ({len(current_samples)} samples)")
        output_files.append(output_file)
    
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Create /think format SFT dataset for Borea-Phi-3.5"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input dataset file (JSONL format)"
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        help="Multiple input dataset files (JSONL format) for merging"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset file (JSONL format)"
    )
    parser.add_argument(
        "--system-message",
        type=str,
        default=None,
        help="Custom system message (default: built-in thinking prompt)"
    )
    parser.add_argument(
        "--quality-filter",
        action="store_true",
        help="Enable quality filtering"
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.7,
        help="Minimum quality score for filtering (default: 0.7)"
    )
    parser.add_argument(
        "--expand-gradually",
        action="store_true",
        help="Expand dataset gradually (5000 -> 10000 -> 25000 -> 50000)"
    )
    parser.add_argument(
        "--target-sizes",
        type=int,
        nargs="+",
        default=[5000, 10000, 25000, 50000],
        help="Target sizes for gradual expansion (default: 5000 10000 25000 50000)"
    )
    
    args = parser.parse_args()
    
    if args.expand_gradually:
        # 段階的拡張
        if not args.input:
            parser.error("--expand-gradually requires --input")
        
        output_files = expand_dataset_gradually(
            base_dataset=args.input,
            target_sizes=args.target_sizes,
            quality_threshold=args.min_quality_score,
            output_dir=args.output.parent if args.output else None
        )
        logger.info(f"[SUCCESS] Expanded dataset to {len(output_files)} stages")
        for output_file in output_files:
            logger.info(f"  - {output_file}")
    
    elif args.inputs:
        # 複数ファイルをマージ
        count = merge_multiple_datasets(
            input_files=args.inputs,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Merged and converted {count} samples")
    elif args.input:
        # 単一ファイルを変換
        count = convert_dataset_to_thinking_format(
            input_file=args.input,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Converted {count} samples")
    else:
        parser.error("Either --input or --inputs must be specified")


if __name__ == "__main__":
    main()





    
    with open(input_file, 'r', encoding='utf-8') as f, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                converted_sample = convert_sample_to_thinking_format(
                    sample,
                    default_system_message=default_system_message
                )
                
                if converted_sample is None:
                    skipped_count += 1
                    continue
                
                out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                converted_count += 1
                
                if converted_count % 1000 == 0:
                    logger.info(f"Converted {converted_count} samples...")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON decode error: {e}")
                skipped_count += 1
                continue
            except Exception as e:
                logger.warning(f"Line {line_num}: Error: {e}")
                skipped_count += 1
                continue
    
    logger.info(f"Conversion complete: {converted_count} samples converted, {skipped_count} skipped")
    logger.info(f"Saved to: {output_file}")
    
    return converted_count


def merge_multiple_datasets(
    input_files: List[Path],
    output_file: Path,
    default_system_message: Optional[str] = None
) -> int:
    """
    複数のデータセットをマージして/think形式に変換
    
    Args:
        input_files: 入力データセットファイルのリスト
        output_file: 出力データセットファイル
        default_system_message: デフォルトのシステムメッセージ
    
    Returns:
        マージされたサンプル数
    """
    total_count = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for input_file in input_files:
            logger.info(f"Processing: {input_file}")
            count = 0
            
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sample = json.loads(line)
                        converted_sample = convert_sample_to_thinking_format(
                            sample,
                            default_system_message=default_system_message
                        )
                        
                        if converted_sample is None:
                            continue
                        
                        out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                        count += 1
                        total_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error processing sample: {e}")
                        continue
            
            logger.info(f"  Processed {count} samples from {input_file}")
    
    logger.info(f"Merged {total_count} total samples")
    logger.info(f"Saved to: {output_file}")
    
    return total_count


def expand_dataset_gradually(
    base_dataset: Path,
    target_sizes: List[int],
    quality_threshold: float = 0.7,
    output_dir: Optional[Path] = None
) -> List[Path]:
    """
    データセットを段階的に拡張
    
    Args:
        base_dataset: ベースデータセットファイル
        target_sizes: 目標サイズのリスト（例: [5000, 10000, 25000, 50000]）
        quality_threshold: 品質スコアの閾値
        output_dir: 出力ディレクトリ（Noneの場合はbase_datasetと同じディレクトリ）
    
    Returns:
        各段階のデータセットファイルパスのリスト
    """
    if output_dir is None:
        output_dir = base_dataset.parent
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # ベースデータセットを読み込み
    logger.info(f"Loading base dataset from: {base_dataset}")
    all_samples = []
    with open(base_dataset, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                all_samples.append(sample)
            except json.JSONDecodeError:
                continue
    
    logger.info(f"Loaded {len(all_samples)} samples from base dataset")
    
    # 品質評価器を初期化
    evaluator = ThinkingDatasetQualityEvaluator()
    
    # 品質評価とフィルタリング
    logger.info(f"Evaluating quality with threshold: {quality_threshold}")
    quality_samples = evaluator.filter_by_quality(all_samples, min_score=quality_threshold)
    logger.info(f"Quality-filtered samples: {len(quality_samples)} / {len(all_samples)}")
    
    # 各段階でデータセットを拡張
    output_files = []
    current_samples = quality_samples.copy()
    
    for target_size in target_sizes:
        logger.info(f"Expanding to {target_size} samples...")
        
        # 目標サイズに達するまでサンプルを追加
        if len(current_samples) < target_size:
            # 不足分を既存サンプルからランダムに複製（簡易版）
            # 実際の実装では、新しいデータソースから追加することを推奨
            needed = target_size - len(current_samples)
            additional_samples = random.sample(quality_samples, min(needed, len(quality_samples)))
            current_samples.extend(additional_samples)
        
        # 目標サイズに合わせてサンプリング
        if len(current_samples) > target_size:
            current_samples = random.sample(current_samples, target_size)
        
        # 出力ファイル名を生成
        output_file = output_dir / f"{base_dataset.stem}_expanded_{target_size}.jsonl"
        
        # データセットを保存
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in current_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved expanded dataset: {output_file} ({len(current_samples)} samples)")
        output_files.append(output_file)
    
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Create /think format SFT dataset for Borea-Phi-3.5"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input dataset file (JSONL format)"
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        help="Multiple input dataset files (JSONL format) for merging"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset file (JSONL format)"
    )
    parser.add_argument(
        "--system-message",
        type=str,
        default=None,
        help="Custom system message (default: built-in thinking prompt)"
    )
    parser.add_argument(
        "--quality-filter",
        action="store_true",
        help="Enable quality filtering"
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.7,
        help="Minimum quality score for filtering (default: 0.7)"
    )
    parser.add_argument(
        "--expand-gradually",
        action="store_true",
        help="Expand dataset gradually (5000 -> 10000 -> 25000 -> 50000)"
    )
    parser.add_argument(
        "--target-sizes",
        type=int,
        nargs="+",
        default=[5000, 10000, 25000, 50000],
        help="Target sizes for gradual expansion (default: 5000 10000 25000 50000)"
    )
    
    args = parser.parse_args()
    
    if args.expand_gradually:
        # 段階的拡張
        if not args.input:
            parser.error("--expand-gradually requires --input")
        
        output_files = expand_dataset_gradually(
            base_dataset=args.input,
            target_sizes=args.target_sizes,
            quality_threshold=args.min_quality_score,
            output_dir=args.output.parent if args.output else None
        )
        logger.info(f"[SUCCESS] Expanded dataset to {len(output_files)} stages")
        for output_file in output_files:
            logger.info(f"  - {output_file}")
    
    elif args.inputs:
        # 複数ファイルをマージ
        count = merge_multiple_datasets(
            input_files=args.inputs,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Merged and converted {count} samples")
    elif args.input:
        # 単一ファイルを変換
        count = convert_dataset_to_thinking_format(
            input_file=args.input,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Converted {count} samples")
    else:
        parser.error("Either --input or --inputs must be specified")


if __name__ == "__main__":
    main()





    
    with open(input_file, 'r', encoding='utf-8') as f, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                converted_sample = convert_sample_to_thinking_format(
                    sample,
                    default_system_message=default_system_message
                )
                
                if converted_sample is None:
                    skipped_count += 1
                    continue
                
                out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                converted_count += 1
                
                if converted_count % 1000 == 0:
                    logger.info(f"Converted {converted_count} samples...")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON decode error: {e}")
                skipped_count += 1
                continue
            except Exception as e:
                logger.warning(f"Line {line_num}: Error: {e}")
                skipped_count += 1
                continue
    
    logger.info(f"Conversion complete: {converted_count} samples converted, {skipped_count} skipped")
    logger.info(f"Saved to: {output_file}")
    
    return converted_count


def merge_multiple_datasets(
    input_files: List[Path],
    output_file: Path,
    default_system_message: Optional[str] = None
) -> int:
    """
    複数のデータセットをマージして/think形式に変換
    
    Args:
        input_files: 入力データセットファイルのリスト
        output_file: 出力データセットファイル
        default_system_message: デフォルトのシステムメッセージ
    
    Returns:
        マージされたサンプル数
    """
    total_count = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for input_file in input_files:
            logger.info(f"Processing: {input_file}")
            count = 0
            
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sample = json.loads(line)
                        converted_sample = convert_sample_to_thinking_format(
                            sample,
                            default_system_message=default_system_message
                        )
                        
                        if converted_sample is None:
                            continue
                        
                        out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                        count += 1
                        total_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error processing sample: {e}")
                        continue
            
            logger.info(f"  Processed {count} samples from {input_file}")
    
    logger.info(f"Merged {total_count} total samples")
    logger.info(f"Saved to: {output_file}")
    
    return total_count


def expand_dataset_gradually(
    base_dataset: Path,
    target_sizes: List[int],
    quality_threshold: float = 0.7,
    output_dir: Optional[Path] = None
) -> List[Path]:
    """
    データセットを段階的に拡張
    
    Args:
        base_dataset: ベースデータセットファイル
        target_sizes: 目標サイズのリスト（例: [5000, 10000, 25000, 50000]）
        quality_threshold: 品質スコアの閾値
        output_dir: 出力ディレクトリ（Noneの場合はbase_datasetと同じディレクトリ）
    
    Returns:
        各段階のデータセットファイルパスのリスト
    """
    if output_dir is None:
        output_dir = base_dataset.parent
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # ベースデータセットを読み込み
    logger.info(f"Loading base dataset from: {base_dataset}")
    all_samples = []
    with open(base_dataset, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                all_samples.append(sample)
            except json.JSONDecodeError:
                continue
    
    logger.info(f"Loaded {len(all_samples)} samples from base dataset")
    
    # 品質評価器を初期化
    evaluator = ThinkingDatasetQualityEvaluator()
    
    # 品質評価とフィルタリング
    logger.info(f"Evaluating quality with threshold: {quality_threshold}")
    quality_samples = evaluator.filter_by_quality(all_samples, min_score=quality_threshold)
    logger.info(f"Quality-filtered samples: {len(quality_samples)} / {len(all_samples)}")
    
    # 各段階でデータセットを拡張
    output_files = []
    current_samples = quality_samples.copy()
    
    for target_size in target_sizes:
        logger.info(f"Expanding to {target_size} samples...")
        
        # 目標サイズに達するまでサンプルを追加
        if len(current_samples) < target_size:
            # 不足分を既存サンプルからランダムに複製（簡易版）
            # 実際の実装では、新しいデータソースから追加することを推奨
            needed = target_size - len(current_samples)
            additional_samples = random.sample(quality_samples, min(needed, len(quality_samples)))
            current_samples.extend(additional_samples)
        
        # 目標サイズに合わせてサンプリング
        if len(current_samples) > target_size:
            current_samples = random.sample(current_samples, target_size)
        
        # 出力ファイル名を生成
        output_file = output_dir / f"{base_dataset.stem}_expanded_{target_size}.jsonl"
        
        # データセットを保存
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in current_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved expanded dataset: {output_file} ({len(current_samples)} samples)")
        output_files.append(output_file)
    
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Create /think format SFT dataset for Borea-Phi-3.5"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input dataset file (JSONL format)"
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        help="Multiple input dataset files (JSONL format) for merging"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset file (JSONL format)"
    )
    parser.add_argument(
        "--system-message",
        type=str,
        default=None,
        help="Custom system message (default: built-in thinking prompt)"
    )
    parser.add_argument(
        "--quality-filter",
        action="store_true",
        help="Enable quality filtering"
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.7,
        help="Minimum quality score for filtering (default: 0.7)"
    )
    parser.add_argument(
        "--expand-gradually",
        action="store_true",
        help="Expand dataset gradually (5000 -> 10000 -> 25000 -> 50000)"
    )
    parser.add_argument(
        "--target-sizes",
        type=int,
        nargs="+",
        default=[5000, 10000, 25000, 50000],
        help="Target sizes for gradual expansion (default: 5000 10000 25000 50000)"
    )
    
    args = parser.parse_args()
    
    if args.expand_gradually:
        # 段階的拡張
        if not args.input:
            parser.error("--expand-gradually requires --input")
        
        output_files = expand_dataset_gradually(
            base_dataset=args.input,
            target_sizes=args.target_sizes,
            quality_threshold=args.min_quality_score,
            output_dir=args.output.parent if args.output else None
        )
        logger.info(f"[SUCCESS] Expanded dataset to {len(output_files)} stages")
        for output_file in output_files:
            logger.info(f"  - {output_file}")
    
    elif args.inputs:
        # 複数ファイルをマージ
        count = merge_multiple_datasets(
            input_files=args.inputs,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Merged and converted {count} samples")
    elif args.input:
        # 単一ファイルを変換
        count = convert_dataset_to_thinking_format(
            input_file=args.input,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Converted {count} samples")
    else:
        parser.error("Either --input or --inputs must be specified")


if __name__ == "__main__":
    main()





    
    with open(input_file, 'r', encoding='utf-8') as f, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                converted_sample = convert_sample_to_thinking_format(
                    sample,
                    default_system_message=default_system_message
                )
                
                if converted_sample is None:
                    skipped_count += 1
                    continue
                
                out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                converted_count += 1
                
                if converted_count % 1000 == 0:
                    logger.info(f"Converted {converted_count} samples...")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON decode error: {e}")
                skipped_count += 1
                continue
            except Exception as e:
                logger.warning(f"Line {line_num}: Error: {e}")
                skipped_count += 1
                continue
    
    logger.info(f"Conversion complete: {converted_count} samples converted, {skipped_count} skipped")
    logger.info(f"Saved to: {output_file}")
    
    return converted_count


def merge_multiple_datasets(
    input_files: List[Path],
    output_file: Path,
    default_system_message: Optional[str] = None
) -> int:
    """
    複数のデータセットをマージして/think形式に変換
    
    Args:
        input_files: 入力データセットファイルのリスト
        output_file: 出力データセットファイル
        default_system_message: デフォルトのシステムメッセージ
    
    Returns:
        マージされたサンプル数
    """
    total_count = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for input_file in input_files:
            logger.info(f"Processing: {input_file}")
            count = 0
            
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sample = json.loads(line)
                        converted_sample = convert_sample_to_thinking_format(
                            sample,
                            default_system_message=default_system_message
                        )
                        
                        if converted_sample is None:
                            continue
                        
                        out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                        count += 1
                        total_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error processing sample: {e}")
                        continue
            
            logger.info(f"  Processed {count} samples from {input_file}")
    
    logger.info(f"Merged {total_count} total samples")
    logger.info(f"Saved to: {output_file}")
    
    return total_count


def expand_dataset_gradually(
    base_dataset: Path,
    target_sizes: List[int],
    quality_threshold: float = 0.7,
    output_dir: Optional[Path] = None
) -> List[Path]:
    """
    データセットを段階的に拡張
    
    Args:
        base_dataset: ベースデータセットファイル
        target_sizes: 目標サイズのリスト（例: [5000, 10000, 25000, 50000]）
        quality_threshold: 品質スコアの閾値
        output_dir: 出力ディレクトリ（Noneの場合はbase_datasetと同じディレクトリ）
    
    Returns:
        各段階のデータセットファイルパスのリスト
    """
    if output_dir is None:
        output_dir = base_dataset.parent
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # ベースデータセットを読み込み
    logger.info(f"Loading base dataset from: {base_dataset}")
    all_samples = []
    with open(base_dataset, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                all_samples.append(sample)
            except json.JSONDecodeError:
                continue
    
    logger.info(f"Loaded {len(all_samples)} samples from base dataset")
    
    # 品質評価器を初期化
    evaluator = ThinkingDatasetQualityEvaluator()
    
    # 品質評価とフィルタリング
    logger.info(f"Evaluating quality with threshold: {quality_threshold}")
    quality_samples = evaluator.filter_by_quality(all_samples, min_score=quality_threshold)
    logger.info(f"Quality-filtered samples: {len(quality_samples)} / {len(all_samples)}")
    
    # 各段階でデータセットを拡張
    output_files = []
    current_samples = quality_samples.copy()
    
    for target_size in target_sizes:
        logger.info(f"Expanding to {target_size} samples...")
        
        # 目標サイズに達するまでサンプルを追加
        if len(current_samples) < target_size:
            # 不足分を既存サンプルからランダムに複製（簡易版）
            # 実際の実装では、新しいデータソースから追加することを推奨
            needed = target_size - len(current_samples)
            additional_samples = random.sample(quality_samples, min(needed, len(quality_samples)))
            current_samples.extend(additional_samples)
        
        # 目標サイズに合わせてサンプリング
        if len(current_samples) > target_size:
            current_samples = random.sample(current_samples, target_size)
        
        # 出力ファイル名を生成
        output_file = output_dir / f"{base_dataset.stem}_expanded_{target_size}.jsonl"
        
        # データセットを保存
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in current_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved expanded dataset: {output_file} ({len(current_samples)} samples)")
        output_files.append(output_file)
    
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Create /think format SFT dataset for Borea-Phi-3.5"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input dataset file (JSONL format)"
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        help="Multiple input dataset files (JSONL format) for merging"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset file (JSONL format)"
    )
    parser.add_argument(
        "--system-message",
        type=str,
        default=None,
        help="Custom system message (default: built-in thinking prompt)"
    )
    parser.add_argument(
        "--quality-filter",
        action="store_true",
        help="Enable quality filtering"
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.7,
        help="Minimum quality score for filtering (default: 0.7)"
    )
    parser.add_argument(
        "--expand-gradually",
        action="store_true",
        help="Expand dataset gradually (5000 -> 10000 -> 25000 -> 50000)"
    )
    parser.add_argument(
        "--target-sizes",
        type=int,
        nargs="+",
        default=[5000, 10000, 25000, 50000],
        help="Target sizes for gradual expansion (default: 5000 10000 25000 50000)"
    )
    
    args = parser.parse_args()
    
    if args.expand_gradually:
        # 段階的拡張
        if not args.input:
            parser.error("--expand-gradually requires --input")
        
        output_files = expand_dataset_gradually(
            base_dataset=args.input,
            target_sizes=args.target_sizes,
            quality_threshold=args.min_quality_score,
            output_dir=args.output.parent if args.output else None
        )
        logger.info(f"[SUCCESS] Expanded dataset to {len(output_files)} stages")
        for output_file in output_files:
            logger.info(f"  - {output_file}")
    
    elif args.inputs:
        # 複数ファイルをマージ
        count = merge_multiple_datasets(
            input_files=args.inputs,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Merged and converted {count} samples")
    elif args.input:
        # 単一ファイルを変換
        count = convert_dataset_to_thinking_format(
            input_file=args.input,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Converted {count} samples")
    else:
        parser.error("Either --input or --inputs must be specified")


if __name__ == "__main__":
    main()





    
    with open(input_file, 'r', encoding='utf-8') as f, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                converted_sample = convert_sample_to_thinking_format(
                    sample,
                    default_system_message=default_system_message
                )
                
                if converted_sample is None:
                    skipped_count += 1
                    continue
                
                out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                converted_count += 1
                
                if converted_count % 1000 == 0:
                    logger.info(f"Converted {converted_count} samples...")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON decode error: {e}")
                skipped_count += 1
                continue
            except Exception as e:
                logger.warning(f"Line {line_num}: Error: {e}")
                skipped_count += 1
                continue
    
    logger.info(f"Conversion complete: {converted_count} samples converted, {skipped_count} skipped")
    logger.info(f"Saved to: {output_file}")
    
    return converted_count


def merge_multiple_datasets(
    input_files: List[Path],
    output_file: Path,
    default_system_message: Optional[str] = None
) -> int:
    """
    複数のデータセットをマージして/think形式に変換
    
    Args:
        input_files: 入力データセットファイルのリスト
        output_file: 出力データセットファイル
        default_system_message: デフォルトのシステムメッセージ
    
    Returns:
        マージされたサンプル数
    """
    total_count = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for input_file in input_files:
            logger.info(f"Processing: {input_file}")
            count = 0
            
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sample = json.loads(line)
                        converted_sample = convert_sample_to_thinking_format(
                            sample,
                            default_system_message=default_system_message
                        )
                        
                        if converted_sample is None:
                            continue
                        
                        out_f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                        count += 1
                        total_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error processing sample: {e}")
                        continue
            
            logger.info(f"  Processed {count} samples from {input_file}")
    
    logger.info(f"Merged {total_count} total samples")
    logger.info(f"Saved to: {output_file}")
    
    return total_count


def expand_dataset_gradually(
    base_dataset: Path,
    target_sizes: List[int],
    quality_threshold: float = 0.7,
    output_dir: Optional[Path] = None
) -> List[Path]:
    """
    データセットを段階的に拡張
    
    Args:
        base_dataset: ベースデータセットファイル
        target_sizes: 目標サイズのリスト（例: [5000, 10000, 25000, 50000]）
        quality_threshold: 品質スコアの閾値
        output_dir: 出力ディレクトリ（Noneの場合はbase_datasetと同じディレクトリ）
    
    Returns:
        各段階のデータセットファイルパスのリスト
    """
    if output_dir is None:
        output_dir = base_dataset.parent
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # ベースデータセットを読み込み
    logger.info(f"Loading base dataset from: {base_dataset}")
    all_samples = []
    with open(base_dataset, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                all_samples.append(sample)
            except json.JSONDecodeError:
                continue
    
    logger.info(f"Loaded {len(all_samples)} samples from base dataset")
    
    # 品質評価器を初期化
    evaluator = ThinkingDatasetQualityEvaluator()
    
    # 品質評価とフィルタリング
    logger.info(f"Evaluating quality with threshold: {quality_threshold}")
    quality_samples = evaluator.filter_by_quality(all_samples, min_score=quality_threshold)
    logger.info(f"Quality-filtered samples: {len(quality_samples)} / {len(all_samples)}")
    
    # 各段階でデータセットを拡張
    output_files = []
    current_samples = quality_samples.copy()
    
    for target_size in target_sizes:
        logger.info(f"Expanding to {target_size} samples...")
        
        # 目標サイズに達するまでサンプルを追加
        if len(current_samples) < target_size:
            # 不足分を既存サンプルからランダムに複製（簡易版）
            # 実際の実装では、新しいデータソースから追加することを推奨
            needed = target_size - len(current_samples)
            additional_samples = random.sample(quality_samples, min(needed, len(quality_samples)))
            current_samples.extend(additional_samples)
        
        # 目標サイズに合わせてサンプリング
        if len(current_samples) > target_size:
            current_samples = random.sample(current_samples, target_size)
        
        # 出力ファイル名を生成
        output_file = output_dir / f"{base_dataset.stem}_expanded_{target_size}.jsonl"
        
        # データセットを保存
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in current_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved expanded dataset: {output_file} ({len(current_samples)} samples)")
        output_files.append(output_file)
    
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Create /think format SFT dataset for Borea-Phi-3.5"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input dataset file (JSONL format)"
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        help="Multiple input dataset files (JSONL format) for merging"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset file (JSONL format)"
    )
    parser.add_argument(
        "--system-message",
        type=str,
        default=None,
        help="Custom system message (default: built-in thinking prompt)"
    )
    parser.add_argument(
        "--quality-filter",
        action="store_true",
        help="Enable quality filtering"
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.7,
        help="Minimum quality score for filtering (default: 0.7)"
    )
    parser.add_argument(
        "--expand-gradually",
        action="store_true",
        help="Expand dataset gradually (5000 -> 10000 -> 25000 -> 50000)"
    )
    parser.add_argument(
        "--target-sizes",
        type=int,
        nargs="+",
        default=[5000, 10000, 25000, 50000],
        help="Target sizes for gradual expansion (default: 5000 10000 25000 50000)"
    )
    
    args = parser.parse_args()
    
    if args.expand_gradually:
        # 段階的拡張
        if not args.input:
            parser.error("--expand-gradually requires --input")
        
        output_files = expand_dataset_gradually(
            base_dataset=args.input,
            target_sizes=args.target_sizes,
            quality_threshold=args.min_quality_score,
            output_dir=args.output.parent if args.output else None
        )
        logger.info(f"[SUCCESS] Expanded dataset to {len(output_files)} stages")
        for output_file in output_files:
            logger.info(f"  - {output_file}")
    
    elif args.inputs:
        # 複数ファイルをマージ
        count = merge_multiple_datasets(
            input_files=args.inputs,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Merged and converted {count} samples")
    elif args.input:
        # 単一ファイルを変換
        count = convert_dataset_to_thinking_format(
            input_file=args.input,
            output_file=args.output,
            default_system_message=args.system_message
        )
        
        # 品質フィルタリング（オプション）
        if args.quality_filter:
            logger.info("Applying quality filtering...")
            evaluator = ThinkingDatasetQualityEvaluator()
            filtered_samples = []
            with open(args.output, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
            filtered_samples = evaluator.filter_by_quality(samples, min_score=args.min_quality_score)
            
            # フィルタリングされたデータセットを保存
            filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
            with open(filtered_output, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"[SUCCESS] Quality-filtered dataset saved: {filtered_output} ({len(filtered_samples)} / {len(samples)} samples)")
        
        logger.info(f"[SUCCESS] Converted {count} samples")
    else:
        parser.error("Either --input or --inputs must be specified")


if __name__ == "__main__":
    main()



