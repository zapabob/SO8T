#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四重推論形式データセット作成スクリプト

既存のデータセット（four_class、domain_knowledge等）を四重推論形式に変換し、
Phi-3.5チャットテンプレート形式で出力する。
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "so8t-mmllm" / "src"))

from models.thinking_tokens import format_quadruple_thinking_output

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/create_quadruple_thinking_dataset.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


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
        assistant_message: アシスタントメッセージ（四重推論を含む）
    
    Returns:
        フォーマット済みテキスト
    """
    template = f"<|system|>\n{system_message}<|end|>\n<|user|>\n{user_message}<|end|>\n<|assistant|>\n{assistant_message}<|end|>"
    return template


def convert_sample_to_quadruple_thinking(
    sample: Dict[str, Any],
    default_system_message: str = "あなたは慎重に考えるAIアシスタントです。四段階の内部推論（Task/Safety/Policy）を行い、その後<final>で日本語で回答してください。"
) -> Optional[Dict[str, Any]]:
    """
    サンプルを四重推論形式に変換
    
    Args:
        sample: 入力サンプル
        default_system_message: デフォルトのシステムメッセージ
    
    Returns:
        変換されたサンプル（Noneの場合はスキップ）
    """
    from scripts.data.create_thinking_dataset import convert_to_quadruple_thinking_format
    
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output = sample.get("output", "")
    text = sample.get("text", "")
    
    # textフィールドがある場合は、それをinstructionとして使用
    if not instruction and not input_text and text:
        title = sample.get("title", "")
        keyword = sample.get("keyword", "")
        if title:
            instruction = f"{title}について説明してください。"
        elif keyword:
            instruction = f"{keyword}について説明してください。"
        else:
            instruction = f"以下の内容について説明してください。\n\n{text[:200]}..."
        input_text = text
    
    # 既に四重推論形式の場合はそのまま使用
    if "<think-task>" in output and "<think-safety>" in output and "<think-policy>" in output and "<final>" in output:
        thinking_output = output
    elif output or text:
        # 四重推論形式に変換
        safety_label = sample.get("safety_label", sample.get("four_class_label", "ALLOW"))
        policy_domain = sample.get("policy_domain", sample.get("domain_label", "general"))
        domain_label = sample.get("domain_label", None)
        
        thinking_output = convert_to_quadruple_thinking_format(
            instruction=instruction,
            input_text=input_text,
            output=output or text,
            safety_label=safety_label,
            policy_domain=policy_domain,
            domain_label=domain_label,
            text=text,
        )
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
    
    # システムメッセージを取得
    system_message = sample.get("system", default_system_message)
    
    # チャットテンプレート形式にフォーマット
    formatted_text = format_phi35_chat_template(
        system_message=system_message,
        user_message=user_message,
        assistant_message=thinking_output
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
    if "policy_domain" in sample:
        new_sample["policy_domain"] = sample["policy_domain"]
    if "system" in sample:
        new_sample["system"] = sample["system"]
    
    return new_sample


def convert_dataset_to_quadruple_thinking(
    input_file: Path,
    output_file: Path,
    default_system_message: Optional[str] = None
) -> int:
    """
    データセットを四重推論形式に変換
    
    Args:
        input_file: 入力データセットファイル（JSONL形式）
        output_file: 出力データセットファイル（JSONL形式）
        default_system_message: デフォルトのシステムメッセージ
    
    Returns:
        変換されたサンプル数
    """
    if default_system_message is None:
        default_system_message = "あなたは慎重に考えるAIアシスタントです。四段階の内部推論（Task/Safety/Policy）を行い、その後<final>で日本語で回答してください。"
    
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
                converted_sample = convert_sample_to_quadruple_thinking(
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
    複数のデータセットをマージして四重推論形式に変換
    
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
                        converted_sample = convert_sample_to_quadruple_thinking(
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


def validate_quadruple_thinking_dataset(dataset_file: Path) -> Dict[str, Any]:
    """
    四重推論形式データセットの品質を検証
    
    Args:
        dataset_file: データセットファイルパス
    
    Returns:
        検証結果の辞書
    """
    results = {
        "total_samples": 0,
        "valid_samples": 0,
        "invalid_samples": 0,
        "has_task": 0,
        "has_safety": 0,
        "has_policy": 0,
        "has_final": 0,
        "has_all_sections": 0,
        "errors": []
    }
    
    logger.info(f"Validating dataset: {dataset_file}")
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                results["total_samples"] += 1
                
                output = sample.get("output", "")
                
                # 各セクションの存在チェック
                has_task = "<think-task>" in output
                has_safety = "<think-safety>" in output
                has_policy = "<think-policy>" in output
                has_final = "<final>" in output
                
                if has_task:
                    results["has_task"] += 1
                if has_safety:
                    results["has_safety"] += 1
                if has_policy:
                    results["has_policy"] += 1
                if has_final:
                    results["has_final"] += 1
                
                if has_task and has_safety and has_policy and has_final:
                    results["has_all_sections"] += 1
                    results["valid_samples"] += 1
                else:
                    results["invalid_samples"] += 1
                    missing = []
                    if not has_task:
                        missing.append("think-task")
                    if not has_safety:
                        missing.append("think-safety")
                    if not has_policy:
                        missing.append("think-policy")
                    if not has_final:
                        missing.append("final")
                    results["errors"].append({
                        "line": line_num,
                        "missing_sections": missing
                    })
                    
            except Exception as e:
                results["invalid_samples"] += 1
                results["errors"].append({
                    "line": line_num,
                    "error": str(e)
                })
    
    # 検証結果をログ出力
    logger.info("="*80)
    logger.info("Validation Results")
    logger.info("="*80)
    logger.info(f"Total samples: {results['total_samples']}")
    logger.info(f"Valid samples: {results['valid_samples']} ({results['valid_samples']/max(results['total_samples'], 1)*100:.1f}%)")
    logger.info(f"Invalid samples: {results['invalid_samples']}")
    logger.info(f"Samples with <think-task>: {results['has_task']}")
    logger.info(f"Samples with <think-safety>: {results['has_safety']}")
    logger.info(f"Samples with <think-policy>: {results['has_policy']}")
    logger.info(f"Samples with <final>: {results['has_final']}")
    logger.info(f"Samples with all sections: {results['has_all_sections']}")
    logger.info("="*80)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Create quadruple thinking format dataset for SO8T/thinking model"
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
        help="Custom system message (default: built-in quadruple thinking prompt)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the output dataset after conversion"
    )
    
    args = parser.parse_args()
    
    if args.inputs:
        # 複数ファイルをマージ
        count = merge_multiple_datasets(
            input_files=args.inputs,
            output_file=args.output,
            default_system_message=args.system_message
        )
        logger.info(f"[SUCCESS] Merged and converted {count} samples")
    elif args.input:
        # 単一ファイルを変換
        count = convert_dataset_to_quadruple_thinking(
            input_file=args.input,
            output_file=args.output,
            default_system_message=args.system_message
        )
        logger.info(f"[SUCCESS] Converted {count} samples")
    else:
        parser.error("Either --input or --inputs must be specified")
    
    # 検証
    if args.validate:
        validate_quadruple_thinking_dataset(args.output)


if __name__ == "__main__":
    main()


