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
from pathlib import Path
from typing import List, Dict, Any, Optional
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
        count = convert_dataset_to_thinking_format(
            input_file=args.input,
            output_file=args.output,
            default_system_message=args.system_message
        )
        logger.info(f"[SUCCESS] Converted {count} samples")
    else:
        parser.error("Either --input or --inputs must be specified")


if __name__ == "__main__":
    main()




