#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四重推論データセットのトークン数圧縮前処理スクリプト

構造（<think-task>, <think-safety>, <think-policy>, <final>）を維持しつつ、
トークン数ベースで圧縮する。
"""

import json
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import sys

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer
from tqdm import tqdm

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/compress_thinking_dataset.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def count_tokens(text: str, tokenizer) -> int:
    """
    テキストのトークン数をカウント
    
    Args:
        text: テキスト
        tokenizer: トークナイザー
    
    Returns:
        トークン数
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def extract_thinking_sections(output: str) -> Dict[str, str]:
    """
    outputから各thinkingセクションを抽出
    
    Args:
        output: 出力テキスト（Phi-3.5チャットテンプレート形式）
    
    Returns:
        セクション辞書: {'think-task': ..., 'think-safety': ..., 'think-policy': ..., 'final': ...}
    """
    sections = {
        'think-task': '',
        'think-safety': '',
        'think-policy': '',
        'final': ''
    }
    
    # <think-task>...</think-task>
    match = re.search(r'<think-task>(.*?)</think-task>', output, re.DOTALL)
    if match:
        sections['think-task'] = match.group(1).strip()
    
    # <think-safety>...</think-safety>
    match = re.search(r'<think-safety>(.*?)</think-safety>', output, re.DOTALL)
    if match:
        sections['think-safety'] = match.group(1).strip()
    
    # <think-policy>...</think-policy>
    match = re.search(r'<think-policy>(.*?)</think-policy>', output, re.DOTALL)
    if match:
        sections['think-policy'] = match.group(1).strip()
    
    # <final>...</final>
    match = re.search(r'<final>(.*?)</final>', output, re.DOTALL)
    if match:
        sections['final'] = match.group(1).strip()
    
    return sections


def trim_text_to_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """
    テキストを指定トークン数までトリム
    
    Args:
        text: テキスト
        tokenizer: トークナイザー
        max_tokens: 最大トークン数
    
    Returns:
        トリムされたテキスト
    """
    if not text:
        return text
    
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    
    # トークン数を制限
    trimmed_tokens = tokens[:max_tokens]
    trimmed_text = tokenizer.decode(trimmed_tokens, skip_special_tokens=True)
    
    return trimmed_text


def trim_thinking_section(
    section_text: str,
    tokenizer,
    max_tokens: int
) -> str:
    """
    特定の<think-*>セクションをトークン数上限でトリム
    
    Args:
        section_text: セクションのテキスト内容（タグなし）
        tokenizer: トークナイザー
        max_tokens: 最大トークン数
    
    Returns:
        トリムされたセクションテキスト
    """
    return trim_text_to_tokens(section_text, tokenizer, max_tokens)


def rebuild_output_with_sections(
    sections: Dict[str, str],
    template_prefix: str,
    template_suffix: str
) -> str:
    """
    セクションからoutputテキストを再構築
    
    Args:
        sections: セクション辞書
        template_prefix: テンプレートの前部分（<|assistant|>まで）
        template_suffix: テンプレートの後部分（<|end|>以降）
    
    Returns:
        再構築されたoutputテキスト
    """
    thinking_parts = []
    
    if sections['think-task']:
        thinking_parts.append(f"<think-task>\n{sections['think-task']}\n</think-task>")
    
    if sections['think-safety']:
        thinking_parts.append(f"<think-safety>\n{sections['think-safety']}\n</think-safety>")
    
    if sections['think-policy']:
        thinking_parts.append(f"<think-policy>\n{sections['think-policy']}\n</think-policy>")
    
    if sections['final']:
        thinking_parts.append(f"<final>\n{sections['final']}\n</final>")
    
    assistant_content = "\n\n".join(thinking_parts)
    
    return f"{template_prefix}\n{assistant_content}\n{template_suffix}"


def compress_thinking_sample(
    sample: Dict[str, Any],
    tokenizer,
    max_total_tokens: int = 6144,
    max_think_section_tokens: int = 1024,
    min_final_tokens: int = 256
) -> Optional[Dict[str, Any]]:
    """
    1サンプルを圧縮
    
    Args:
        sample: サンプル辞書
        tokenizer: トークナイザー
        max_total_tokens: 全体の最大トークン数
        max_think_section_tokens: 各<think-*>セクションの最大トークン数
        min_final_tokens: <final>セクションの最小保持トークン数
    
    Returns:
        圧縮されたサンプル（Noneの場合は除外）
    """
    output = sample.get("output", "")
    if not output:
        return None
    
    # テンプレートの前後部分を分離
    # <|assistant|>の前までと、<|end|>以降を保持
    assistant_match = re.search(r'(.*<\|assistant\|>\n)', output, re.DOTALL)
    if not assistant_match:
        logger.warning("Could not find <|assistant|> tag, skipping sample")
        return None
    
    template_prefix = assistant_match.group(1)
    
    # <|end|>以降を抽出
    end_match = re.search(r'(<\|end\|>.*)$', output, re.DOTALL)
    if end_match:
        template_suffix = end_match.group(1)
    else:
        template_suffix = "<|end|>"
    
    # セクションを抽出
    sections = extract_thinking_sections(output)
    
    # 各セクションのトークン数をカウント
    section_tokens = {}
    for key, text in sections.items():
        section_tokens[key] = count_tokens(text, tokenizer) if text else 0
    
    # テンプレート部分のトークン数
    template_tokens = count_tokens(template_prefix + template_suffix, tokenizer)
    
    # 全体のトークン数を計算
    total_tokens = template_tokens + sum(section_tokens.values())
    
    # 既に上限内ならそのまま返す
    if total_tokens <= max_total_tokens:
        return sample
    
    # 圧縮が必要
    # 優先順位: <final> > <think-task> > <think-safety> > <think-policy>
    
    # 1. <final>セクションを最小保持トークン数で保持
    if sections['final']:
        final_tokens = section_tokens['final']
        if final_tokens > min_final_tokens:
            # 最小保持トークン数までトリム
            sections['final'] = trim_thinking_section(
                sections['final'],
                tokenizer,
                min_final_tokens
            )
            section_tokens['final'] = count_tokens(sections['final'], tokenizer)
    
    # 2. 各<think-*>セクションを最大トークン数でトリム
    think_sections = ['think-task', 'think-safety', 'think-policy']
    for section_key in think_sections:
        if sections[section_key] and section_tokens[section_key] > max_think_section_tokens:
            sections[section_key] = trim_thinking_section(
                sections[section_key],
                tokenizer,
                max_think_section_tokens
            )
            section_tokens[section_key] = count_tokens(sections[section_key], tokenizer)
    
    # 3. 再度全体トークン数を計算
    total_tokens = template_tokens + sum(section_tokens.values())
    
    # 4. まだ上限を超える場合は、比例的に削減
    if total_tokens > max_total_tokens:
        # 利用可能なトークン数（テンプレート分を除く）
        available_tokens = max_total_tokens - template_tokens
        
        if available_tokens < min_final_tokens:
            # <final>の最小保持トークン数すら確保できない場合は除外
            return None
        
        # <final>を優先的に確保
        final_tokens_allocated = min(section_tokens['final'], available_tokens - 100)  # 100トークンのバッファ
        if final_tokens_allocated < min_final_tokens:
            final_tokens_allocated = min_final_tokens
        
        remaining_tokens = available_tokens - final_tokens_allocated
        
        # 残りのトークンを各<think-*>セクションに比例配分
        think_total_tokens = sum(section_tokens[key] for key in think_sections)
        if think_total_tokens > 0:
            for section_key in think_sections:
                if sections[section_key]:
                    ratio = section_tokens[section_key] / think_total_tokens
                    allocated = int(remaining_tokens * ratio)
                    if allocated > 0:
                        sections[section_key] = trim_thinking_section(
                            sections[section_key],
                            tokenizer,
                            allocated
                        )
        
        # <final>も必要に応じて調整
        if section_tokens['final'] > final_tokens_allocated:
            sections['final'] = trim_thinking_section(
                sections['final'],
                tokenizer,
                final_tokens_allocated
            )
    
    # 5. outputを再構築
    compressed_output = rebuild_output_with_sections(
        sections,
        template_prefix,
        template_suffix
    )
    
    # 6. 圧縮後のサンプルを作成
    compressed_sample = sample.copy()
    compressed_sample['output'] = compressed_output
    
    return compressed_sample


def compress_dataset(
    input_file: Path,
    output_file: Path,
    tokenizer_path: str,
    max_total_tokens: int = 6144,
    max_think_section_tokens: int = 1024,
    min_final_tokens: int = 256
) -> Dict[str, Any]:
    """
    データセット全体を圧縮
    
    Args:
        input_file: 入力データセットファイル（JSONL形式）
        output_file: 出力データセットファイル（JSONL形式）
        tokenizer_path: トークナイザーパス
        max_total_tokens: 全体の最大トークン数
        max_think_section_tokens: 各<think-*>セクションの最大トークン数
        min_final_tokens: <final>セクションの最小保持トークン数
    
    Returns:
        圧縮統計情報
    """
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Compressing dataset: {input_file}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Max total tokens: {max_total_tokens}")
    logger.info(f"Max think section tokens: {max_think_section_tokens}")
    logger.info(f"Min final tokens: {min_final_tokens}")
    
    stats = {
        'total_samples': 0,
        'compressed_samples': 0,
        'skipped_samples': 0,
        'excluded_samples': 0,
        'original_tokens': [],
        'compressed_tokens': [],
        'compression_ratios': []
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 総行数を事前にカウント（進捗バー用）
    logger.info("Counting total lines in input file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    logger.info(f"Total lines: {total_lines:,}")
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        # tqdmで進捗バーを表示
        pbar = tqdm(
            enumerate(f_in, 1),
            total=total_lines,
            desc="Compressing",
            unit="samples",
            ncols=100
        )
        
        for line_num, line in pbar:
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                stats['total_samples'] += 1
                
                # 元のトークン数をカウント
                original_output = sample.get("output", "")
                original_token_count = count_tokens(original_output, tokenizer)
                stats['original_tokens'].append(original_token_count)
                
                # 圧縮
                compressed_sample = compress_thinking_sample(
                    sample,
                    tokenizer,
                    max_total_tokens=max_total_tokens,
                    max_think_section_tokens=max_think_section_tokens,
                    min_final_tokens=min_final_tokens
                )
                
                if compressed_sample is None:
                    stats['excluded_samples'] += 1
                    logger.debug(f"Line {line_num}: Excluded (too long even after compression)")
                    continue
                
                # 圧縮後のトークン数をカウント
                compressed_output = compressed_sample.get("output", "")
                compressed_token_count = count_tokens(compressed_output, tokenizer)
                stats['compressed_tokens'].append(compressed_token_count)
                
                # 圧縮率を計算
                if original_token_count > 0:
                    ratio = compressed_token_count / original_token_count
                    stats['compression_ratios'].append(ratio)
                
                # 圧縮されたかどうか
                if compressed_token_count < original_token_count:
                    stats['compressed_samples'] += 1
                else:
                    stats['skipped_samples'] += 1
                
                # 出力
                f_out.write(json.dumps(compressed_sample, ensure_ascii=False) + '\n')
                
                # 進捗バーを更新
                pbar.set_postfix({
                    'compressed': stats['compressed_samples'],
                    'excluded': stats['excluded_samples'],
                    'skipped': stats['skipped_samples']
                })
                
                # 詳細ログ（100サンプルごと）
                if line_num % 100 == 0:
                    logger.debug(f"Processed {line_num}/{total_lines} samples (compressed: {stats['compressed_samples']}, excluded: {stats['excluded_samples']}, skipped: {stats['skipped_samples']})")
            
            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: JSON decode error: {e}")
                stats['skipped_samples'] += 1
                pbar.set_postfix({
                    'compressed': stats['compressed_samples'],
                    'excluded': stats['excluded_samples'],
                    'skipped': stats['skipped_samples'],
                    'error': 'JSON decode'
                })
                continue
            except Exception as e:
                logger.error(f"Line {line_num}: Unexpected error: {e}", exc_info=True)
                stats['skipped_samples'] += 1
                pbar.set_postfix({
                    'compressed': stats['compressed_samples'],
                    'excluded': stats['excluded_samples'],
                    'skipped': stats['skipped_samples'],
                    'error': str(e)[:20]
                })
                continue
        
        # 進捗バーを閉じる
        pbar.close()
    
    return stats


def print_compression_statistics(stats: Dict[str, Any]):
    """圧縮統計情報を表示"""
    logger.info("=" * 80)
    logger.info("COMPRESSION STATISTICS")
    logger.info("=" * 80)
    
    logger.info("\n[OVERVIEW]")
    logger.info(f"  Total samples: {stats['total_samples']:,}")
    logger.info(f"  Compressed samples: {stats['compressed_samples']:,} ({stats['compressed_samples']/max(stats['total_samples'], 1)*100:.1f}%)")
    logger.info(f"  Skipped samples (no compression needed): {stats['skipped_samples']:,} ({stats['skipped_samples']/max(stats['total_samples'], 1)*100:.1f}%)")
    logger.info(f"  Excluded samples (too long): {stats['excluded_samples']:,} ({stats['excluded_samples']/max(stats['total_samples'], 1)*100:.1f}%)")
    
    if stats['original_tokens']:
        logger.info("\n[ORIGINAL TOKENS]")
        logger.info(f"  Average: {sum(stats['original_tokens'])/len(stats['original_tokens']):.0f}")
        logger.info(f"  Min: {min(stats['original_tokens']):,}")
        logger.info(f"  Max: {max(stats['original_tokens']):,}")
    
    if stats['compressed_tokens']:
        logger.info("\n[COMPRESSED TOKENS]")
        logger.info(f"  Average: {sum(stats['compressed_tokens'])/len(stats['compressed_tokens']):.0f}")
        logger.info(f"  Min: {min(stats['compressed_tokens']):,}")
        logger.info(f"  Max: {max(stats['compressed_tokens']):,}")
    
    if stats['compression_ratios']:
        logger.info("\n[COMPRESSION RATIO]")
        avg_ratio = sum(stats['compression_ratios']) / len(stats['compression_ratios'])
        logger.info(f"  Average: {avg_ratio:.2%}")
        logger.info(f"  Min: {min(stats['compression_ratios']):.2%}")
        logger.info(f"  Max: {max(stats['compression_ratios']):.2%}")
    
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Compress quadruple thinking dataset by token count"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input dataset file (JSONL format)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output dataset file (JSONL format). If not specified, will be generated automatically."
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="models/Borea-Phi-3.5-mini-Instruct-Jp",
        help="Tokenizer path (default: models/Borea-Phi-3.5-mini-Instruct-Jp)"
    )
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=6144,
        help="Maximum total tokens per sample (default: 6144)"
    )
    parser.add_argument(
        "--max-think-section-tokens",
        type=int,
        default=1024,
        help="Maximum tokens per <think-*> section (default: 1024)"
    )
    parser.add_argument(
        "--min-final-tokens",
        type=int,
        default=256,
        help="Minimum tokens for <final> section (default: 256)"
    )
    
    args = parser.parse_args()
    
    # 出力ファイル名を生成（指定されていない場合）
    if args.output is None:
        input_stem = args.input.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("D:/webdataset/processed/thinking_quadruple")
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = output_dir / f"{input_stem}_compressed_{timestamp}.jsonl"
    
    # 入力ファイルの存在確認
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return
    
    # 圧縮実行
    stats = compress_dataset(
        input_file=args.input,
        output_file=args.output,
        tokenizer_path=args.tokenizer,
        max_total_tokens=args.max_total_tokens,
        max_think_section_tokens=args.max_think_section_tokens,
        min_final_tokens=args.min_final_tokens
    )
    
    # 統計情報を表示
    print_compression_statistics(stats)
    
    logger.info(f"\n[SUCCESS] Compressed dataset saved to: {args.output}")


if __name__ == "__main__":
    main()

