#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
50kデータセット確認スクリプト
既存の50kデータセットの内容を確認
"""

import os
import sys
from pathlib import Path
import json

# Windows cp932エンコーディング対策
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

def check_50k_datasets():
    """既存の50kデータセットを確認"""
    datasets_50k = [
        'data/so8t_training_dataset_integrated_50k.jsonl',
        'data/integrated_large_sft_dataset.jsonl'
    ]

    for dataset_path in datasets_50k:
        path = Path(dataset_path)
        if path.exists():
            print(f'\n=== {dataset_path} ===')

            # 最初の数行を確認
            with open(path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 3:  # 最初の3行だけ
                        break
                    try:
                        item = json.loads(line.strip())
                        print(f'Line {i+1}: {list(item.keys())}')
                        if 'text' in item:
                            print(f'  Text preview: {item["text"][:100]}...')
                        elif 'instruction' in item and 'output' in item:
                            print(f'  Instruction preview: {item["instruction"][:50]}...')
                            print(f'  Output preview: {item["output"][:50]}...')
                    except Exception as e:
                        print(f'Line {i+1}: Parse error - {e}')

            # 総行数をカウント
            with open(path, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)
            print(f'Total lines: {total_lines}')

            # instruction-output形式のサンプルを確認
            with open(path, 'r', encoding='utf-8') as f:
                instruction_output_count = 0
                text_only_count = 0
                other_count = 0

                for line_num, line in enumerate(f):
                    if line_num >= 1000:  # 1000件確認したら停止
                        break
                    try:
                        item = json.loads(line.strip())
                        if 'instruction' in item and 'output' in item:
                            instruction_output_count += 1
                        elif 'text' in item:
                            text_only_count += 1
                        else:
                            other_count += 1
                    except:
                        pass

                print(f'Instruction-Output format: {instruction_output_count}')
                print(f'Text-only format: {text_only_count}')
                print(f'Other format: {other_count}')
                print(f'Sample ratio: {instruction_output_count/(instruction_output_count+text_only_count+other_count)*100:.1f}% instruction-output')

def create_final_sft_dataset():
    """最終的なSFTデータセット作成"""
    print("\n=== Creating Final AEGIS v2.1 SFT 50k Dataset ===")

    # 最も高品質なデータセットを使用
    source_dataset = 'data/integrated_large_sft_dataset.jsonl'
    target_dataset = 'data/aegis_v21_sft_50k_final.jsonl'

    if not Path(source_dataset).exists():
        print(f"[ERROR] Source dataset not found: {source_dataset}")
        return

    collected_samples = []

    print(f"Processing: {source_dataset}")

    with open(source_dataset, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if len(collected_samples) >= 50000:
                break

            try:
                item = json.loads(line.strip())

                # text形式をinstruction-output形式に変換
                if 'text' in item and 'instruction' not in item:
                    text = item['text'].strip()
                    # テキストをinstructionとoutputに分割
                    if 'Instruction:' in text and 'Output:' in text:
                        # 既に構造化されている場合
                        parts = text.split('Output:', 1)
                        if len(parts) == 2:
                            instruction_part = parts[0].replace('Instruction:', '').strip()
                            output_part = parts[1].strip()

                            standardized_item = {
                                'instruction': instruction_part,
                                'output': output_part,
                                'source_dataset': 'integrated_large_sft_dataset',
                                'converted_from': 'text_format',
                                'original_metadata': item.get('metadata', {}),
                                'original_source': item.get('source', '')
                            }
                            collected_samples.append(standardized_item)
                    else:
                        # 単一テキストの場合、instructionを生成
                        instruction = f"以下のテキストについて考え、適切に応答してください：\\n\\n{text[:200]}..."
                        output = text

                        standardized_item = {
                            'instruction': instruction,
                            'output': output,
                            'source_dataset': 'integrated_large_sft_dataset',
                            'converted_from': 'single_text',
                            'original_metadata': item.get('metadata', {}),
                            'original_source': item.get('source', '')
                        }
                        collected_samples.append(standardized_item)
                elif 'instruction' in item and 'output' in item:
                    # 既に正しい形式の場合
                    standardized_item = item.copy()
                    standardized_item['source_dataset'] = 'integrated_large_sft_dataset'
                    standardized_item['converted_from'] = 'already_formatted'
                    collected_samples.append(standardized_item)

            except Exception as e:
                print(f"[WARNING] Error processing line {line_num}: {e}")
                continue

    print(f"Collected {len(collected_samples)} samples")

    # シャッフル
    import random
    random.shuffle(collected_samples)

    # 50k件に制限
    final_samples = collected_samples[:50000]

    # 保存
    print(f"Saving to: {target_dataset}")
    with open(target_dataset, 'w', encoding='utf-8') as f:
        for item in final_samples:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

    # 統計
    stats = {
        'total_samples': len(final_samples),
        'source_dataset': source_dataset,
        'conversion_stats': {
            'text_format': sum(1 for s in final_samples if s.get('converted_from') == 'text_format'),
            'single_text': sum(1 for s in final_samples if s.get('converted_from') == 'single_text'),
            'already_formatted': sum(1 for s in final_samples if s.get('converted_from') == 'already_formatted')
        }
    }

    stats_file = target_dataset.replace('.jsonl', '.stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Statistics saved to: {stats_file}")
    print(f"Final dataset: {len(final_samples)} samples")

    return final_samples, stats

def main():
    """メイン処理"""
    print("[START] 50k SFT Dataset Analysis")
    print("=" * 40)

    # 既存データセットの確認
    check_50k_datasets()

    # 最終データセット作成
    samples, stats = create_final_sft_dataset()

    print("\n[SUCCESS] AEGIS v2.1 SFT 50k Dataset Created!")
    print(f"  Dataset: data/aegis_v21_sft_50k_final.jsonl")
    print(f"  Samples: {stats['total_samples']}")
    print(f"  Text format conversions: {stats['conversion_stats']['text_format']}")
    print(f"  Single text conversions: {stats['conversion_stats']['single_text']}")
    print(f"  Already formatted: {stats['conversion_stats']['already_formatted']}")

if __name__ == "__main__":
    main()
