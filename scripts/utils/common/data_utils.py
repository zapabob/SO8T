# -*- coding: utf-8 -*-
"""
Common Data Utilities
共通データユーティリティ
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


def calculate_dataset_stats(dataset_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    データセットの統計情報を計算

    Args:
        dataset_path: データセットパス

    Returns:
        統計情報辞書またはNone
    """
    dataset_path = Path(dataset_path)

    try:
        if dataset_path.suffix.lower() == '.jsonl':
            return _calculate_jsonl_stats(dataset_path)
        elif dataset_path.suffix.lower() == '.json':
            return _calculate_json_stats(dataset_path)
        elif dataset_path.suffix.lower() in ['.csv', '.tsv']:
            return _calculate_csv_stats(dataset_path)
        else:
            logger.warning(f"Unsupported file format: {dataset_path.suffix}")
            return None

    except Exception as e:
        logger.error(f"Failed to calculate dataset stats: {e}")
        return None


def _calculate_jsonl_stats(file_path: Path) -> Dict[str, Any]:
    """JSONLファイルの統計を計算"""
    stats = {
        'total_samples': 0,
        'avg_text_length': 0,
        'max_text_length': 0,
        'min_text_length': float('inf'),
        'fields': set()
    }

    total_text_length = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                sample = json.loads(line.strip())
                stats['total_samples'] += 1

                # テキスト長計算
                if 'text' in sample:
                    text_len = len(sample['text'])
                    total_text_length += text_len
                    stats['max_text_length'] = max(stats['max_text_length'], text_len)
                    stats['min_text_length'] = min(stats['min_text_length'], text_len)

                # フィールド収集
                stats['fields'].update(sample.keys())

            except json.JSONDecodeError:
                continue

    if stats['total_samples'] > 0:
        stats['avg_text_length'] = total_text_length / stats['total_samples']

    stats['fields'] = list(stats['fields'])
    stats['min_text_length'] = stats['min_text_length'] if stats['min_text_length'] != float('inf') else 0

    return stats


def _calculate_json_stats(file_path: Path) -> Dict[str, Any]:
    """JSONファイルの統計を計算"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        # リスト形式の場合
        return _calculate_jsonl_stats(file_path)
    elif isinstance(data, dict):
        # 辞書形式の場合
        return {
            'type': 'dict',
            'keys': list(data.keys()),
            'size': len(str(data))
        }
    else:
        return {'type': 'unknown'}


def _calculate_csv_stats(file_path: Path) -> Dict[str, Any]:
    """CSVファイルの統計を計算"""
    try:
        df = pd.read_csv(file_path)
        return {
            'total_samples': len(df),
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return None


def validate_dataset_format(dataset_path: Union[str, Path], expected_format: str = 'jsonl') -> bool:
    """
    データセット形式の妥当性を検証

    Args:
        dataset_path: データセットパス
        expected_format: 期待される形式

    Returns:
        妥当かどうか
    """
    dataset_path = Path(dataset_path)

    if expected_format == 'jsonl':
        return _validate_jsonl_format(dataset_path)
    elif expected_format == 'json':
        return _validate_json_format(dataset_path)
    else:
        logger.warning(f"Unsupported validation format: {expected_format}")
        return False


def _validate_jsonl_format(file_path: Path) -> bool:
    """JSONL形式の検証"""
    if not file_path.exists():
        return False

    valid_lines = 0
    total_lines = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                total_lines = line_num
                line = line.strip()
                if not line:  # 空行はスキップ
                    continue

                try:
                    json.loads(line)
                    valid_lines += 1
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num}: {e}")
                    continue

        # 80%以上の行が有効ならOK
        validity_ratio = valid_lines / total_lines if total_lines > 0 else 0
        return validity_ratio >= 0.8

    except Exception as e:
        logger.error(f"Failed to validate JSONL: {e}")
        return False


def _validate_json_format(file_path: Path) -> bool:
    """JSON形式の検証"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
        return True
    except Exception as e:
        return False


def merge_datasets(output_path: Union[str, Path], *input_paths: Union[str, Path]) -> bool:
    """
    複数のデータセットをマージ

    Args:
        output_path: 出力パス
        *input_paths: 入力データセットパス

    Returns:
        成功かどうか
    """
    output_path = Path(output_path)

    try:
        merged_data = []

        for input_path in input_paths:
            input_path = Path(input_path)

            if input_path.suffix.lower() == '.jsonl':
                with open(input_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            merged_data.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
            elif input_path.suffix.lower() == '.json':
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        merged_data.extend(data)
                    else:
                        merged_data.append(data)

        # マージデータを保存
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in merged_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"Merged {len(input_paths)} datasets into {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to merge datasets: {e}")
        return False


def split_dataset(
    input_path: Union[str, Path],
    train_path: Union[str, Path],
    val_path: Union[str, Path],
    train_ratio: float = 0.8
) -> bool:
    """
    データセットを学習/検証に分割

    Args:
        input_path: 入力データセットパス
        train_path: 学習データ出力パス
        val_path: 検証データ出力パス
        train_ratio: 学習データの割合

    Returns:
        成功かどうか
    """
    input_path = Path(input_path)
    train_path = Path(train_path)
    val_path = Path(val_path)

    try:
        data = []

        # データ読み込み
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        # 分割
        split_idx = int(len(data) * train_ratio)
        train_data = data[:split_idx]
        val_data = data[split_idx:]

        # 保存
        for path, dataset in [(train_path, train_data), (val_path, val_data)]:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                for item in dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"Split dataset: {len(train_data)} train, {len(val_data)} val samples")
        return True

    except Exception as e:
        logger.error(f"Failed to split dataset: {e}")
        return False
