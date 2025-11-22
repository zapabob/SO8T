#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
エンコーディング統一ユーティリティ

UTF-8エンコーディング統一、ファイル読み書き時のエンコーディング自動検出と変換
"""

import json
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any
import warnings
warnings.filterwarnings("ignore")

# chardetライブラリのインポート（エンコーディング検出用）
try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False
    logging.warning("chardet not available. Install with: pip install chardet")

# ロギング設定
logger = logging.getLogger(__name__)


def detect_encoding(file_path: Path, sample_size: int = 10000) -> str:
    """
    ファイルのエンコーディングを自動検出
    
    Args:
        file_path: ファイルパス
        sample_size: 検出に使用するサンプルサイズ（バイト）
    
    Returns:
        検出されたエンコーディング名（デフォルト: utf-8）
    """
    if not file_path.exists():
        logger.warning(f"[ENCODING] File not found: {file_path}")
        return 'utf-8'
    
    try:
        # ファイルの先頭部分を読み込んでエンコーディングを検出
        with open(file_path, 'rb') as f:
            raw_data = f.read(sample_size)
        
        if CHARDET_AVAILABLE:
            result = chardet.detect(raw_data)
            detected_encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0.0)
            
            # 信頼度が低い場合はUTF-8を試す
            if confidence < 0.7:
                logger.debug(f"[ENCODING] Low confidence ({confidence:.2f}), trying UTF-8")
                detected_encoding = 'utf-8'
            
            logger.debug(f"[ENCODING] Detected encoding: {detected_encoding} (confidence: {confidence:.2f})")
            return detected_encoding
        else:
            # chardetが利用できない場合はUTF-8を試す
            logger.debug("[ENCODING] chardet not available, trying UTF-8")
            return 'utf-8'
    
    except Exception as e:
        logger.warning(f"[ENCODING] Failed to detect encoding: {e}, using UTF-8")
        return 'utf-8'


def convert_to_utf8(file_path: Path, output_path: Optional[Path] = None, create_backup: bool = True) -> bool:
    """
    ファイルをUTF-8に変換
    
    Args:
        file_path: 入力ファイルパス
        output_path: 出力ファイルパス（Noneの場合は上書き）
        create_backup: バックアップを作成するか
    
    Returns:
        変換成功フラグ
    """
    if not file_path.exists():
        logger.error(f"[ENCODING] File not found: {file_path}")
        return False
    
    try:
        # エンコーディングを検出
        detected_encoding = detect_encoding(file_path)
        
        # バックアップ作成
        if create_backup and output_path is None:
            backup_path = file_path.with_suffix(file_path.suffix + '.backup')
            shutil.copy2(file_path, backup_path)
            logger.info(f"[ENCODING] Backup created: {backup_path}")
        
        # ファイルを読み込んでUTF-8に変換
        with open(file_path, 'r', encoding=detected_encoding, errors='replace') as f:
            content = f.read()
        
        # UTF-8で書き込み
        output_file = output_path or file_path
        with open(output_file, 'w', encoding='utf-8', errors='strict') as f:
            f.write(content)
        
        logger.info(f"[ENCODING] Converted {file_path} from {detected_encoding} to UTF-8")
        return True
    
    except UnicodeDecodeError as e:
        logger.error(f"[ENCODING] Failed to decode file {file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"[ENCODING] Failed to convert file {file_path}: {e}")
        return False


def safe_read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """
    エンコーディングエラーを回避してJSONLを読み込み
    
    Args:
        file_path: JSONLファイルパス
    
    Returns:
        読み込んだサンプルのリスト
    """
    samples = []
    
    if not file_path.exists():
        logger.warning(f"[ENCODING] File not found: {file_path}")
        return samples
    
    # エンコーディングを検出
    detected_encoding = detect_encoding(file_path)
    
    # 複数のエンコーディングを試す
    encodings_to_try = [detected_encoding, 'utf-8', 'utf-8-sig', 'shift_jis', 'cp932', 'latin-1']
    
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sample = json.loads(line)
                        samples.append(sample)
                    except json.JSONDecodeError as e:
                        logger.debug(f"[ENCODING] JSON decode error at line {line_num}: {e}")
                        continue
                    except Exception as e:
                        logger.debug(f"[ENCODING] Error processing line {line_num}: {e}")
                        continue
            
            logger.info(f"[ENCODING] Successfully read {len(samples)} samples from {file_path} using {encoding}")
            return samples
        
        except UnicodeDecodeError:
            logger.debug(f"[ENCODING] Failed to read with {encoding}, trying next encoding...")
            samples = []  # リセット
            continue
        except Exception as e:
            logger.debug(f"[ENCODING] Error reading with {encoding}: {e}")
            samples = []  # リセット
            continue
    
    logger.warning(f"[ENCODING] Failed to read file {file_path} with all encodings")
    return samples


def safe_write_jsonl(file_path: Path, data: List[Dict[str, Any]], append: bool = False) -> bool:
    """
    UTF-8でJSONLを書き込み
    
    Args:
        file_path: 出力ファイルパス
        data: 書き込むデータのリスト
        append: 追記モードか
    
    Returns:
        書き込み成功フラグ
    """
    try:
        # ディレクトリを作成
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ファイルを開く（追記モードまたは新規作成）
        mode = 'a' if append and file_path.exists() else 'w'
        
        with open(file_path, mode, encoding='utf-8', errors='strict') as f:
            for sample in data:
                json_line = json.dumps(sample, ensure_ascii=False)
                f.write(json_line + '\n')
        
        logger.info(f"[ENCODING] Successfully wrote {len(data)} samples to {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"[ENCODING] Failed to write file {file_path}: {e}")
        return False


def fix_directory_encoding(directory_path: Path, backup_dir: Optional[Path] = None, file_pattern: str = "*.jsonl") -> Dict[str, Any]:
    """
    ディレクトリ内の全ファイルをUTF-8に変換
    
    Args:
        directory_path: ディレクトリパス
        backup_dir: バックアップディレクトリ（Noneの場合は各ファイルの.backup）
        file_pattern: 処理するファイルパターン
    
    Returns:
        変換結果の辞書
    """
    directory_path = Path(directory_path)
    
    if not directory_path.exists():
        logger.error(f"[ENCODING] Directory not found: {directory_path}")
        return {
            'success': False,
            'total_files': 0,
            'converted_files': 0,
            'failed_files': 0,
            'errors': []
        }
    
    # バックアップディレクトリを作成
    if backup_dir:
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
    
    # ファイルを検索
    files_to_process = list(directory_path.rglob(file_pattern))
    
    logger.info(f"[ENCODING] Found {len(files_to_process)} files to process")
    
    results = {
        'success': True,
        'total_files': len(files_to_process),
        'converted_files': 0,
        'failed_files': 0,
        'errors': []
    }
    
    for file_path in files_to_process:
        try:
            # バックアップパスを決定
            if backup_dir:
                relative_path = file_path.relative_to(directory_path)
                backup_path = backup_dir / relative_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, backup_path)
            else:
                backup_path = None
            
            # 変換実行
            success = convert_to_utf8(file_path, create_backup=(backup_path is None))
            
            if success:
                results['converted_files'] += 1
            else:
                results['failed_files'] += 1
                results['errors'].append(str(file_path))
        
        except Exception as e:
            results['failed_files'] += 1
            results['errors'].append(f"{file_path}: {str(e)}")
            logger.error(f"[ENCODING] Failed to process {file_path}: {e}")
    
    logger.info(f"[ENCODING] Conversion completed: {results['converted_files']}/{results['total_files']} files converted")
    
    return results


def validate_utf8_file(file_path: Path) -> bool:
    """
    ファイルがUTF-8で正しく読み込めるか検証
    
    Args:
        file_path: ファイルパス
    
    Returns:
        検証成功フラグ
    """
    if not file_path.exists():
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='strict') as f:
            f.read()
        return True
    except UnicodeDecodeError:
        return False
    except Exception:
        return False


































































































































