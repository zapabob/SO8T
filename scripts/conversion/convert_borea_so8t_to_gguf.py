#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Borea-Phi-3.5 SO8T/thinking GGUF変換スクリプト

焼き込み済みモデルをGGUF形式に変換（F16, Q8_0, Q4_K_M）
"""

import os
import sys
import subprocess
import logging
import argparse
from pathlib import Path
from typing import List
import warnings
warnings.filterwarnings("ignore")

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/convert_borea_so8t_to_gguf.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def convert_to_gguf(
    model_path: Path,
    output_dir: Path,
    model_name: str = "borea_phi35_so8t_thinking",
    quantization_types: List[str] = ["f16", "q8_0", "q4_k_m"]
) -> bool:
    """
    Hugging FaceモデルをGGUF形式に変換
    
    Args:
        model_path: 焼き込み済みモデルパス
        output_dir: 出力ディレクトリ
        model_name: モデル名
        quantization_types: 量子化タイプのリスト
    
    Returns:
        成功したかどうか
    """
    logger.info("="*80)
    logger.info("GGUF Conversion")
    logger.info("="*80)
    logger.info(f"Input model: {model_path}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Quantization types: {quantization_types}")
    
    # llama.cppのconvertスクリプトパス
    convert_script = PROJECT_ROOT / "external" / "llama.cpp-master" / "convert-hf-to-gguf.py"
    
    if not convert_script.exists():
        logger.error(f"Convert script not found: {convert_script}")
        logger.error("Please ensure llama.cpp is cloned to external/llama.cpp-master")
        return False
    
    # 出力ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    
    for quant_type in quantization_types:
        logger.info(f"\n[CONVERT] Converting to {quant_type}...")
        
        output_file = output_dir / f"{model_name}_{quant_type}.gguf"
        
        # 変換コマンド
        cmd = [
            "py", "-3",
            str(convert_script),
            str(model_path),
            "--outfile", str(output_file),
            "--outtype", quant_type
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            if output_file.exists():
                file_size_mb = output_file.stat().st_size / (1024 * 1024)
                logger.info(f"[OK] {quant_type}: {output_file} ({file_size_mb:.1f} MB)")
                success_count += 1
            else:
                logger.warning(f"[WARNING] {quant_type}: Output file not found")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] {quant_type}: Conversion failed")
            logger.error(f"Command: {' '.join(cmd)}")
            logger.error(f"Error: {e.stderr}")
            continue
        except Exception as e:
            logger.error(f"[ERROR] {quant_type}: Unexpected error: {e}")
            continue
    
    logger.info(f"\n[SUCCESS] Converted {success_count}/{len(quantization_types)} quantization types")
    return success_count > 0


def main():
    parser = argparse.ArgumentParser(
        description="Convert baked Borea-Phi-3.5 SO8T model to GGUF"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Baked model path (Hugging Face format)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("D:/webdataset/gguf_models/borea_phi35_so8t_thinking"),
        help="Output directory for GGUF files"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="borea_phi35_so8t_thinking",
        help="Model name for output files"
    )
    parser.add_argument(
        "--quantization-types",
        type=str,
        nargs="+",
        default=["f16", "q8_0", "q4_k_m"],
        help="Quantization types (f16, q8_0, q4_k_m, etc.)"
    )
    
    args = parser.parse_args()
    
    success = convert_to_gguf(
        model_path=args.model_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        quantization_types=args.quantization_types
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()









