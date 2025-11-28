#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Borea-Phi3.5-instinct-jp GGUF変換スクリプト
Borea-Phi3.5-instinct-jp GGUF Conversion Script

ABCテストのAモデルとして使用するGGUF版を生成
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_borea_phi35_to_gguf(model_path: str = "models/Borea-Phi-3.5-mini-Instruct-Jp",
                               output_dir: str = "D:/webdataset/gguf_models",
                               quantization: str = "f16"):
    """
    Borea-Phi3.5-instinct-jpをGGUF形式に変換

    Args:
        model_path: 変換元のモデルパス
        output_dir: 出力ディレクトリ
        quantization: 量子化タイプ (f16/bf16, q8_0, q4_k_m)
    """

    logger.info("[GGUF] Starting Borea-Phi3.5-instinct-jp to GGUF conversion...")

    # パス解決
    model_path = Path(model_path)
    if not model_path.is_absolute():
        # プロジェクトルートからの相対パス
        project_root = Path(__file__).parent.parent.parent
        model_path = project_root / model_path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # モデル存在確認
    if not model_path.exists():
        logger.error(f"[ERROR] Model path does not exist: {model_path}")
        # 別の場所も確認
        alt_paths = [
            Path("models/Borea-Phi-3.5-mini-Instruct-Jp"),
            Path("D:/webdataset/models/Borea-Phi-3.5-mini-Instruct-Jp"),
            Path("models/Borea-Phi-3.5-mini-Instruct-Jp"),
            Path("D:/webdataset/models/final/borea_phi35_so8t"),
            Path("models/Borea-Phi-3.5-mini-Instruct-Jp"),
            Path("D:/webdataset/models/Borea-Phi-3.5-mini-Instruct-Jp")
        ]

        for alt_path in alt_paths:
            if alt_path.exists():
                logger.info(f"[INFO] Found model at alternative path: {alt_path}")
                model_path = alt_path
                break
        else:
            logger.error("[ERROR] Could not find Borea-Phi3.5-instinct-jp model in any expected location")
            return None

    # llama.cppのGGUF変換スクリプトパス
    project_root = Path(__file__).parent.parent.parent
    llama_cpp_dir = project_root / "external" / "llama.cpp-master"
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        logger.error(f"[ERROR] llama.cpp convert script not found: {convert_script}")
        return None

    # 出力ファイル名
    model_name = "borea_phi35_instruct_jp"
    output_file = output_dir / f"{model_name}_{quantization}.gguf"

    logger.info(f"[GGUF] Converting model: {model_path}")
    logger.info(f"[GGUF] Output file: {output_file}")
    logger.info(f"[GGUF] Quantization: {quantization}")

    # 変換コマンド実行
    cmd = [
        sys.executable,
        str(convert_script),
        str(model_path),
        "--outfile", str(output_file),
        "--outtype", quantization
    ]

    logger.info(f"[GGUF] Running command: {' '.join(cmd)}")

    try:
        # 作業ディレクトリをllama.cppに変更して実行
        result = subprocess.run(
            cmd,
            cwd=str(llama_cpp_dir),
            capture_output=True,
            text=True,
            encoding='utf-8'
        )

        if result.returncode == 0:
            logger.info("[SUCCESS] GGUF conversion completed successfully!")
            logger.info(f"[SUCCESS] Output file: {output_file}")

            # ファイルサイズ確認
            if output_file.exists():
                size_mb = output_file.stat().st_size / (1024 * 1024)
                logger.info(f"[SUCCESS] File size: {size_mb:.1f} MB")

            return str(output_file)
        else:
            logger.error("[ERROR] GGUF conversion failed!")
            logger.error(f"[ERROR] STDOUT: {result.stdout}")
            logger.error(f"[ERROR] STDERR: {result.stderr}")
            return None

    except Exception as e:
        logger.error(f"[ERROR] GGUF conversion exception: {e}")
        return None


def create_abc_test_model_config(gguf_path: str) -> dict:
    """
    ABCテスト用のモデル設定を作成

    Args:
        gguf_path: 生成されたGGUFファイルのパス

    Returns:
        ABCテスト用のモデル設定辞書
    """

    model_config = {
        'modela': {
            'path': gguf_path,
            'type': 'gguf',
            'description': 'Borea-Phi3.5-instruct-jp (GGUF Q8_0) - ABC Test Model A',
            'source': 'microsoft/phi-3.5-mini-instruct (Japanese fine-tuned)',
            'quantization': 'Q8_0',
            'context_length': 4096,
            'architecture': 'Phi-3.5 Transformer'
        }
    }

    # 設定ファイル保存
    config_file = Path("configs/abc_test_model_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        import json
        json.dump(model_config, f, indent=2, ensure_ascii=False)

    logger.info(f"[CONFIG] ABC test model config saved to {config_file}")

    return model_config


def verify_gguf_model(gguf_path: str) -> bool:
    """
    生成されたGGUFモデルの検証

    Args:
        gguf_path: 検証するGGUFファイルのパス

    Returns:
        検証成功ならTrue
    """

    logger.info(f"[VERIFY] Verifying GGUF model: {gguf_path}")

    try:
        # llama.cppでモデル読み込みテスト
        from llama_cpp import Llama

        llm = Llama(
            model_path=gguf_path,
            n_ctx=512,  # 小さなコンテキストでテスト
            n_threads=1,
            verbose=False
        )

        # 簡単な推論テスト
        test_prompt = "こんにちは、今日は良い天気ですね。"
        response = llm(test_prompt, max_tokens=10, echo=False)

        if response and 'choices' in response and len(response['choices']) > 0:
            generated_text = response['choices'][0]['text']
            logger.info(f"[VERIFY] Test generation successful: {generated_text[:50]}...")
            return True
        else:
            logger.error("[VERIFY] Test generation failed - no response")
            return False

    except Exception as e:
        logger.error(f"[VERIFY] GGUF model verification failed: {e}")
        return False


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Convert Borea-Phi3.5-instruct-jp to GGUF format for ABC testing"
    )
    parser.add_argument(
        '--model_path',
        default='models/Borea-Phi-3.5-mini-Instruct-Jp',
        help='Path to Borea-Phi3.5-instruct-jp model'
    )
    parser.add_argument(
        '--output_dir',
        default='D:/webdataset/gguf_models',
        help='Output directory for GGUF files'
    )
    parser.add_argument(
        '--quantization',
        default='q8_0',
        choices=['f16', 'q8_0', 'q4_k_m'],
        help='Quantization type'
    )
    parser.add_argument(
        '--create_config',
        action='store_true',
        help='Create ABC test model configuration file'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify the generated GGUF model'
    )

    args = parser.parse_args()

    # GGUF変換実行
    gguf_path = convert_borea_phi35_to_gguf(
        model_path=args.model_path,
        output_dir=args.output_dir,
        quantization=args.quantization
    )

    if not gguf_path:
        logger.error("[FAILED] GGUF conversion failed!")
        sys.exit(1)

    # モデル検証
    if args.verify:
        if verify_gguf_model(gguf_path):
            logger.info("[SUCCESS] GGUF model verification passed!")
        else:
            logger.error("[FAILED] GGUF model verification failed!")
            sys.exit(1)

    # ABCテスト設定作成
    if args.create_config:
        model_config = create_abc_test_model_config(gguf_path)
        logger.info("[SUCCESS] ABC test model configuration created!")

    logger.info(f"[SUCCESS] Borea-Phi3.5-instruct-jp GGUF conversion completed!")
    logger.info(f"[SUCCESS] GGUF file: {gguf_path}")

    # オーディオ通知
    try:
        import winsound
        winsound.PlaySound(r"C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav", winsound.SND_FILENAME)
    except:
        print('\a')


if __name__ == '__main__':
    main()
