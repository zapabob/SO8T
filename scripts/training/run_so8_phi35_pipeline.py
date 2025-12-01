#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO(8) Phi-3.5 Complete Pipeline
SO(8)アダプター学習からGGUF変換までの一括実行スクリプト

実行順序：
1. SO(8)アダプター注入と学習
2. 標準LoRA形式への変換
3. GGUF変換
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import subprocess
import time

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_training(model_path: str, dataset_path: str, output_path: str, **kwargs):
    """SO(8)アダプタートレーニング実行"""
    logger.info("=== Phase 1: SO(8) Adapter Training ===")

    cmd = [
        sys.executable, "scripts/training/train_so8_phi35_adapter.py",
        "--model_path", model_path,
        "--dataset_path", dataset_path,
        "--output_path", output_path,
    ]

    # オプション引数の追加
    if "num_epochs" in kwargs:
        cmd.extend(["--num_epochs", str(kwargs["num_epochs"])])
    if "batch_size" in kwargs:
        cmd.extend(["--batch_size", str(kwargs["batch_size"])])
    if "learning_rate" in kwargs:
        cmd.extend(["--learning_rate", str(kwargs["learning_rate"])])
    if "max_steps" in kwargs:
        cmd.extend(["--max_steps", str(kwargs["max_steps"])])
    if "save_steps" in kwargs:
        cmd.extend(["--save_steps", str(kwargs["save_steps"])])

    logger.info(f"Running command: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)

    if result.returncode != 0:
        logger.error("Training failed!")
        return False

    logger.info("Training completed successfully!")
    return True


def run_gguf_conversion(lora_path: str, gguf_output_path: str, quantization: str = "bf16"):
    """GGUF変換実行"""
    logger.info("=== Phase 2: GGUF Conversion ===")

    cmd = [
        sys.executable, "scripts/conversion/convert_so8_lora_to_gguf.py",
        "--lora_path", lora_path,
        "--output_path", gguf_output_path,
        "--quantization", quantization
    ]

    logger.info(f"Running command: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)

    if result.returncode != 0:
        logger.error("GGUF conversion failed!")
        return False

    logger.info("GGUF conversion completed successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(description="SO(8) Phi-3.5 Complete Pipeline")
    parser.add_argument("--model_path", type=str,
                       default="models/Borea-Phi-3.5-mini-Instruct-Jp",
                       help="Path to Phi-3.5 model")
    parser.add_argument("--dataset_path", type=str,
                       default="data/integrated/so8t_integrated_ppo_dataset_main_20251201_205340.jsonl",
                       help="Path to training dataset")
    parser.add_argument("--output_dir", type=str,
                       default="outputs/so8_phi35_training",
                       help="Output directory for training results")
    parser.add_argument("--gguf_output", type=str,
                       default="models/gguf/so8_phi35_adapter.gguf",
                       help="Output path for GGUF file")
    parser.add_argument("--quantization", type=str, default="bf16",
                       choices=["f16", "bf16", "f32", "q8_0", "q4_k_m", "q4_0"],
                       help="Quantization type for GGUF")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum training steps")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training phase (use existing LoRA model)")
    parser.add_argument("--skip_gguf", action="store_true",
                       help="Skip GGUF conversion phase")

    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # GGUF出力ディレクトリ作成
    gguf_output_path = Path(args.gguf_output)
    gguf_output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=== SO(8) Phi-3.5 Complete Pipeline Started ===")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Dataset: {args.dataset_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"GGUF: {args.gguf_output}")
    logger.info(f"Quantization: {args.quantization}")
    logger.info("=" * 50)

    start_time = time.time()

    try:
        # Phase 1: SO(8)アダプタートレーニング
        if not args.skip_training:
            success = run_training(
                model_path=args.model_path,
                dataset_path=args.dataset_path,
                output_path=str(output_dir),
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                max_steps=args.max_steps,
                save_steps=args.save_steps
            )

            if not success:
                logger.error("Pipeline failed at training phase")
                sys.exit(1)
        else:
            logger.info("Skipping training phase...")

        # Phase 2: GGUF変換
        if not args.skip_gguf:
            success = run_gguf_conversion(
                lora_path=str(output_dir),
                gguf_output_path=str(gguf_output_path),
                quantization=args.quantization
            )

            if not success:
                logger.error("Pipeline failed at GGUF conversion phase")
                sys.exit(1)
        else:
            logger.info("Skipping GGUF conversion phase...")

        # 完了
        elapsed_time = time.time() - start_time
        logger.info("=" * 50)
        logger.info("🎉 SO(8) Phi-3.5 Pipeline Completed Successfully!")
        logger.info(".2f")
        logger.info(f"GGUF Model: {args.gguf_output}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Test the GGUF model with llama.cpp")
        logger.info("2. Compare performance with baseline Phi-3.5")
        logger.info("3. Upload to Hugging Face if desired")

        # 音声通知
        try:
            subprocess.run([
                "powershell", "-ExecutionPolicy", "Bypass",
                "-File", "scripts/utils/play_audio_notification.ps1"
            ], check=True)
        except Exception as e:
            logger.warning(f"Audio notification failed: {e}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
