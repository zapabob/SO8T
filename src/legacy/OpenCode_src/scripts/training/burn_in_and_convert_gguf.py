#!/usr/bin/env python3
"""
SO8T焼き込み + GGUF変換スクリプト
学習済みモデルにSO8T焼き込みを適用し、GGUF形式に変換
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from so8t_core import burn_in_phi4_model

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def burn_in_model(
    model_path: str,
    output_path: str,
):
    """
    SO8T焼き込みを適用
    
    Args:
        model_path: 学習済みモデルパス
        output_path: 焼き込み済みモデル保存先
    """
    logger.info(f"[STEP 1] Loading trained model from {model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # メモリ節約のためCPU
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    logger.info("[STEP 2] Applying SO8T burn-in...")
    
    summary = burn_in_phi4_model(
        model=model,
        save_path=Path(output_path),
        verify_threshold=1e-5,
    )
    
    logger.info(f"[BURN_IN] Summary: {summary['success_rate']*100:.1f}% successful")
    
    return output_path


def convert_to_gguf(
    model_path: str,
    output_path: str,
    quantization: str = "F16",
):
    """
    GGUF形式に変換
    
    Args:
        model_path: 焼き込み済みモデルパス
        output_path: GGUF出力パス
        quantization: 量子化タイプ（F16, Q4_K_M, Q8_0等）
    """
    logger.info(f"[STEP 3] Converting to GGUF format...")
    
    # llama.cppのconvert_hf_to_gguf.pyを使用
    convert_script = Path("external/llama.cpp-master/convert_hf_to_gguf.py")
    
    if not convert_script.exists():
        logger.error(f"[ERROR] convert_hf_to_gguf.py not found: {convert_script}")
        logger.info("[INFO] Please ensure llama.cpp is cloned in external/")
        return None
    
    # F16変換
    output_f16 = Path(output_path) / f"{Path(model_path).name}_f16.gguf"
    
    cmd = [
        sys.executable,
        str(convert_script),
        str(model_path),
        "--outfile", str(output_f16),
        "--outtype", "f16",
    ]
    
    logger.info(f"[CONVERT] Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("[CONVERT] F16 conversion successful")
        logger.debug(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"[ERROR] F16 conversion failed: {e}")
        logger.error(e.stderr)
        return None
    
    # 量子化（オプション）
    if quantization != "F16":
        logger.info(f"[STEP 4] Quantizing to {quantization}...")
        
        quantize_bin = Path("external/llama.cpp-master/build/bin/Release/quantize.exe")
        
        if not quantize_bin.exists():
            logger.warning(f"[WARNING] quantize.exe not found: {quantize_bin}")
            logger.info("[INFO] Skipping quantization, F16 model saved")
            return str(output_f16)
        
        output_quant = Path(output_path) / f"{Path(model_path).name}_{quantization.lower()}.gguf"
        
        cmd_quant = [
            str(quantize_bin),
            str(output_f16),
            str(output_quant),
            quantization,
        ]
        
        logger.info(f"[QUANTIZE] Running: {' '.join(cmd_quant)}")
        
        try:
            result = subprocess.run(cmd_quant, check=True, capture_output=True, text=True)
            logger.info(f"[QUANTIZE] {quantization} quantization successful")
            logger.debug(result.stdout)
            return str(output_quant)
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Quantization failed: {e}")
            logger.error(e.stderr)
            return str(output_f16)
    
    return str(output_f16)


def main():
    parser = argparse.ArgumentParser(description="Burn-in SO8T and convert to GGUF")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/phi4_so8t_japanese_final",
        help="Path to trained model"
    )
    parser.add_argument(
        "--burn_in_output",
        type=str,
        default="models/phi4_so8t_baked",
        help="Output path for burned-in model"
    )
    parser.add_argument(
        "--gguf_output",
        type=str,
        default="models/phi4_so8t_baked",
        help="Output path for GGUF files"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="F16",
        choices=["F16", "Q4_K_M", "Q8_0", "Q5_K_M"],
        help="Quantization type"
    )
    parser.add_argument(
        "--skip_burn_in",
        action="store_true",
        help="Skip burn-in (model already burned)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("SO8T Burn-in + GGUF Conversion Pipeline")
    logger.info("=" * 70)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Burn-in output: {args.burn_in_output}")
    logger.info(f"GGUF output: {args.gguf_output}")
    logger.info(f"Quantization: {args.quantization}")
    logger.info("=" * 70)
    
    # 焼き込み
    if not args.skip_burn_in:
        burned_model_path = burn_in_model(
            model_path=args.model_path,
            output_path=args.burn_in_output,
        )
    else:
        burned_model_path = args.model_path
        logger.info(f"[SKIP] Using pre-burned model: {burned_model_path}")
    
    # GGUF変換
    gguf_path = convert_to_gguf(
        model_path=burned_model_path,
        output_path=args.gguf_output,
        quantization=args.quantization,
    )
    
    if gguf_path:
        logger.info("[SUCCESS] Pipeline completed!")
        logger.info(f"GGUF model saved to: {gguf_path}")
    else:
        logger.error("[FAILED] Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

