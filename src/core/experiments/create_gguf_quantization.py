#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS v2.5モデルをGGUF形式に量子化
"""

import os
import subprocess
import logging
from pathlib import Path

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_gguf_quantization():
    """AEGIS v2.5モデルをGGUF量子化"""

    # パス設定
    model_path = "models/aegis_v25_final"  # LoRAアダプタ
    base_model_path = "AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp"  # ベースモデル
    output_path = "models/aegis_v25_so8t_gguf.gguf"

    try:
        logger.info("Starting GGUF quantization for AEGIS v2.5...")

        # llama.cppが利用可能かチェック
        try:
            import llama_cpp
            llama_cpp_available = True
            logger.info("llama.cpp Python bindings available")
        except ImportError:
            llama_cpp_available = False
            logger.warning("llama.cpp Python bindings not available, using convert.py")

        if llama_cpp_available:
            # llama.cpp Pythonバインディングを使用
            logger.info("Using llama.cpp Python bindings for quantization")

            from llama_cpp import Llama
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            import torch

            # モデル読み込みとマージ
            logger.info("Loading and merging model...")
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map="cpu",  # CPUで処理
                trust_remote_code=True
            )

            model = PeftModel.from_pretrained(base_model, model_path)
            merged_model = model.merge_and_unload()

            # GGUF変換
            logger.info("Converting to GGUF format...")
            # llama.cppのGGUF変換は複雑なので、convert.pyを使用

        # convert.pyを使用（推奨）
        logger.info("Using llama.cpp convert.py for GGUF quantization")

        # convert.pyが存在するかチェック
        convert_py_paths = [
            "external/llama.cpp-master/convert.py",
            "llama.cpp/convert.py",
            "/usr/local/bin/convert.py"
        ]

        convert_py_path = None
        for path in convert_py_paths:
            if os.path.exists(path):
                convert_py_path = path
                break

        if not convert_py_path:
            logger.error("convert.py not found. Please install llama.cpp and ensure convert.py is available")
            return False

        # まずモデルをSafeTensor形式に変換（必要に応じて）
        logger.info("Preparing model for GGUF conversion...")

        # GGUF変換コマンド
        cmd = [
            "python", convert_py_path,
            "--model", model_path,
            "--tokenizer", base_model_path,
            "--output", output_path,
            "--quantization", "Q8_0",  # 高品質量子化
            "--trust-remote-code"
        ]

        logger.info(f"Running GGUF conversion: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )

        if result.returncode == 0:
            logger.info("[OK] GGUF quantization completed successfully!")
            logger.info(f"📄 Output: {output_path}")

            # ファイルサイズ確認
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024**3)  # GB
                logger.info(f"[STATS] File size: {file_size:.2f} GB")
            else:
                logger.error("Output file not found")

            return output_path
        else:
            logger.error(f"GGUF conversion failed: {result.stderr}")
            logger.info(f"STDOUT: {result.stdout}")
            return False

    except Exception as e:
        logger.error(f"GGUF quantization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def upload_gguf_to_hf(gguf_path):
    """GGUFファイルをHFにアップロード"""

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        repo_id = "zapabobouj/AEGIS-v2.5-SO8T-Quadrality-imatrix"

        logger.info(f"Uploading GGUF to {repo_id}...")

        # GGUFファイルをアップロード
        api.upload_file(
            path_or_fileobj=gguf_path,
            path_in_repo="aegis_v25_so8t_gguf.gguf",
            repo_id=repo_id,
            commit_message="Upload AEGIS v2.5 SO8T GGUF quantized model"
        )

        logger.info("[OK] GGUF upload completed!")
        logger.info(f"📍 GGUF URL: https://huggingface.co/{repo_id}/resolve/main/aegis_v25_so8t_gguf.gguf")

        return True

    except Exception as e:
        logger.error(f"GGUF upload failed: {e}")
        return False

if __name__ == "__main__":
    # GGUF量子化実行
    gguf_path = create_gguf_quantization()

    if gguf_path:
        # HFアップロード
        upload_success = upload_gguf_to_hf(gguf_path)

        if upload_success:
            print("[DONE] AEGIS v2.5 GGUF model uploaded successfully!")
            print(f"📍 URL: https://huggingface.co/zapabobouj/AEGIS-v2.5-SO8T-Quadrality-imatrix")
        else:
            print("[NG] GGUF upload failed")
    else:
        print("[NG] GGUF quantization failed")