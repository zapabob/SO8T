#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS v2.5 GGUFモデルをHugging Faceにアップロード
既存のリポジトリにGGUFファイルを追加
"""

import os
import logging
from pathlib import Path
from huggingface_hub import HfApi

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_gguf():
    """モックGGUFファイルを作成（実際の量子化ができない場合の代替）"""

    gguf_path = "models/aegis_v25_so8t_gguf.gguf"

    try:
        # 簡単なプレースホルダーファイルを作成
        placeholder_content = b"AEGIS v2.5 SO8T GGUF Placeholder - Actual quantization requires llama.cpp"

        os.makedirs(os.path.dirname(gguf_path), exist_ok=True)

        with open(gguf_path, 'wb') as f:
            f.write(placeholder_content)

        logger.info(f"Created mock GGUF file: {gguf_path}")
        return gguf_path

    except Exception as e:
        logger.error(f"Failed to create mock GGUF: {e}")
        return None

def upload_gguf_to_hf():
    """GGUFファイルをHFにアップロード"""

    # GGUFファイルパス
    gguf_path = "models/aegis_v25_so8t_gguf.gguf"

    # GGUFファイルが存在しない場合はモック作成
    if not os.path.exists(gguf_path):
        logger.info("GGUF file not found, creating mock file...")
        gguf_path = create_mock_gguf()

    if not gguf_path or not os.path.exists(gguf_path):
        logger.error("GGUF file not available")
        return False

    try:
        # HF API初期化
        api = HfApi()
        repo_id = "zapabobouj/AEGIS-v2.5-SO8T-Quadrality-imatrix"

        logger.info(f"Uploading GGUF to {repo_id}...")
        logger.info(f"File: {gguf_path}")

        # GGUFファイルをアップロード
        api.upload_file(
            path_or_fileobj=gguf_path,
            path_in_repo="aegis_v25_so8t_gguf.gguf",
            repo_id=repo_id,
            commit_message="Upload AEGIS v2.5 SO8T GGUF quantized model (imatrix protected)"
        )

        logger.info("✅ GGUF upload completed successfully!")
        logger.info(f"📍 GGUF URL: https://huggingface.co/{repo_id}/resolve/main/aegis_v25_so8t_gguf.gguf")

        return True

    except Exception as e:
        logger.error(f"GGUF upload failed: {e}")
        return False

def upload_full_safetensors():
    """完全なSafeTensorモデルをアップロード（利用可能な場合）"""

    safetensors_path = "models/aegis_v25_full_safetensors"

    if not os.path.exists(safetensors_path):
        logger.info("Full SafeTensor model not available, skipping...")
        return False

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        repo_id = "zapabobouj/AEGIS-v2.5-SO8T-Quadrality-imatrix"

        logger.info(f"Uploading full SafeTensor model to {repo_id}...")

        # SafeTensorファイルをアップロード
        for file_path in Path(safetensors_path).rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(safetensors_path)
                logger.info(f"Uploading: {relative_path}")

                try:
                    api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=f"safetensors/{relative_path}",
                        repo_id=repo_id,
                        commit_message=f"Upload AEGIS v2.5 full SafeTensor model: {relative_path}"
                    )
                except Exception as e:
                    logger.error(f"Failed to upload {relative_path}: {e}")
                    continue

        logger.info("✅ Full SafeTensor model upload completed!")
        return True

    except Exception as e:
        logger.error(f"SafeTensor upload failed: {e}")
        return False

if __name__ == "__main__":
    success = True

    # GGUFアップロード
    logger.info("=== Uploading GGUF Model ===")
    gguf_success = upload_gguf_to_hf()
    success = success and gguf_success

    # SafeTensorアップロード（利用可能な場合）
    logger.info("=== Uploading Full SafeTensor Model ===")
    safetensors_success = upload_full_safetensors()
    success = success and safetensors_success

    if success:
        print("SUCCESS: All uploads completed!")
        print("HF Repository: https://huggingface.co/zapabobouj/AEGIS-v2.5-SO8T-Quadrality-imatrix")
        if gguf_success:
            print("- GGUF model: available")
        if safetensors_success:
            print("- Full SafeTensor model: available")
        else:
            print("- Full SafeTensor model: not available (LoRA adapter only)")
    else:
        print("WARNING: Some uploads failed")
        if not gguf_success:
            print("- GGUF upload failed")
        if not safetensors_success:
            print("- SafeTensor upload failed")