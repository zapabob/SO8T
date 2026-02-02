#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最終版日英表記モデルカードをHFにアップロード
ABCテストグラフを記載した完全版
"""

import os
import logging
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def upload_final_model_card():
    """最終版モデルカードをアップロード"""

    api = HfApi()
    repo_id = "zapabobouj/AEGIS-v2.5-SO8T-Quadrality-imatrix"

    logger.info("Uploading final bilingual model card with ABC test charts...")

    # 最終版モデルカードをアップロード
    try:
        api.upload_file(
            path_or_fileobj="bilingual_model_card.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message="Upload final bilingual model card with comprehensive ABC test results and visualizations"
        )
        logger.info("[UPLOAD] Final model card uploaded as README.md")
    except Exception as e:
        logger.error(f"[ERROR] Failed to upload model card: {e}")
        return False

    # ABCテスト関連ファイルをアップロード（まだアップロードされていない場合）
    additional_files = [
        ("abc_test_results.json", "abc_test_charts/abc_test_results.json"),
        ("abc_test_report.md", "abc_test_charts/abc_test_report.md"),
        ("create_abc_test_charts.py", "abc_test_charts/create_abc_test_charts.py")
    ]

    for local_file, repo_path in additional_files:
        if os.path.exists(local_file):
            try:
                api.upload_file(
                    path_or_fileobj=local_file,
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    commit_message=f"Upload ABC test data: {local_file}"
                )
                logger.info(f"[UPLOAD] {local_file} -> {repo_path}")
            except Exception as e:
                logger.warning(f"[SKIP] {local_file} may already exist or upload failed: {e}")

    logger.info("[SUCCESS] Final model card upload completed!")
    logger.info(f"[REPO] https://huggingface.co/{repo_id}")
    logger.info("[FEATURES] Bilingual documentation, ABC test charts, comprehensive analysis")

    return True

def main():
    """メイン実行関数"""
    print("[UPLOAD] Uploading Final Bilingual Model Card...")
    print("Includes: ABC test charts, statistical analysis, industry comparisons")

    success = upload_final_model_card()

    if success:
        print("\n[SUCCESS] Final model card uploaded!")
        print("[FEATURES] Features:")
        print("   - Bilingual documentation (English + Japanese)")
        print("   - Comprehensive ABC test results")
        print("   - Error bar charts with statistical significance")
        print("   - Industry standard comparisons")
        print("   - Performance ranking visualizations")
        print("\n[LINK] View at: https://huggingface.co/zapabobouj/AEGIS-v2.5-SO8T-Quadrality-imatrix")
    else:
        print("\n[ERROR] Upload failed")

if __name__ == "__main__":
    main()