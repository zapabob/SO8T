#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T Rolling Checkpoint System Test
ローリングチェックポイントシステムのテスト
"""

import os
import time
import torch
import torch.nn as nn
from pathlib import Path
import logging
from utils.checkpoint_manager import RollingCheckpointManager

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DummyModel(nn.Module):
    """テスト用のダミーモデル"""

    def __init__(self, size=100):
        super().__init__()
        self.linear = nn.Linear(size, size)
        self.layer_norm = nn.LayerNorm(size)

    def forward(self, x):
        return self.layer_norm(self.linear(x))

class DummyTokenizer:
    """テスト用のダミートークナイザー"""

    def save_pretrained(self, path):
        """トークナイザーを保存"""
        import json
        tokenizer_data = {
            "vocab_size": 50000,
            "model_max_length": 2048,
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|end_of_text|>",
            "pad_token": "<|pad|>",
            "unk_token": "<|unk|>"
        }

        with open(Path(path) / "tokenizer.json", 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, indent=2, ensure_ascii=False)

def test_rolling_checkpoint_system():
    """ローリングチェックポイントシステムをテスト"""

    logger.info("=== SO8T Rolling Checkpoint System Test ===")

    # テストディレクトリ
    test_dir = Path("test_checkpoints")
    test_dir.mkdir(exist_ok=True)

    # ローリングチェックポイントマネージャー作成
    checkpoint_manager = RollingCheckpointManager(
        base_dir=test_dir,
        max_keep=5,
        save_interval_sec=1  # テスト用に1秒
    )

    # ダミーモデルとトークナイザー
    model = DummyModel()
    tokenizer = DummyTokenizer()

    logger.info("Testing checkpoint saving and rolling...")

    # 10回チェックポイントを保存（5個以上になるはず）
    for i in range(10):
        logger.info(f"Saving checkpoint {i+1}/10...")

        # 適当なステップ情報を追加
        step_info = f"test_epoch_{i+1:02d}_step_{i*100:04d}"

        # チェックポイント保存
        checkpoint_manager.save_checkpoint(model, tokenizer, step_info)

        # 保存タイミングを待つ
        time.sleep(1.1)  # save_interval_secより長く待つ

        # 現在のチェックポイント数を確認
        checkpoints = list(test_dir.glob("ckpt_*"))
        logger.info(f"Current checkpoints: {len(checkpoints)}")

        # 5個以内に収まっていることを確認
        if len(checkpoints) > 5:
            logger.error(f"Too many checkpoints: {len(checkpoints)} (should be <= 5)")
            return False

    # 最終チェックポイント数の確認
    final_checkpoints = list(test_dir.glob("ckpt_*"))
    logger.info(f"Final checkpoint count: {len(final_checkpoints)}")

    if len(final_checkpoints) != 5:
        logger.error(f"Expected 5 checkpoints, got {len(final_checkpoints)}")
        return False

    # 最新チェックポイントの取得テスト
    latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
    if latest_checkpoint is None:
        logger.error("Latest checkpoint is None")
        return False

    logger.info(f"Latest checkpoint: {latest_checkpoint}")

    # チェックポイントの内容確認
    if not latest_checkpoint.exists():
        logger.error(f"Latest checkpoint does not exist: {latest_checkpoint}")
        return False

    # ファイルリスト表示
    logger.info("Final checkpoint files:")
    for cp in sorted(final_checkpoints, key=lambda x: x.stat().st_mtime, reverse=True):
        mtime = time.ctime(cp.stat().st_mtime)
        logger.info(f"  {cp.name} (modified: {mtime})")

    logger.info("[OK] Rolling Checkpoint System Test PASSED")

    # クリーンアップ
    logger.info("Cleaning up test directory...")
    import shutil
    shutil.rmtree(test_dir)

    return True

def test_execution_checkpoint():
    """実行チェックポイントのテスト"""

    logger.info("=== Execution Checkpoint Test ===")

    from scripts.automation.so8t_auto_pipeline_runner import SO8TAutoPipelineRunner

    # テスト用ランナー作成
    runner = SO8TAutoPipelineRunner(
        pipeline_script="scripts/training/train_borea_phi35_so8t_ppo.py",
        dataset_path="data/so8t_quadruple_dataset.jsonl",
        output_dir="test_outputs",
        checkpoint_dir="test_checkpoints",
        interval_minutes=1,  # テスト用
        max_checkpoints=3,
        max_iterations=1
    )

    # システムチェックテスト
    logger.info("Testing system readiness check...")
    system_ready = runner._check_system_ready()
    logger.info(f"System ready: {system_ready}")

    # 実行チェックポイント保存テスト
    logger.info("Testing execution checkpoint saving...")
    runner._save_execution_checkpoint("test_execution_0001", "success")

    # チェックポイントファイルの存在確認
    checkpoint_files = list(Path("test_checkpoints").glob("execution_*.json"))
    if checkpoint_files:
        logger.info(f"[OK] Execution checkpoint saved: {checkpoint_files[0]}")

        # 内容確認
        import json
        with open(checkpoint_files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"Execution data: {data.keys()}")
    else:
        logger.error("[NG] Execution checkpoint not found")
        return False

    # クリーンアップ
    import shutil
    if Path("test_outputs").exists():
        shutil.rmtree("test_outputs")
    if Path("test_checkpoints").exists():
        shutil.rmtree("test_checkpoints")

    logger.info("[OK] Execution Checkpoint Test PASSED")
    return True

def main():
    """メイン実行関数"""

    logger.info("Starting SO8T Rolling Checkpoint System Tests...")

    # テスト1: ローリングチェックポイントシステム
    test1_passed = test_rolling_checkpoint_system()

    # テスト2: 実行チェックポイント
    test2_passed = test_execution_checkpoint()

    # 結果表示
    logger.info("=== Test Results ===")
    logger.info(f"Rolling Checkpoint Test: {'PASSED' if test1_passed else 'FAILED'}")
    logger.info(f"Execution Checkpoint Test: {'PASSED' if test2_passed else 'FAILED'}")

    if test1_passed and test2_passed:
        logger.info("[DONE] All tests PASSED! SO8T Rolling Checkpoint System is ready.")
        return 0
    else:
        logger.error("[NG] Some tests FAILED. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())





