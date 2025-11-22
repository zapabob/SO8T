#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四値分類セットアップテストスクリプト
"""

import os
import sys
import logging
from pathlib import Path

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoTokenizer
from scripts.training.train_four_class_classifier import FourClassDataset, FourClassModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_loading():
    """データ読み込みテスト"""
    logger.info("Testing data loading...")

    tokenizer = AutoTokenizer.from_pretrained("./models/Borea-Phi-3.5-mini-Instruct-Jp", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 小さなテストデータセットを作成
    test_data = [
        {"text": "これは許可されるべき内容です。", "label": "ALLOW"},
        {"text": "これは拒否されるべき内容です。", "label": "DENY"},
    ]

    # 一時ファイルに保存
    test_file = Path("test_data.jsonl")
    with open(test_file, 'w', encoding='utf-8') as f:
        import json
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    try:
        dataset = FourClassDataset(test_file, tokenizer, max_length=512)
        logger.info(f"Dataset loaded successfully: {len(dataset)} samples")

        # 最初のサンプルを取得
        sample = dataset[0]
        logger.info(f"Sample input_ids shape: {sample['input_ids'].shape}")
        logger.info(f"Sample labels shape: {sample['labels'].shape}")

        return True
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return False
    finally:
        # テストファイルを削除
        if test_file.exists():
            test_file.unlink()

def test_model_setup():
    """モデルセットアップテスト"""
    logger.info("Testing model setup...")

    try:
        # トークナイザー
        tokenizer = AutoTokenizer.from_pretrained("./models/Borea-Phi-3.5-mini-Instruct-Jp", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ダミーの小さなモデルでテスト（実際のモデル読み込みは時間がかかる）
        hidden_size = 768  # 小さなサイズでテスト
        num_classes = 4

        # シンプルなテストモデル
        class TestBaseModel(torch.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.linear = torch.nn.Linear(hidden_size, hidden_size)

            def forward(self, input_ids, attention_mask=None, output_hidden_states=True):
                batch_size, seq_len = input_ids.shape
                # ダミーの隠れ状態を生成
                hidden_states = torch.randn(batch_size, seq_len, hidden_size)
                return type('Output', (), {'hidden_states': [hidden_states] * 32})()

        base_model = TestBaseModel(hidden_size)
        model = FourClassModel(base_model, num_classes=num_classes, hidden_size=hidden_size)

        # テスト入力
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        model.eval()
        with torch.no_grad():
            output = model(input_ids)
            logger.info(f"Model output loss: {output['loss']}")
            logger.info(f"Model output logits shape: {output['logits'].shape}")

        # gradient_checkpointing_enableメソッドがあるかテスト
        if hasattr(model, 'gradient_checkpointing_enable'):
            logger.info("gradient_checkpointing_enable method found")
            model.gradient_checkpointing_enable()
        else:
            logger.error("gradient_checkpointing_enable method not found")

        return True

    except Exception as e:
        logger.error(f"Model setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン関数"""
    logger.info("Starting four-class classification setup tests...")

    success = True

    # データ読み込みテスト
    if not test_data_loading():
        success = False

    # モデルセットアップテスト
    if not test_model_setup():
        success = False

    if success:
        logger.info("[SUCCESS] All tests passed!")
        return 0
    else:
        logger.error("[FAILED] Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
